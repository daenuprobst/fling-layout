import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import networkx as nx
import torch
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fling import make_graph
from fling import data as fdata
from fling.baselines import sgd_stress
from fling.metrics import fast_D, normalized_stress
from lib import DEV, HELD_FRAC, blocks, field_curve, held_stress, split, stress_rows

# The estimator creates its tensors without an explicit device and follows torch's default, so
# this is what puts the field arm on the GPU. It also decides which generator torch.nn.Linear
# draws its init from, so the linear arm moves with it: the published table was produced with
# the default device set, and leaving it on CPU changes both arms.
if DEV.type == "cuda":
    torch.set_default_device(DEV)

# the paper's nine-graph benchmark plus the two extras the published table carried
GRAPHS = [
    "grid_400",
    "tree_400",
    "sbm_240",
    "lesmis",
    "delaunay_600",
    "dwt_1005",
    "jagmesh8",
    "cora",
    "tree_4000",
    "3elt",
    "ego-Facebook",
]
SEEDS = [0, 1, 2, 3, 4]
LADDER = [10, 25, 50, 75, 150, 300, 600, 1200, 2400]
LRS = [5e-3, 5e-2, 5e-1]  # every trained arm selects over the same grid
GAMMAS = [1e-3, 1e-2, 1e-1, 1.0]

MAIN_ARMS = ["field", "linear", "kernel", "pivotmds", "landmarkmds"]
MATCHED = MAIN_ARMS + ["pca"]
CONTROLS = ["table1k", "sparse", "s_gd2", "random"]
LABEL = {
    "random": "random",
    "field": "FlingStress",
    "linear": "linear",
    "kernel": "kernel",
    "pivotmds": "PivotMDS",
    "landmarkmds": "landmark MDS",
    "s_gd2": r"\textsc{s\_gd2} (all $N$)",
    "sparse": "sparse stress (all $N$)",
    "table1k": "table on $M$ rows",
    "pca": "PCA (no fit)",
}


def graph_of(name):
    if name == "grid_400":
        graph = make_graph("grid", 400)
    elif name == "sbm_240":
        graph = fdata._sbm((80, 80, 80))
    elif name == "lesmis":
        graph = nx.les_miserables_graph()
    elif name == "cora":
        graph = fdata.load_cora()[0]
    elif name == "ego-Facebook":
        graph = nx.read_edgelist(fdata.DATA / "facebook_combined.txt", nodetype=int)
    elif name.startswith("tree_"):
        graph = make_graph("tree", int(name.split("_")[1]))
    elif name.startswith("delaunay_"):
        graph = make_graph("delaunay", int(name.split("_")[1]))
    else:
        graph = fdata._load_mtx(name)

    return nx.convert_node_labels_to_integers(
        graph.subgraph(max(nx.connected_components(graph), key=len))
    )


def rbf(a, b, gamma):
    return torch.exp(-gamma * torch.cdist(a, b) ** 2)


def pivot_mds(dist_all):
    # Brandes & Pich: double-centre the (N x k) pivot block, top-2 singular vectors
    sq = dist_all.astype(np.float64) ** 2
    gram = -0.5 * (
        sq - sq.mean(0, keepdims=True) - sq.mean(1, keepdims=True) + sq.mean()
    )
    uvec, sval, _ = np.linalg.svd(gram, full_matrices=False)

    return uvec[:, :2] * sval[:2]


def landmark_mds(dist_all, piv_rows):
    # de Silva & Tenenbaum: classical MDS on the landmark block, the others by the closed form
    dl = dist_all[piv_rows].astype(np.float64)
    sq = dl**2
    gram = -0.5 * (
        sq - sq.mean(0, keepdims=True) - sq.mean(1, keepdims=True) + sq.mean()
    )
    vals, vecs = np.linalg.eigh(gram)
    order = np.argsort(vals)[::-1][:2]
    vals, vecs = np.maximum(vals[order], 1e-12), vecs[:, order]
    land_pos = vecs * np.sqrt(vals)

    pinv = (vecs / np.sqrt(vals)).T
    mean_sq = (dl**2).mean(1)
    out = -0.5 * (dist_all.astype(np.float64) ** 2 - mean_sq[None, :]) @ pinv.T
    out[piv_rows] = land_pos

    return out


def pca2(feat):
    centred = feat - feat.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)

    return centred @ vt[:2].T


def curve(
    kind,
    feat,
    dist_tr,
    dist_ho,
    piv_rows,
    ho_rows,
    tix,
    oos,
    dmat,
    seed,
    lr,
    iters,
    gamma=None,
):
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    ft = torch.as_tensor(feat, device=DEV)
    rows_used = np.concatenate([tix, piv_rows])
    ru = torch.as_tensor(rows_used, device=DEV)
    tgt = torch.as_tensor(dist_tr, device=DEV)[ru]
    pr = torch.arange(len(tix), len(rows_used), device=DEV)
    ho_pr = torch.as_tensor(ho_rows, device=DEV)
    train_cols = torch.arange(dist_tr.shape[1], device=DEV)

    if kind in ("sparse", "table1k"):
        # a free (N,2) table on the same columns: Ortmann's sparse stress subsamples PAIRS where
        # the field subsamples NODES. table1k is the same solver on the rows the energy sees
        table = (
            torch.randn(dist_tr.shape[0], 2, device=DEV, generator=gen) * 0.1
        ).requires_grad_()
        params = [table]

        if kind == "sparse":

            def positions(idx=None):
                return table

            tgt = torch.as_tensor(dist_tr, device=DEV)
            pr = torch.as_tensor(piv_rows, device=DEV)
        else:

            def positions(idx=None):
                return table[ru] if idx is not None else table

    elif kind == "linear":
        model = torch.nn.Linear(ft.shape[1], 2).to(DEV)
        params = list(model.parameters())

        def positions(idx=None):
            return model(ft[idx]) if idx is not None else model(ft)

    else:
        anchors = ft[torch.as_tensor(tix, device=DEV)]
        k_rows, k_all = rbf(ft[ru], anchors, gamma), rbf(ft, anchors, gamma)
        alpha = torch.randn(len(tix), 2, device=DEV, generator=gen)

        # scale so the initial embedding has an O(1) radius: otherwise the kernel starts as a
        # POINT and spends its first hundreds of steps inflating, which reads as a slower rate
        alpha = (alpha / ((k_rows @ alpha).std() + 1e-9)).requires_grad_()
        params = [alpha]

        def positions(idx=None):
            return k_rows @ alpha if idx is not None else k_all @ alpha

    opt = torch.optim.Adam(params, lr=lr)

    for step in range(1, iters + 1):
        cols = train_cols[
            torch.randint(0, len(train_cols), (32,), device=DEV, generator=gen)
        ]
        opt.zero_grad(set_to_none=True)
        stress_rows(positions(ru), tgt, pr, cols).backward()
        opt.step()

        for grp in opt.param_groups:  # each budget anneals over ITSELF
            grp["lr"] = lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * step / iters)))

    with torch.no_grad():
        full = positions()
        held = held_stress(
            full,
            torch.as_tensor(oos, device=DEV),
            torch.as_tensor(dist_ho, device=DEV),
            ho_pr,
        )
        gold = float(normalized_stress(full.cpu().numpy().astype(float), dmat))

    return held, gold


def cell(name, graph, dmat, seed, cfg):
    n = graph.number_of_nodes()
    m_eff = min(
        cfg["m_train"], n // 2
    )  # M must stay a genuine SUBSAMPLE on the small graphs
    train, tix, held_pivots, npiv_eff, oos = split(graph, m_eff, cfg["npiv"], seed)
    rows = []

    def add(arm, stress, secs, **extra):
        rows.append(
            dict(
                graph=name,
                n=n,
                m=m_eff,
                seed=seed,
                arm=arm,
                stress=float(stress),
                secs=secs,
                **extra,
            )
        )
        print(
            f"  {name:<14} seed={seed} {arm:<12} {stress:.4f}  {secs:6.1f}s {extra or ''}",
            flush=True,
        )

    # offset the stream: make_graph("delaunay", ...) draws its point set from default_rng(0), so
    # a "random" layout at seed 0 would BE the mesh's own planar embedding
    rng = np.random.default_rng(10_000 + seed)
    add("random", normalized_stress(rng.random((n, 2)), dmat), 0.0)

    tic = time.perf_counter()
    best, keep = None, None

    for lr in cfg["lrs"]:
        for iters in cfg["ladder"]:
            held, gold, est, tixe = field_curve(
                graph,
                train,
                held_pivots,
                oos,
                dmat,
                seed,
                lr,
                iters,
                npiv_eff,
                cfg["n_land"],
            )

            if best is None or held < best[0]:
                best, keep = (held, gold, lr, iters), (est, tixe)

    add("field", best[1], time.perf_counter() - tic, lr=best[2], step=best[3])

    est, tixe = keep
    feat, dist_tr, dist_ho, piv_rows, ho_rows = blocks(est, graph, held_pivots)

    for arm in ("linear", "sparse", "table1k"):
        tic = time.perf_counter()
        pick = None

        for lr in cfg["lrs"]:
            for iters in cfg["ladder"]:
                held, gold = curve(
                    arm,
                    feat,
                    dist_tr,
                    dist_ho,
                    piv_rows,
                    ho_rows,
                    tixe,
                    oos,
                    dmat,
                    seed,
                    lr,
                    iters,
                )

                if pick is None or held < pick[0]:
                    pick = (held, gold, lr, iters)

        add(arm, pick[1], time.perf_counter() - tic, lr=pick[2], step=pick[3])

    tic = time.perf_counter()
    pick = None

    for gamma in cfg["gammas"]:
        for lr in cfg["lrs"]:
            for iters in cfg["ladder"]:
                held, gold = curve(
                    "kernel",
                    feat,
                    dist_tr,
                    dist_ho,
                    piv_rows,
                    ho_rows,
                    tixe,
                    oos,
                    dmat,
                    seed,
                    lr,
                    iters,
                    gamma=gamma,
                )

                if pick is None or held < pick[0]:
                    pick = (held, gold, lr, iters, gamma)

    add(
        "kernel",
        pick[1],
        time.perf_counter() - tic,
        lr=pick[2],
        step=pick[3],
        gamma=pick[4],
    )

    # the closed forms have no use for a validation set, so their matched-information
    # configuration is ALL the columns. Each is reported at whichever block is better for it
    land = dist_tr[:, : min(cfg["n_land"], dist_tr.shape[1])]
    allcols = np.concatenate([dist_tr, dist_ho], 1)
    all_rows = np.concatenate([piv_rows, ho_rows])

    for arm, fn in (
        ("pivotmds", lambda blk, rws: pivot_mds(blk)),
        ("landmarkmds", landmark_mds),
    ):
        tic = time.perf_counter()
        cand = {}

        for tag, blk, rws in (
            ("land32", land, piv_rows[: land.shape[1]]),
            ("pivall", allcols, all_rows),
        ):
            try:
                cand[tag] = float(normalized_stress(fn(blk, rws), dmat))
            except Exception as exc:  # noqa: BLE001
                print(f"    {arm}/{tag} failed: {exc}", flush=True)

        if cand:
            pickt = min(cand, key=cand.get)
            add(arm, cand[pickt], time.perf_counter() - tic, block=pickt, both=cand)

    tic = time.perf_counter()
    add(
        "s_gd2",
        normalized_stress(sgd_stress(graph, seed=seed), dmat),
        time.perf_counter() - tic,
    )

    # the zero-parameter control: two leading directions of the same standardised block
    tic = time.perf_counter()
    add(
        "pca",
        normalized_stress(pca2(feat.astype(np.float64)), dmat),
        time.perf_counter() - tic,
    )

    return rows


def cfg_hash(cfg):
    blob = json.dumps(cfg, sort_keys=True).encode()

    return hashlib.sha1(blob).hexdigest()[:8]


def collect(names, seeds, cfg, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = cfg_hash(cfg)
    rows = []

    for name in names:
        graph = dmat = None

        for seed in seeds:
            path = cache_dir / f"{name}__s{seed}__{tag}.json"

            if path.exists():
                rows += json.loads(path.read_text())
                print(f"  {name:<14} seed={seed} cached", flush=True)
                continue

            if graph is None:
                graph = graph_of(name)
                dmat = fast_D(graph)
                print(
                    f"\n=== {name}  n={graph.number_of_nodes()}  "
                    f"e={graph.number_of_edges()}",
                    flush=True,
                )

            got = cell(name, graph, dmat, seed, cfg)
            path.write_text(json.dumps(got, indent=1))
            rows += got

    return rows


def fmt(vals, bold=False, prec=3):
    txt = f"{np.mean(vals):.{prec}f}"

    if len(vals) > 1:
        txt += f"\\,\\pm\\,{np.std(vals, ddof=1):.{prec}f}"

    return f"$\\mathbf{{{txt}}}$" if bold else f"${txt}$"


def write_lean(graphs, agg, ms, arms, path, bold_best):
    lines = [
        "\\begin{tabular}{lr" + "r" * len(arms) + "}",
        "\\toprule",
        "graph & $M$ & " + " & ".join(LABEL[a] for a in arms) + " \\\\",
        "\\midrule",
    ]

    for g in graphs:
        have = [a for a in arms if agg.get((g, a))]
        best = (
            min(have, key=lambda a: np.mean(agg[(g, a)]))
            if bold_best and have
            else None
        )
        cells = [
            fmt(agg[(g, a)], bold=(a == best)) if agg.get((g, a)) else "--"
            for a in arms
        ]
        lines.append(
            g.replace("_", r"\_") + f" & {ms[g]} & " + " & ".join(cells) + " \\\\"
        )

    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}", flush=True)


def separable(a, b, alpha=0.05):
    # paired over the seeds, so the seed's difficulty cancels, a zero-variance pair (a closed
    # form that ignores the seed) counts as not separable rather than as a free win
    if len(a) != len(b) or len(a) < 3:
        return False

    diff = np.asarray(a, float) - np.asarray(b, float)

    if not np.any(np.abs(diff) > 0) or np.std(diff, ddof=1) == 0:
        return False

    return bool(ttest_rel(a, b).pvalue < alpha)


def write_main(graphs, agg, ms, path):
    lines = [
        "\\begin{tabular}{lr" + "r" * len(MAIN_ARMS) + "}",
        "\\toprule",
        " & & \\multicolumn{3}{c}{trained on the sampled energy}"
        " & \\multicolumn{2}{c}{closed form} \\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}",
        "graph & $M$ & " + " & ".join(LABEL[a] for a in MAIN_ARMS) + " \\\\",
        "\\midrule",
    ]
    n_star = 0

    for g in graphs:
        order = sorted(
            (a for a in MAIN_ARMS if agg.get((g, a))),
            key=lambda a: np.mean(agg[(g, a)]),
        )
        best = order[0]
        # a star means the field wins AND the runner-up is separable from it
        star = (
            len(order) > 1
            and best == "field"
            and separable(agg[(g, best)], agg[(g, order[1])])
        )
        n_star += star
        cells = []

        for a in MAIN_ARMS:
            if not agg.get((g, a)):
                cells.append("--")
                continue

            txt = f"{np.mean(agg[(g, a)]):.3f}"

            if len(agg[(g, a)]) > 1:
                txt += f"\\,\\pm\\,{np.std(agg[(g, a)], ddof=1):.3f}"

            if a == best:
                txt = f"\\mathbf{{{txt}}}"

            # every cell reserves the star's width, so a starred number stays column-aligned
            mark = "^{*}" if star and a == best else r"\phantom{^{*}}"
            cells.append(f"${txt}{mark}$")

        lines.append(
            g.replace("_", "\\_") + f" & {ms[g]} & " + " & ".join(cells) + " \\\\"
        )

    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}", flush=True)
    print(
        f"  field separable from the runner-up on {n_star}/{len(graphs)} graphs",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--graphs", nargs="+", default=GRAPHS)
    ap.add_argument("--tables-dir", default=str(ROOT / "tables"))
    ap.add_argument("--cache-dir", default=str(ROOT / "cache" / "lean_tables"))
    ap.add_argument(
        "--data-dir",
        default=None,
        help="directory holding 3elt/, dwt_1005/, jagmesh8/, cora/, "
        "facebook_combined.txt",
    )
    ap.add_argument("--m-train", type=int, default=500)
    ap.add_argument("--npiv", type=int, default=400)
    ap.add_argument("--n-land", type=int, default=32)
    ap.add_argument(
        "--ladder",
        nargs="+",
        type=int,
        default=LADDER,
        help="stopping-point ladder (default: the published one)",
    )
    args = ap.parse_args()

    if args.data_dir:
        fdata.DATA = Path(args.data_dir)
    elif (ROOT / "data").exists():
        fdata.DATA = ROOT / "data"

    cfg = dict(
        m_train=args.m_train,
        npiv=args.npiv,
        n_land=args.n_land,
        ladder=list(args.ladder),
        lrs=LRS,
        gammas=GAMMAS,
        held_frac=HELD_FRAC,
    )
    print(f"device {DEV}, config {cfg_hash(cfg)}", flush=True)

    rows = collect(args.graphs, args.seeds, cfg, Path(args.cache_dir))

    agg, ms = defaultdict(list), {}

    for r in rows:
        agg[(r["graph"], r["arm"])].append(r["stress"])
        ms[r["graph"]] = r["m"]

    graphs = list(dict.fromkeys(r["graph"] for r in rows))
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    write_main(graphs, agg, ms, tables_dir / "tab_main.tex")
    write_lean(graphs, agg, ms, MATCHED, tables_dir / "tab_lean.tex", bold_best=True)
    write_lean(
        graphs, agg, ms, CONTROLS, tables_dir / "tab_lean_controls.tex", bold_best=False
    )


if __name__ == "__main__":
    main()
