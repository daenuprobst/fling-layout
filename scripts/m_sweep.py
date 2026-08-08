import argparse
import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from lib import DEV, blocks, field_curve, held_stress, split, stress_rows

import fling.data as fdata
from fling import figstyle as fs
from fling import make_graph
from fling.metrics import fast_D, normalized_stress

GRID = [50, 100, 200, 500, 1000, 2000]
GRAPHS = ["dwt_1005", "jagmesh8", "cora", "tree_4000", "ego-Facebook", "3elt"]
ARMS = ("field", "linear", "pivotmds", "landmarkmds", "random")
STYLE = [
    ("field", "field", "C0"),
    ("linear", "linear", "C1"),
    ("pivotmds", "PivotMDS", "C2"),
    ("landmarkmds", "landmark MDS", "C3"),
]

NPIV = 400
N_LAND = 32
LADDER = [10, 25, 50, 75, 150, 300, 600, 1200, 2400]
LRS = [5e-3, 5e-2, 5e-1]


def resolve_data(given):
    for cand in (given, ROOT / "data", fdata.DATA):
        if cand is not None and Path(cand).is_dir():
            return Path(cand)

    raise SystemExit(
        "no data directory found: pass --data (needs cora/, 3elt/, "
        "dwt_1005/, jagmesh8/, facebook_combined.txt)"
    )


def graph_of(name):
    if name == "cora":
        graph = fdata.load_cora()[0]
    elif name == "ego-Facebook":
        graph = nx.read_edgelist(fdata.DATA / "facebook_combined.txt", nodetype=int)
    elif name.startswith("tree_"):
        graph = make_graph("tree", int(name.split("_")[1]))
    elif name.startswith("delaunay_"):
        graph = make_graph("delaunay", int(name.split("_")[1]))
    else:
        graph = fdata._load_mtx(name)

    comp = max(nx.connected_components(graph), key=len)

    return nx.convert_node_labels_to_integers(graph.subgraph(comp))


def linear_curve(
    feat, dist_tr, dist_ho, piv_rows, ho_rows, tix, oos, dmat, seed, lr, iters
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

    model = torch.nn.Linear(ft.shape[1], 2).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(1, iters + 1):
        cols = train_cols[
            torch.randint(0, len(train_cols), (32,), device=DEV, generator=gen)
        ]
        opt.zero_grad(set_to_none=True)
        stress_rows(model(ft[ru]), tgt, pr, cols).backward()
        opt.step()

        for grp in opt.param_groups:  # each budget anneals over ITSELF
            grp["lr"] = lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * step / iters)))

    with torch.no_grad():
        full = model(ft)
        held = held_stress(
            full,
            torch.as_tensor(oos, device=DEV),
            torch.as_tensor(dist_ho, device=DEV),
            ho_pr,
        )
        gold = float(normalized_stress(full.cpu().numpy().astype(float), dmat))

    return held, gold


def pivot_mds(block):
    sq = block.astype(np.float64) ** 2
    gram = -0.5 * (
        sq - sq.mean(0, keepdims=True) - sq.mean(1, keepdims=True) + sq.mean()
    )
    uvec, sval, _ = np.linalg.svd(gram, full_matrices=False)

    return uvec[:, :2] * sval[:2]


def landmark_mds(block, rows):
    dl = block[rows].astype(np.float64)
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
    out = -0.5 * (block.astype(np.float64) ** 2 - mean_sq[None, :]) @ pinv.T
    out[rows] = land_pos

    return out


def compute(names, grid, seeds, cache):
    cells = json.loads(cache.read_text()) if cache.exists() else {}

    def save():
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cells, indent=1))
        tmp.replace(cache)

    for name in names:
        graph = graph_of(name)
        n = graph.number_of_nodes()
        ms = [m for m in grid if m <= n // 2]
        dmat = None
        print(f"\n=== {name}  n={n}  Ms={ms}", flush=True)

        for m_train in ms:
            for seed in seeds:
                todo = [a for a in ARMS if f"{name}|{m_train}|{seed}|{a}" not in cells]

                if not todo:
                    continue

                if dmat is None:
                    dmat = fast_D(graph)

                def put(arm, stress, secs, **extra):
                    cells[f"{name}|{m_train}|{seed}|{arm}"] = dict(
                        stress=float(stress), secs=round(secs, 1), **extra
                    )
                    save()
                    print(
                        f"  M={m_train} seed={seed} {arm:<12} {stress:.4f}  "
                        f"{secs:6.1f}s",
                        flush=True,
                    )

                if "random" in todo:
                    # offset the stream: a delaunay graph's own point set comes from
                    # default_rng(0), which would make the "random" floor its planar embedding
                    rng = np.random.default_rng(10_000 + seed)
                    put("random", normalized_stress(rng.random((n, 2)), dmat), 0.0)

                if todo == ["random"]:
                    continue

                train, tix, held_pivots, npiv_eff, oos = split(
                    graph, m_train, NPIV, seed
                )

                tic = time.perf_counter()
                best, keep = None, None
                for lr in LRS:
                    for iters in LADDER:
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
                            N_LAND,
                        )
                        if best is None or held < best[0]:
                            best, keep = (held, gold, lr, iters), (est, tixe)
                if "field" in todo:
                    put(
                        "field",
                        best[1],
                        time.perf_counter() - tic,
                        lr=best[2],
                        step=best[3],
                    )

                est, tixe = keep
                feat, dist_tr, dist_ho, piv_rows, ho_rows = blocks(
                    est, graph, held_pivots
                )

                if "linear" in todo:
                    tic = time.perf_counter()
                    pick = None
                    for lr in LRS:
                        for iters in LADDER:
                            held, gold = linear_curve(
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
                    put(
                        "linear",
                        pick[1],
                        time.perf_counter() - tic,
                        lr=pick[2],
                        step=pick[3],
                    )

                # the closed forms get every column the field saw, training and selection alike
                land = dist_tr[:, : min(N_LAND, dist_tr.shape[1])]
                allcols = np.concatenate([dist_tr, dist_ho], 1)
                all_rows = np.concatenate([piv_rows, ho_rows])
                for arm, fn in (
                    ("pivotmds", lambda blk, rws: pivot_mds(blk)),
                    ("landmarkmds", landmark_mds),
                ):
                    if arm not in todo:
                        continue
                    tic = time.perf_counter()
                    cand = {}
                    for tag, blk, rws in (
                        ("land32", land, piv_rows[: land.shape[1]]),
                        ("pivall", allcols, all_rows),
                    ):
                        try:
                            cand[tag] = float(normalized_stress(fn(blk, rws), dmat))
                        except Exception as exc:
                            print(f"    {arm}/{tag} failed: {exc}", flush=True)
                    if cand:
                        pickt = min(cand, key=cand.get)
                        put(arm, cand[pickt], time.perf_counter() - tic, block=pickt)

            done = [k for k in cells if k.startswith(f"{name}|{m_train}|")]
            mean = {
                a: np.mean(
                    [
                        cells[k]["stress"]
                        for k in done
                        if k.endswith(f"|{a}") and int(k.split("|")[2]) in seeds
                    ]
                    or [np.nan]
                )
                for a in ("field", "linear", "pivotmds")
            }

            print(
                f"== DONE {name} M={m_train}  field={mean['field']:.4f}  "
                f"linear={mean['linear']:.4f}  pivotmds={mean['pivotmds']:.4f}",
                flush=True,
            )

    save()
    print(f"\nwrote {cache}")

    return cells


def plot(cells, names, seeds, out_dir):
    fs.use()
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.4), sharex=False)

    for ax, g in zip(axes.ravel(), names):
        ms = sorted({int(k.split("|")[1]) for k in cells if k.startswith(g + "|")})

        if not ms:
            ax.set_axis_off()
            continue

        for arm, label, color in STYLE:
            mu, sd, xs = [], [], []
            for m in ms:
                v = [
                    cells[f"{g}|{m}|{s}|{arm}"]["stress"]
                    for s in seeds
                    if f"{g}|{m}|{s}|{arm}" in cells
                ]
                if v:
                    xs.append(m)
                    mu.append(np.mean(v))
                    sd.append(np.std(v, ddof=1) if len(v) > 1 else 0.0)
            mu, sd = np.array(mu), np.array(sd)
            ax.plot(
                xs,
                mu,
                "-",
                color=color,
                lw=1.0,
                marker="o",
                ms=2.2,
                mfc="white",
                mew=0.6,
                label=label,
            )
            ax.fill_between(xs, mu - sd, mu + sd, color=color, alpha=0.18, lw=0)

        # the published operating point M = min(500, N/2)
        ax.axvline(
            500 if max(ms) >= 500 else max(ms),
            color="0.55",
            lw=0.7,
            ls=(0, (3, 3)),
            zorder=0,
        )
        ax.set_xscale("log")
        ax.set_xticks(ms)
        ax.set_xticklabels([str(m) for m in ms], fontsize=5.4, rotation=0)
        ax.minorticks_off()
        ax.tick_params(labelsize=5.4, pad=1.2, length=2.0)
        ax.set_title(g, fontsize=7.0, pad=2)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0, 0].legend(
        fontsize=5.6,
        frameon=False,
        handlelength=1.4,
        borderaxespad=0.2,
        labelspacing=0.25,
    )

    for ax in axes[:, 0]:
        ax.set_ylabel("exact stress", fontsize=6.6)

    for ax in axes[1, :]:
        ax.set_xlabel("$M$", fontsize=6.6)

    fig.tight_layout(pad=0.4)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"m_sweep.{ext}", dpi=600)

    print(f"wrote {out_dir / 'm_sweep.pdf'} and .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--graphs", nargs="*", default=GRAPHS)
    ap.add_argument("--grid", type=int, nargs="*", default=GRID)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "figures")
    ap.add_argument("--cache", type=Path, default=ROOT / "cache" / "m_sweep.json")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    fdata.DATA = resolve_data(args.data)

    if args.plot_only:
        cells = json.loads(args.cache.read_text())
    else:
        cells = compute(args.graphs, args.grid, args.seeds, args.cache)

    plot(cells, args.graphs, args.seeds, args.out_dir)


if __name__ == "__main__":
    main()
