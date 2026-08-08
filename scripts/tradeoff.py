import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fling import FlingVis, figstyle as fs  # noqa: E402
from fling import data as fdata  # noqa: E402
from fling import vis as vis_mod  # noqa: E402
from fling.baselines import all_pairs_D, impred, label_purity  # noqa: E402
from fling.baselines import sgd_stress, tsnet, tsnet_star  # noqa: E402
from fling.metrics import evaluate, fast_D, _neighbor_sets  # noqa: E402
from fling.vis import seg_dist  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

CACHE = ROOT / "cache" / "tradeoff"
FIGDIR = ROOT / "figures"

GRAPHS = ["lesmis", "cora"]
SEEDS = [0, 1, 2, 3, 4, 5]
W_STRESS = 2.0  # the weight the stress arm carries on the all-pair term
IMPRED_ITERS = 100
FRAC = 0.25  # occlusion threshold, as a fraction of the median edge length

# Hue encodes the METHOD. `+ImPrEd` is the same base drawing refined, so it takes its base's
# hue and is drawn hollow. Kept in step with the palette the other paper figures use.
SERIES = [
    ("s_gd2", "s_gd2", "#12a698", "s"),
    ("tsNET", "tsNET", "#8fa8ee", "s"),
    ("tsNET*", "tsNET*", "#3a5fb0", "s"),
    ("vis_base", "FlingVis base", "#d1495b", "o"),
    ("vis_base+ImPrEd", "+ ImPrEd", "#d1495b", "o"),
    ("vis_full", "FlingVis (default)", "#e8871a", "o"),
    ("vis_stress", "FlingVis + stress", "#e8871a", "D"),
]

# The legibility axis is `occ_viol`, the metric the paper defines: a quarter of the median EDGE
# LENGTH. `occ_viol_ns` is scored and printed but not reported, because its threshold divides by
# a node spacing the layout itself sets, so a term that compresses the node cloud moves the
# metric without moving the drawing.
PANELS = [
    ("occ_viol", "purity", "node-edge occlusion", "class purity"),
    ("stress", "np_jaccard", "stress", "NP"),
]

PAIR_GAP = 0.05  # figure fraction: how far each pair is pushed from the centre


def load_graph(name):
    labels = None

    if name == "cora":
        graph, labels, _ = fdata.load_cora()
    else:
        by_name = dict(fdata.graphs(quick=True))
        if name not in by_name:
            raise SystemExit(f"unknown graph {name}")
        graph = by_name[name]

    graph = nx.convert_node_labels_to_integers(
        graph.subgraph(max(nx.connected_components(graph), key=len))
    )

    if labels is None:  # no ground truth: use modularity communities as the target
        comms = nx.community.greedy_modularity_communities(graph)
        labels = np.zeros(graph.number_of_nodes(), int)
        for cid, comm in enumerate(comms):
            for node in comm:
                labels[node] = cid

    return graph, labels


def stress_wrapper(base, dist, weight):
    tt = torch.tensor(np.asarray(dist, np.float32))
    ok = tt > 0

    def wrapped(pos, *args, **kwargs):
        loss = base(pos, *args, **kwargs)

        if weight <= 0:
            return loss

        d = tt.to(pos.device)
        m = ok.to(pos.device)
        emb = torch.cdist(pos, pos)
        safe = torch.where(m, d, torch.ones_like(d))
        ratio = emb / safe

        # the scale that minimises the normalised stress, in closed form. Without it the term
        # would fight the neighbour embedding over the SIZE of the drawing, which is otherwise
        # unconstrained, rather than over its shape.
        a = (ratio * m).sum() / ((ratio**2 * m).sum() + 1e-12)
        term = ((((a * emb - safe) / safe) ** 2) * m).sum() / m.sum()

        return loss + weight * term

    return wrapped


def fit_with_stress(graph, seed, dmat, weight):
    # the term is added by replacing `fling.vis.ne_loss`, so both phases carry it and every
    # other line of the estimator is the shipped one
    original = vis_mod.ne_loss
    vis_mod.ne_loss = stress_wrapper(original, np.asarray(dmat, float), weight)

    try:
        return FlingVis(n_components=2, seed=seed).fit_transform(graph)
    finally:
        vis_mod.ne_loss = original


def fitters(graph, dist_full, dmat, w_stress):
    return {
        "s_gd2": lambda s: sgd_stress(graph, seed=s),
        "tsNET": lambda s: tsnet(dist_full, seed=s),
        "tsNET*": lambda s: tsnet_star(dist_full, seed=s),
        "vis_base": lambda s: FlingVis(
            n_components=2, seed=s, aes_iters=0
        ).fit_transform(graph),
        "vis_full": lambda s: FlingVis(n_components=2, seed=s).fit_transform(graph),
        "vis_stress": lambda s: fit_with_stress(graph, s, dmat, w_stress),
    }


def lay_path(cache, graph, arm, seed):
    return cache / "layouts" / f"{graph}__{arm.replace('*', 'star')}__{seed}.npy"


def clearances(pos, edges, dev, cap=4_000_000):
    pos_t = torch.tensor(np.asarray(pos, float), dtype=torch.float32, device=dev)
    ed = torch.tensor(np.asarray(edges), device=dev)
    n_nodes = pos_t.shape[0]
    chunk = max(1, int(cap // max(1, len(edges))))
    vals, args = [], []

    for start in range(0, n_nodes, chunk):
        idx = torch.arange(start, min(start + chunk, n_nodes), device=dev)
        dist = seg_dist(pos_t[idx][:, None, :], pos_t[ed[:, 0]], pos_t[ed[:, 1]])
        inc = (ed[None, :, 0] == idx[:, None]) | (ed[None, :, 1] == idx[:, None])
        best = dist.masked_fill(inc, float("inf")).min(1)
        vals.append(best.values)
        args.append(best.indices)

    return torch.cat(vals).cpu().numpy(), torch.cat(args).cpu().numpy()


def nn_spacing(pos, dev):
    p = torch.tensor(np.asarray(pos, float), dtype=torch.float32, device=dev)
    d = torch.cdist(p, p)
    d.fill_diagonal_(float("inf"))

    return d.min(1).values.cpu().numpy()


def occ_metrics(pos, edges, dev, frac=FRAC):
    pos = np.asarray(pos, float)
    edges = np.asarray(edges)
    clear, near = clearances(pos, edges, dev)
    elen = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)
    med_edge = float(np.median(elen))
    med_nn = float(np.median(nn_spacing(pos, dev)))

    viol_ns = clear < frac * med_nn
    long_decile = elen >= np.quantile(elen, 0.9)
    share = float(long_decile[near[viol_ns]].mean()) if viol_ns.any() else float("nan")

    return dict(
        occ_viol=float(np.mean(clear < frac * med_edge)),
        occ_viol_ns=float(viol_ns.mean()),
        occ_min_ns=float(clear.min() / med_nn),
        occ_p01_ns=float(np.quantile(clear, 0.01) / med_nn),
        med_edge=med_edge,
        med_nn=med_nn,
        edge_tail=float(
            elen.max() / med_edge
        ),  # heavy tail => the published threshold drifts
        long_edge_share=share,  # violations owed to the longest decile
    )


def score(pos, graph, labels, edges, dmat, nbrs, dev):
    # crossings are O(E^2) and are not a disentangling measure: skip them on dense graphs
    res = evaluate(pos, graph, dmat, nbrs, cross_max_e=1200, angle_max_e=1200)
    keep = np.array([np.sum(labels == lab) > 1 for lab in labels])
    res["silhouette"] = (
        float(silhouette_score(pos[keep], labels[keep]))
        if keep.sum() > 10
        else float("nan")
    )
    res["purity"] = float(label_purity(pos, labels, n_nbrs=10))
    res.update(occ_metrics(pos, edges, dev))

    return res


def fit_and_score(names, seeds, w_stress, impred_iters, with_impred, cache, dev):
    path = cache / "scores.json"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "layouts").mkdir(parents=True, exist_ok=True)
    cells = json.loads(path.read_text()) if path.exists() else {}

    for name in names:
        graph, labels = load_graph(name)
        edges = np.array(graph.edges())
        dmat = fast_D(graph)
        nbrs = _neighbor_sets(graph)
        arms = list(fitters(graph, None, dmat, w_stress))

        if with_impred:
            arms.append("vis_base+ImPrEd")

        print(
            f"\n=== {name} n={graph.number_of_nodes()} e={len(edges)} "
            f"communities={len(set(labels))}",
            flush=True,
        )

        need_d = any(
            not lay_path(cache, name, a, s).exists()
            for a in ("tsNET", "tsNET*")
            for s in seeds
        )

        dist_full = all_pairs_D(graph) if need_d else None
        built = fitters(graph, dist_full, dmat, w_stress)

        for arm in arms:
            for seed in seeds:
                key = f"{name}|{arm}|{seed}"
                out = lay_path(cache, name, arm, seed)

                if key in cells and out.exists():
                    print(f"  {arm:<16} s{seed} cached", flush=True)
                    continue

                if not out.exists():
                    tic = time.perf_counter()

                    if arm == "vis_base+ImPrEd":
                        base = lay_path(cache, name, "vis_base", seed)
                        pos = impred(np.load(base), edges, iters=impred_iters)
                    else:
                        pos = np.asarray(built[arm](seed), float)

                    secs = time.perf_counter() - tic
                    np.save(out, np.asarray(pos, float))
                else:
                    secs = float("nan")

                res = score(np.load(out), graph, labels, edges, dmat, nbrs, dev)
                cells[key] = dict(
                    graph=name,
                    n=graph.number_of_nodes(),
                    arm=arm,
                    seed=seed,
                    secs=secs,
                    **res,
                )

                path.write_text(json.dumps(cells, indent=1))
                print(
                    f"  {arm:<16} s{seed} occ={res['occ_viol']:.3f} pur={res['purity']:.3f} "
                    f"np={res['np_jaccard']:.3f} stress={res['stress']:.4f} "
                    f"{secs:6.1f}s",
                    flush=True,
                )

    return cells


def rows_by_arm(cache):
    agg = defaultdict(list)
    path = cache / "scores.json"

    if not path.exists():
        raise SystemExit(f"no scores at {path}")

    for r in json.loads(path.read_text()).values():
        agg[(r["graph"], r["arm"])].append(r)

    return agg


def figure(
    names,
    cache,
    outdir,
    name="fig_a3_tradeoff",
    height=1.66,
    label_drop=0.100,
    legend_room=0.155,
    legend_y=0.045,
):
    agg = rows_by_arm(cache)
    series = [row for row in SERIES if any((g, row[0]) in agg for g in names)]
    graphs = [(g, agg[(g, "s_gd2")][0]["n"]) for g in names]

    fs.use()

    # figstyle.use() sets savefig.bbox="tight", which would trim the page to the ink and deliver
    # something other than the 5.5 in the paper includes it at
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        fig, axes = plt.subplots(1, len(PANELS) * len(graphs), figsize=(5.5, height))
        axes = np.atleast_1d(axes)
        small = dict(title=7.0, label=6.8, tick=6.0)

        for p in range(
            len(PANELS)
        ):  # each pair shares one y scale, joined before drawing
            base = p * len(graphs)
            for j in range(1, len(graphs)):
                axes[base + j].sharey(axes[base])

        col = 0
        for xkey, ykey, xname, yname in PANELS:
            for g, n in graphs:
                ax = axes[col]

                for arm, label, hue, mark in series:
                    rs = agg.get((g, arm))

                    if not rs:
                        print(f"  WARNING: no rows for {g}/{arm}", flush=True)
                        continue

                    x = np.mean([r[xkey] for r in rs])
                    y = np.mean([r[ykey] for r in rs])
                    xe = np.std([r[xkey] for r in rs], ddof=1)
                    ye = np.std([r[ykey] for r in rs], ddof=1)
                    ours = mark in ("o", "v", "D")
                    refined = arm.endswith("+ImPrEd")
                    ax.errorbar(
                        x,
                        y,
                        xerr=xe,
                        yerr=ye,
                        fmt=mark,
                        ms=(5 if mark == "o" else 4.6) if ours else 4,
                        color=hue,
                        markerfacecolor="none" if refined else hue,
                        mec=hue if refined else ("0.2" if ours else "none"),
                        mew=0.6,
                        ecolor="0.6",
                        elinewidth=0.6,
                        capsize=1.2,
                        zorder=3,
                        label=label if col == 0 else None,
                    )

                ax.set_title(f"{g}, $N={n:,}$", pad=3, fontsize=small["title"])

                if col % len(graphs) == 0:
                    ax.set_ylabel(
                        f"{yname} $\\uparrow$", fontsize=small["label"], labelpad=1
                    )
                else:
                    ax.tick_params(labelleft=False)

                ax.tick_params(labelsize=small["tick"], pad=1.0, length=2.0)
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                ax.yaxis.set_major_locator(plt.MaxNLocator(4))
                ax.invert_xaxis()  # lower-is-better to the right => best is top right
                ax.set_box_aspect(1.0)  # square panels
                fs.grid(ax, axis="both")
                col += 1

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            ncol=len(series),
            fontsize=5.6,
            handlelength=0.8,
            columnspacing=0.9,
            handletextpad=0.35,
            borderpad=0.0,
            borderaxespad=0.0,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, legend_y),
        )

        # The rect's x-margins are what PAIR_GAP spends: with savefig.bbox cleared nothing
        # reclaims ink that falls off the page. Its bottom is the room the single legend row
        # needs, a third of what three rows took.
        fig.tight_layout(
            pad=0.08,
            w_pad=0.02,
            rect=(PAIR_GAP / 2, legend_room, 1 - PAIR_GAP / 2, 1.0),
        )

        # One x label per pair, added after tight_layout, which cannot see fig.text.
        for p, (_, _, xname, _) in enumerate(PANELS):
            shift = (p - 0.5) * PAIR_GAP

            for j in range(len(graphs)):
                b = axes[p * len(graphs) + j].get_position()
                axes[p * len(graphs) + j].set_position(
                    [b.x0 + shift, b.y0, b.width, b.height]
                )

            lo = axes[p * len(graphs)].get_position()
            hi = axes[(p + 1) * len(graphs) - 1].get_position()
            fig.text(
                0.5 * (lo.x0 + hi.x1),
                lo.y0 - label_drop,
                f"{xname} $\\downarrow$",
                ha="center",
                va="center",
                fontsize=small["label"],
            )

        outdir.mkdir(parents=True, exist_ok=True)

        # NOT fs.save: it passes bbox_inches="tight" explicitly, which is the same trim.
        fig.savefig(outdir / f"{name}.pdf")
        fig.savefig(outdir / f"{name}.png", dpi=600)
        plt.close(fig)

    print(f"saved {outdir / name}.pdf + .png", flush=True)


def stat(rs, key):
    vals = [r[key] for r in rs if r.get(key) is not None]

    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)


def table(cache):
    agg = rows_by_arm(cache)
    print(
        f"\n{'graph':<8}{'arm':<18}{'stress':>9}{'NP':>8}{'purity':>8}{'occ':>8}{'occ_ns':>9}"
    )

    for (g, arm), rs in sorted(agg.items()):
        cell = {
            k: stat(rs, k)
            for k in ("stress", "np_jaccard", "purity", "occ_viol", "occ_viol_ns")
        }

        print(
            f"{g:<8}{arm:<18}{cell['stress'][0]:>9.4f}{cell['np_jaccard'][0]:>8.3f}"
            f"{cell['purity'][0]:>8.3f}{cell['occ_viol'][0]:>8.3f}"
            f"{cell['occ_viol_ns'][0]:>9.3f}   +-{cell['stress'][1]:.4f} n={len(rs)}",
            flush=True,
        )


def main(
    names, seeds, w_stress, impred_iters, with_impred, data_dir, cache, outdir, device
):
    fdata.DATA = Path(data_dir) if data_dir else ROOT / "data"
    dev = torch.device(
        "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    # the estimators create their tensors without an explicit device, so they follow the torch
    # default: without this every arm would run on the CPU
    torch.set_default_device(dev)

    fit_and_score(names, seeds, w_stress, impred_iters, with_impred, cache, dev)
    table(cache)
    figure(names, cache, outdir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--graphs", nargs="*", default=GRAPHS)
    p.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    p.add_argument("--w-stress", type=float, default=W_STRESS)
    p.add_argument("--impred-iters", type=int, default=IMPRED_ITERS)
    p.add_argument(
        "--skip-impred",
        action="store_true",
        help="drop the refinement arm, which is the slow one on cora",
    )
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--outdir", type=Path, default=FIGDIR)
    p.add_argument("--device", default="cuda")

    a = p.parse_args()

    main(
        a.graphs,
        a.seeds,
        a.w_stress,
        a.impred_iters,
        not a.skip_impred,
        a.data_dir,
        a.cache,
        a.outdir,
        a.device,
    )
