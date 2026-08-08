import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fling import FlingBlend, figstyle as fs  # noqa: E402
from fling.metrics import fast_D, _neighbor_sets  # noqa: E402
from fling.metrics import normalized_stress, neighborhood_preservation  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

CACHE = ROOT / "cache" / "lambda_family"
FIGDIR = ROOT / "figures"

TEXT_W = 5.5  # \textwidth of the paper's text block, in inches
INK = "0.0"
TINT = "#e4e9f0"  # background of the panels whose lambda the fit actually saw
RED = "#FF0000"
CMAP = plt.get_cmap("turbo")

ANCHORS = (0.0, 1.0)  # the only lambdas the two-anchor fit ever sees
ANCHORS9 = tuple(np.round(np.linspace(0.0, 1.0, 9), 4))  # the reference ladder
LADDER = np.round(np.linspace(0.0, 1.0, 201), 4)  # every rung that gets scored
N_PANEL = 7
N_SPEC = 60  # independent single-lambda fits, plotted as the grey band


class AnchorBlend(FlingBlend):
    anchors_ = ANCHORS

    def _sample_lams(self, gen, dev):
        pick = torch.randint(
            0, len(self.anchors_), (self.n_lam,), device=dev, generator=gen
        )

        return torch.tensor(self.anchors_, device=dev, dtype=torch.float32)[pick]


class EndsBlend(AnchorBlend):
    anchors_ = ANCHORS


class NineBlend(AnchorBlend):
    anchors_ = ANCHORS9


def ring():
    graph = nx.watts_strogatz_graph(300, 6, 0.02, seed=0)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    return nx.convert_node_labels_to_integers(
        graph.subgraph(max(nx.connected_components(graph), key=len))
    )


def framed(pos):
    # every member is centred and scaled by its OWN RMS radius: nothing pins the extent at
    # lambda=0, so a shared divisor shrinks every panel but one to a dot
    out = pos - pos.mean(1, keepdims=True)

    return out / np.sqrt((out**2).sum(-1).mean(-1))[:, None, None]


def by_travel(pos, lams, n=N_PANEL):
    # rungs at equal fractions of the family's path length, not at equal lambda: the family
    # does not move at a constant rate, and this rule is fixed before the result is seen
    p = pos - pos.mean(1, keepdims=True)
    p = p / np.sqrt((p**2).sum(-1).mean(-1))[:, None, None]
    step = np.sqrt((np.diff(p, axis=0) ** 2).sum(-1).mean(-1))
    travel = np.concatenate([[0.0], np.cumsum(step)])
    want = np.linspace(0.0, travel[-1], n)

    taken, out = set(), []
    for w in want:
        order = np.argsort(np.abs(travel - w))
        pick = next(int(i) for i in order if int(i) not in taken)
        taken.add(pick)
        out.append(pick)

    return lams[np.sort(out)]


def metrics(pos, dmat, nbrs):
    return np.array(
        [
            [normalized_stress(xy, dmat), neighborhood_preservation(xy, nbrs)]
            for xy in pos
        ]
    )


def compute(seed, iters, device, cache):
    cell = cache / f"seed{seed}_iters{iters}"
    (cell / "spec").mkdir(parents=True, exist_ok=True)
    graph = ring()
    dmat, nbrs = fast_D(graph), _neighbor_sets(graph)
    edges = np.array([[u, v] for u, v in graph.edges()])
    idx = np.arange(graph.number_of_nodes())

    ends_path = cell / "ends.npz"
    if ends_path.exists():
        blob = np.load(ends_path, allow_pickle=False)
        pos, met, scales = blob["pos"], blob["met"], blob["scales"]
        print("ends fit cached", flush=True)
    else:
        model = EndsBlend(iters=iters, seed=seed, device=device).fit(graph)
        pos = model.family(LADDER)
        met = metrics(pos, dmat, nbrs)
        scales = np.array(model.scales_, float)
        np.savez_compressed(ends_path, pos=pos, met=met, scales=scales)
        print(f"ends fit {model.fit_seconds_:.1f}s", flush=True)

    # The reference the interpolation is read against: one unconditional field trained at a
    # single lambda, sixty times over. They are handed the conditional fit's calibration so the
    # two energies are weighed identically, and the same budget and seed.
    spec_lams = np.round(np.linspace(0.0, 1.0, N_SPEC), 4)
    spec_met = []
    for j, lam in enumerate(spec_lams):
        one_path = cell / "spec" / f"{j:03d}.npy"

        if one_path.exists():
            spec_met.append(np.load(one_path))
            continue

        one = FlingBlend(
            iters=iters,
            seed=seed,
            device=device,
            lam=float(lam),
            scales=tuple(float(v) for v in scales),
        ).fit(graph)
        xy = one.draw(float(lam))
        row = np.array(
            [normalized_stress(xy, dmat), neighborhood_preservation(xy, nbrs)]
        )
        np.save(one_path, row)
        spec_met.append(row)
        print(
            f"specialist {j + 1}/{N_SPEC} lam={lam:.3f} "
            f"stress={row[0]:.4f} NP={row[1]:.4f}",
            flush=True,
        )

    spec_met = np.array(spec_met)

    nine_path = cell / "nine.npz"

    if nine_path.exists():
        met9 = np.load(nine_path, allow_pickle=False)["met9"]
        print("nine-anchor fit cached", flush=True)
    else:
        nine = NineBlend(iters=iters, seed=seed, device=device).fit(graph)
        met9 = metrics(nine.family(LADDER), dmat, nbrs)
        np.savez_compressed(nine_path, met9=met9)
        print(f"nine-anchor fit {nine.fit_seconds_:.1f}s", flush=True)

    return {
        "ring": idx,
        "edges": edges,
        "lams": LADDER,
        "pos": pos,
        "met": met,
        "anchors": np.array(ANCHORS),
        "spec_lams": spec_lams,
        "spec_met": spec_met,
        "met9": met9,
    }


def panel(ax, pos, edges, colours, win, tint, node, lw):
    ax.add_collection(
        LineCollection(pos[edges], colors="0.68", linewidths=lw, zorder=1)
    )
    ax.scatter(pos[:, 0], pos[:, 1], s=node, c=colours, linewidths=0, zorder=2)
    ax.set_xlim(win[0], win[1])
    ax.set_ylim(win[2], win[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(TINT if tint else "white")

    for side in ax.spines.values():
        side.set_visible(False)


def sheet(data, outdir, name, nrow, ncol):
    lams, met, edges = data["lams"], data["met"], data["edges"]
    colours = CMAP(data["ring"] / len(data["ring"]))
    shown = framed(data["pos"])

    shown_lams = by_travel(data["pos"], lams)
    take = np.array([int(np.argmin(np.abs(lams - s))) for s in shown_lams])
    trained = np.isin(np.round(lams, 3), np.round(data["anchors"], 3))
    trained9 = np.isin(np.round(lams, 4), np.round(ANCHORS9, 4))
    n_new_panels = int((~trained[take]).sum())
    n_new_scored = int((~trained).sum()) - n_new_panels

    box = shown[take].reshape(-1, 2)
    lo, hi = box.min(0), box.max(0)
    mid, half = (hi + lo) / 2, 1.04 * (hi - lo) / 2
    win = (mid[0] - half[0], mid[0] + half[0], mid[1] - half[1], mid[1] + half[1])
    aspect = half[0] / half[1]

    print(
        f"for the caption: {n_new_panels} of {len(take)} panels untrained, "
        f"{n_new_scored} further scored rungs untrained, panel lambdas "
        f"{[float(lams[k]) for k in take]}",
        flush=True,
    )

    strips = (
        (
            "stress $\\downarrow$",
            met[:, 0],
            data["spec_met"][:, 0],
            data["met9"][:, 0],
            True,
        ),
        (
            "NP $\\uparrow$",
            met[:, 1],
            data["spec_met"][:, 1],
            data["met9"][:, 1],
            False,
        ),
    )

    pad, gap, rgap = 0.05, 0.05, 0.075
    side = (TEXT_W - 2 * pad - (ncol - 1) * gap) / ncol
    ph = side / aspect
    lab, key, mh, mb = 0.15, 0.10, 0.52, 0.26
    height = pad + nrow * (ph + lab) + (nrow - 1) * rgap + key + mh + mb
    top = height - pad
    node, lw = 1.15 * side + 0.4, 0.055 * side + 0.10

    fs.use()
    fig = plt.figure(figsize=(TEXT_W, height))

    def axes_at(x0, y0, w, h):
        return fig.add_axes([x0 / TEXT_W, y0 / height, w / TEXT_W, h / height])

    for slot, k in enumerate(take):
        row, col = slot // ncol, slot % ncol
        x0 = pad + col * (side + gap)
        y0 = top - (row + 1) * (ph + lab) - row * rgap
        ax = axes_at(x0, y0, side, ph)
        panel(ax, shown[k], edges, colours, win, trained[k], node, lw)

        ax.set_title(
            f"$\\lambda={lams[k]:g}$",
            fontsize=7.0,
            pad=2.0,
            color=INK if trained[k] else "0.45",
            fontweight="bold" if trained[k] else "normal",
        )

    tick, hang, sgap = 0.42, 0.10, 0.20
    cw = (TEXT_W - 2 * pad - sgap) / 2
    mw = cw - tick - hang

    for which, (label, val, spec, nine, flip) in enumerate(strips):
        ax = axes_at(pad + which * (cw + sgap) + tick, mb, mw, mh)
        ylo = min(val.min(), spec.min(), nine.min())
        yhi = max(val.max(), spec.max(), nine.max())
        ax.vlines(lams[trained], ylo, yhi, color="0.86", lw=0.5, zorder=0)

        # the independent runs sit behind both curves as a broad band, so a curve that agrees
        # is legible as riding along it
        ax.plot(
            data["spec_lams"],
            spec,
            color="0.72",
            lw=5.2,
            zorder=1,
            solid_capstyle="round",
        )
        ax.plot(lams, nine, color=RED, lw=1.0, zorder=2)
        ax.plot(lams, val, color=INK, lw=1.0, zorder=3)

        # each run carries a ring on the lambdas it was trained at
        ax.plot(
            lams[trained9],
            nine[trained9],
            ls="none",
            marker="o",
            ms=3.0,
            mfc="white",
            mec=RED,
            mew=0.8,
            zorder=5,
        )
        ax.plot(
            lams[trained],
            val[trained],
            ls="none",
            marker="o",
            ms=3.4,
            mfc="white",
            mec=INK,
            mew=0.8,
            zorder=6,
        )

        ax.set_xlabel("$\\lambda$", fontsize=6.8, color=INK, labelpad=1)
        ax.set_ylabel(label, fontsize=6.8, color=INK, labelpad=2)

        if flip:
            ax.invert_yaxis()

        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.tick_params(labelsize=6.0, colors=INK, width=0.6, length=2.2, pad=1)

        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(INK)
            ax.spines[spine].set_linewidth(0.6)

    outdir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context({"savefig.bbox": None}):
        for ext, kw in (("png", {"dpi": 600}), ("pdf", {})):
            fig.savefig(outdir / f"{name}.{ext}", **kw)

    plt.close(fig)

    print(f"saved {outdir / name}.pdf  {TEXT_W} x {height:.2f} in", flush=True)


def main(seeds, iters, device, outdir, cache, nrow, ncol):
    for which, seed in enumerate(seeds):
        print(f"=== seed {seed}", flush=True)

        data = compute(seed, iters, device, cache)
        name = "fam_family" if which == 0 else f"fam_family_seed{seed}"

        sheet(data, outdir, name, nrow, ncol)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[0],
        help="the figure is drawn from the first seed, the rest get their own file",
    )
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", type=Path, default=FIGDIR)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--nrow", type=int, default=1)
    ap.add_argument("--ncol", type=int, default=7)
    a = ap.parse_args()

    main(a.seeds, a.iters, a.device, a.outdir, a.cache, a.nrow, a.ncol)
