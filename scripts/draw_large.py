import argparse
import json
import pathlib
import sys

import colorcet
import datashader as ds
import datashader.transfer_functions as tf
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fling.graphs import edge_array
from million import arm_tag, build, graph_tag

CMAP = {
    "fling": ["#0b1026", "#2a4d8f", "#4fa3d1", "#bfe6f5"],
    "flingstress": ["#0b1026", "#2a4d8f", "#4fa3d1", "#bfe6f5"],
    "sgd2": ["#140b05", "#7a3b12", "#d17a2a", "#f5dcbf"],
}


def edge_frame(pos, edges):
    # datashader draws a polyline per run of rows, so a NaN row breaks one edge from the next
    n = len(edges)
    x = np.full(3 * n, np.nan, np.float32)
    y = np.full(3 * n, np.nan, np.float32)
    x[0::3], x[1::3] = pos[edges[:, 0], 0], pos[edges[:, 1], 0]
    y[0::3], y[1::3] = pos[edges[:, 0], 1], pos[edges[:, 1], 1]

    return pd.DataFrame({"x": x, "y": y})


def shade(pos, edges, px, cmap, how="eq_hist", line_width=0.7):
    pos = np.asarray(pos, np.float32)
    pos = pos - pos.mean(0)
    lim = 1.02 * np.abs(pos).max()
    canvas = ds.Canvas(
        plot_width=px, plot_height=px, x_range=(-lim, lim), y_range=(-lim, lim)
    )
    agg = canvas.line(
        edge_frame(pos, edges), "x", "y", agg=ds.count(), line_width=line_width
    )

    return tf.shade(agg, cmap=cmap, how=how)


def render(pos, edges, px, arm, how="eq_hist"):
    img = shade(pos, edges, px, CMAP[arm], how)

    return tf.set_background(img, CMAP[arm][0]).to_pil()


def render_labels(pos, labels, px, how="eq_hist", background="#101010"):
    # one colour per cluster, so a good drawing is one where the colours land in separate islands
    pos = np.asarray(pos, np.float32)
    pos = pos - pos.mean(0)
    lim = 1.02 * np.abs(pos).max()
    frame = pd.DataFrame(
        {
            "x": pos[:, 0],
            "y": pos[:, 1],
            "cat": pd.Categorical(np.asarray(labels).astype(str)),
        }
    )
    canvas = ds.Canvas(
        plot_width=px, plot_height=px, x_range=(-lim, lim), y_range=(-lim, lim)
    )
    agg = canvas.points(frame, "x", "y", ds.count_cat("cat"))
    key = {
        c: colorcet.glasbey_dark[i % len(colorcet.glasbey_dark)]
        for i, c in enumerate(frame["cat"].cat.categories)
    }
    img = tf.shade(agg, color_key=key, how=how)

    return tf.set_background(img, background).to_pil()


def main(
    kind, n, seed, px, out, cache, data, how, alpha, diff_k, landmarks, arms, npiv=None
):
    tag = graph_tag(kind, n)
    edges = edge_array(build(kind, n, data))
    out.mkdir(parents=True, exist_ok=True)

    for arm in arms:
        name = arm_tag(arm, alpha, diff_k, landmarks, npiv)
        stats = json.loads((cache / f"{name}_{tag}_seed{seed}.json").read_text())
        pos = np.load(cache / f"{name}_{tag}_seed{seed}.npy")
        path = out / f"{tag}_{name}.png"
        render(pos, edges, px, arm, how).save(path)
        print(f"{path}  {stats['t']:.0f}s  stress {stats['stress']:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--graph", default="roadnet", choices=["delaunay", "roadnet", "neurons"]
    )
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--px", type=int, default=1400)
    p.add_argument("--how", default="eq_hist")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--diff-k", type=int, default=20)
    p.add_argument("--landmarks", type=int, default=64)
    p.add_argument("--npiv", type=int, default=None)
    p.add_argument("--arms", nargs="*", default=["fling", "sgd2"])
    p.add_argument("--data", type=pathlib.Path, default=ROOT / "data")
    p.add_argument("--cache", type=pathlib.Path, default=ROOT / "cache" / "million")
    p.add_argument("--out", type=pathlib.Path, default=ROOT / "figures")
    a = p.parse_args()
    main(
        a.graph,
        a.n,
        a.seed,
        a.px,
        a.out,
        a.cache,
        a.data,
        a.how,
        a.alpha,
        a.diff_k,
        a.landmarks,
        a.arms,
        a.npiv,
    )
