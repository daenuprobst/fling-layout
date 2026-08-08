import os
from pathlib import Path

import matplotlib

matplotlib.use(
    "Agg"
)  # must be set before pyplot is imported, so this import stays below it
import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parents[1] / "figures"

WIDTH_1COL = (3.4, 2.6)
WIDTH_2COL = (7.0, 3.0)
WIDTH_FULL = (7.0, 4.4)

CMAP_NODE = "coolwarm"  # node scalars (Fiedler vector, field value): red/blue
CMAP_CLASS = "tab10"  # categorical node classes
CMAP_FIELD = "coolwarm"  # background field / heatmap
MESH_BLUE = "#2b6cb0"  # mesh wireframes
EDGE_GREY = "0.62"
GOOD = "#1a7f37"  # highlight for the best value in a row/group

_METHOD_COLORS = {
    "Fling": "#d1495b",
    "FlingFR": "#e79892",
    "FlingStress": "#8b2f47",
    "FlingVis": "#e8871a",
    "NNP-NET": "#2a6f97",
    # teal for s_gd2 (green would collide with our red under deuteranopia), indigo for tsNET
    "tsNET": "#8fa8ee",
    "s_gd2": "#12a698",
    "PivotMDS": "#8d99ae",
    "ForceAtlas2": "#6a4c93",
    "FR": "#b8b8d1",
    "NeuLay": "#7f7f7f",
}
_ALIAS = {  # every spelling the scripts use -> one canonical name
    "FLING-fast": "FlingFR",
    "F-fast": "FlingFR",
    "F-fast-pivot": "FlingFR",
    "F-shell": "Fling",
    "FLING-stress": "FlingStress",
    "F-stress": "FlingStress",
    "F-stress-pivot": "FlingStress",
    "FLING-NE": "FlingVis",
    "F-NE": "FlingVis",
    "F-NE-spec": "FlingVis",
    "FA2": "ForceAtlas2",
    "NNP-Net": "NNP-NET",
    "Fling (FR)": "FlingFR",
    "FlingFR ": "FlingFR",
}

METHOD_ORDER = [
    "Fling",
    "FlingFR",
    "FlingStress",
    "FlingVis",
    "NNP-NET",
    "tsNET",
    "s_gd2",
    "PivotMDS",
    "ForceAtlas2",
    "FR",
    "NeuLay",
]


def canonical(name):
    return _ALIAS.get(name, name)


def method_color(name):
    return _METHOD_COLORS.get(canonical(name), "0.5")


def is_ours(name):
    return canonical(name).startswith("Fling")


def use():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "0.35",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.color": "0.35",
            "ytick.color": "0.35",
            "axes.labelcolor": "0.15",
            "text.color": "0.15",
            "grid.color": "0.9",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "lines.markersize": 4,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def panel(ax, letter, dx=-0.02, dy=1.04):
    ax.text(
        dx,
        dy,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="0.1",
    )


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, zorder=0)
    ax.set_axisbelow(True)


def legend(ax, **kwargs):
    kwargs.setdefault("loc", "best")
    kwargs.setdefault("handlelength", 1.4)

    return ax.legend(**kwargs)


def save(fig, name, outdir=None):
    outdir = str(outdir or FIGURES)
    os.makedirs(outdir, exist_ok=True)

    # both formats are written, so the caller can include vector for line art and raster for
    # anything with a large scatter in it
    for ext, kw in (("png", {"dpi": 600}), ("pdf", {})):
        fig.savefig(f"{outdir}/{name}.{ext}", bbox_inches="tight", **kw)

    plt.close(fig)
    print(f"saved {outdir}/{name}.png + .pdf", flush=True)
