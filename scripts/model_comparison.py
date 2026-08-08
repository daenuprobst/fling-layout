import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.stats import friedmanchisquare, rankdata
import torch

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fling import Fling, FlingStress, FlingVis, make_graph, neulay
from fling import data as fdata
from fling import figstyle as fs
from fling.baselines import NNPNet, all_pairs_D, fa2, sgd_stress, tsnet
from fling.metrics import _neighbor_sets, evaluate, fast_D

# below figstyle, which selects the Agg backend before pyplot is imported
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

GRAPH_SPECS = [
    ("grid_400", lambda: make_graph("grid", 400)),
    ("tree_400", lambda: make_graph("tree", 400)),
    ("delaunay_600", lambda: make_graph("delaunay", 600)),
    ("sbm_240", lambda: fdata._sbm((80, 80, 80))),
    ("lesmis", lambda: nx.les_miserables_graph()),
    ("dwt_1005", lambda: fdata._load_mtx("dwt_1005")),
    ("3elt", lambda: fdata._load_mtx("3elt")),
    ("jagmesh8", lambda: fdata._load_mtx("jagmesh8")),
    (
        "ego-Facebook",
        lambda: nx.read_edgelist(fdata.DATA / "facebook_combined.txt", nodetype=int),
    ),
]
ALL_GRAPHS = [nm for nm, _ in GRAPH_SPECS]
QUICK_GRAPHS = ALL_GRAPHS[:5]
MESHES = {"grid_400", "delaunay_600", "dwt_1005", "3elt", "jagmesh8"}


def build_graph(name):
    spec = dict(GRAPH_SPECS)[name]
    graph = spec()

    return nx.convert_node_labels_to_integers(
        graph.subgraph(max(nx.connected_components(graph), key=len))
    )


def _landmarks(dmat, k, seed):
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(dmat)))]
    mind = dmat[chosen[0]].copy()

    for _ in range(k - 1):
        chosen.append(int(np.argmax(mind)))
        mind = np.minimum(mind, dmat[chosen[-1]])

    return chosen


def landmark_mds_layout(graph, seed=0, k=32):
    # de Silva & Tenenbaum: classical MDS on the k x k block, the rest by distance-based extension
    dmat = fast_D(graph)
    src = _landmarks(dmat, min(k, dmat.shape[0] - 1), seed)
    sub = dmat[np.ix_(src, src)].astype(float) ** 2
    m = len(src)
    cen = np.eye(m) - np.ones((m, m)) / m
    val, vec = np.linalg.eigh(-0.5 * cen @ sub @ cen)
    keep = np.argsort(val)[::-1][:2]
    lam = np.clip(val[keep], 1e-12, None)
    pseudo = (vec[:, keep] / np.sqrt(lam)).T

    return (-0.5 * pseudo @ ((dmat[:, src].astype(float) ** 2) - sub.mean(0)).T).T


def pivot_mds_layout(graph, seed=0, k=32):
    # Brandes & Pich: classical scaling of the full n x k pivot block, not just the k x k corner
    dmat = fast_D(graph)
    src = _landmarks(dmat, min(k, dmat.shape[0] - 1), seed)
    c = dmat[:, src].astype(float) ** 2
    c = c - c.mean(0, keepdims=True) - c.mean(1, keepdims=True) + c.mean()
    c *= -0.5
    _, _, vt = np.linalg.svd(c, full_matrices=False)

    return c @ vt[:2].T


ROWS = [
    ("Fling", lambda G, s, k: Fling(n_components=2, seed=s).fit_transform(G)),
    (
        "FlingStress",
        lambda G, s, k: FlingStress(n_components=2, seed=s).fit_transform(G),
    ),
    ("FlingVis", lambda G, s, k: FlingVis(n_components=2, seed=s).fit_transform(G)),
    ("LandmarkMDS", lambda G, s, k: landmark_mds_layout(G, seed=s, k=k)),
    ("PivotMDS", lambda G, s, k: pivot_mds_layout(G, seed=s, k=k)),
    ("s_gd2", lambda G, s, k: sgd_stress(G, seed=s)),
    ("ForceAtlas2", lambda G, s, k: fa2(G)),  # fa2_modified has no seed argument
    ("tsNET", lambda G, s, k: tsnet(all_pairs_D(G), seed=s)),
    ("NNP-NET", lambda G, s, k: NNPNet(n_components=2, seed=s).fit_transform(G)),
    ("NeuLay", lambda G, s, k: neulay(G, seed=s)[0]),
]
METHODS = [m for m, _ in ROWS]
OURS = {"Fling", "FlingStress", "FlingVis"}

# s_gd2 carries free coordinates and optimises the full all-pairs stress, so it is a quality
# reference rather than a competitor and is excluded from the rank summary only.
RANK_EXCLUDE = {"s_gd2"}

# (tex tag, metric key, rank-table header, direction, decimals)
TABLES = [
    ("stress", "stress", "stress", "min", 3),
    ("np", "np_jaccard", "neigh.\\ pres.", "max", 3),
    ("edgecov", "edge_cov", "edge-length CoV", "min", 3),
    ("cross", "crosslessness", "crosslessness", "max", 3),
]

# the four per-metric tables are emitted in the paper's order, which is not the rank-table order
TABLE_ORDER = ["stress", "np", "cross", "edgecov"]

# Nemenyi q at alpha=0.05, indexed by number of methods
Q05 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
    11: 3.219,
    12: 3.268,
}

MAIN_METHODS = [
    "Fling",
    "FlingStress",
    "FlingVis",
    "NeuLay",
    "tsNET",
    "NNP-NET",
    "s_gd2",
]
EXTRA_METHODS = ["LandmarkMDS", "PivotMDS", "ForceAtlas2"]

BASE = "#00CDDB"


def _shade(hexcol, f):
    r, g, b = (int(hexcol[i : i + 2], 16) / 255 for i in (1, 3, 5))

    if f <= 1:
        return (r * f, g * f, b * f)

    return tuple(min(1, c + (1 - c) * (f - 1)) for c in (r, g, b))


C_NODE, C_EDGE, C_MESH = _shade(BASE, 0.62), _shade(BASE, 0.95), _shade(BASE, 0.78)


def esc(name):
    return name.replace("_", r"\_")


class Progress:
    def __init__(self, path):
        self.fh = open(path, "a")
        self.log("--", "--", "run started")

    def log(self, gname, mname, msg):
        line = f"{time.strftime('%H:%M:%S')}  {gname:14s} {mname:12s} {msg}"
        self.fh.write(line + "\n")
        self.fh.flush()
        print(line, flush=True)


class Runner:
    def __init__(self, args):
        self.cache = Path(args.cache_dir)
        self.layouts = self.cache / "layouts"
        self.metrics_json = self.cache / "metrics.json"
        self.layouts.mkdir(parents=True, exist_ok=True)
        self.device = args.device
        self.timeout = args.timeout
        self.data_dir = args.data_dir
        self.no_compute = args.no_compute
        self.n_land = args.landmarks
        self.seeds = args.seeds
        self.prog = Progress(self.cache / "progress.txt")

    def paths(self, mname, gname, seed):
        slug = f"{mname.replace(' ', '')}__{gname}__s{seed}"

        return self.layouts / f"{slug}.npy", self.layouts / f"{slug}.na"

    def layout(self, mname, gname, seed):
        npy, na = self.paths(mname, gname, seed)

        if npy.exists():
            self.prog.log(gname, mname, f"s{seed} cached")

            return np.load(npy)

        if self.no_compute:
            raise SystemExit(
                f"--no-compute: {mname} on {gname} seed {seed} is not in the cache ({npy})"
            )

        if na.exists():
            self.prog.log(
                gname, mname, f"s{seed} n/a ({na.read_text().strip()[:60]}, cached)"
            )

            return None

        t0 = time.perf_counter()
        cmd = [
            sys.executable,
            __file__,
            "--one",
            mname,
            gname,
            str(seed),
            "--device",
            self.device,
            "--cache-dir",
            str(self.cache),
            "--landmarks",
            str(self.n_land),
        ]

        if self.data_dir:
            cmd += ["--data-dir", str(self.data_dir)]

        try:
            subprocess.run(cmd, timeout=self.timeout, check=True, capture_output=True)
            self.prog.log(gname, mname, f"s{seed} ok  {time.perf_counter() - t0:6.1f}s")

            return np.load(npy)
        except subprocess.TimeoutExpired:
            reason = f"timeout >{self.timeout}s"
        except subprocess.CalledProcessError as exc:
            reason = "error: " + (exc.stderr or b"").decode()[-200:].replace("\n", " ")

        na.write_text(reason)
        self.prog.log(gname, mname, f"s{seed} n/a ({reason[:80]})")

        return None

    def metric_rows(self, gs):
        cache = (
            json.loads(self.metrics_json.read_text())
            if self.metrics_json.exists()
            else {}
        )
        out = {}

        for gname, graph in gs:
            need = any(
                f"{m}|{gname}|{s}" not in cache for m in METHODS for s in self.seeds
            )
            dmat = nbrs = None

            if need:
                dmat, nbrs = fast_D(graph), _neighbor_sets(graph)

            for mname in METHODS:
                for seed in self.seeds:
                    key = f"{mname}|{gname}|{seed}"

                    if key not in cache:
                        pos = self.layout(mname, gname, seed)
                        cache[key] = (
                            None
                            if pos is None
                            else evaluate(
                                pos,
                                graph,
                                D=dmat,
                                nbrs=nbrs,
                                cross_max_e=4000,
                                angle_max_e=0,
                            )
                        )
                        tmp = str(self.metrics_json) + ".tmp"
                        Path(tmp).write_text(json.dumps(cache, indent=1))
                        os.replace(tmp, self.metrics_json)

                    out[(mname, gname, seed)] = cache[key]

        return out


def compute_one(mname, gname, seed, cache_dir, n_land, device):
    if device != "cpu":
        torch.set_default_device(device)

    graph = build_graph(gname)
    pos = np.asarray(dict(ROWS)[mname](graph, seed, n_land), float)[:, :2]
    slug = f"{mname.replace(' ', '')}__{gname}__s{seed}"
    np.save(Path(cache_dir) / "layouts" / f"{slug}.npy", pos)


def stats_of(rows, mname, gname, seeds, key):
    vals = [
        rows[(mname, gname, s)][key]
        for s in seeds
        if rows[(mname, gname, s)] is not None
        and rows[(mname, gname, s)][key] is not None
    ]

    if not vals:
        return None

    return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)


def write_metric_tables(rows, graph_names, seeds, tables_dir):
    by_tag = {tag: (key, direction, dec) for tag, key, _, direction, dec in TABLES}

    for tag in TABLE_ORDER:
        key, direction, dec = by_tag[tag]
        tol = 0.5 * 10**-dec
        lines = [
            r"\begin{tabular}{l" + "c" * len(METHODS) + "}",
            r"\toprule",
            "graph & " + " & ".join(esc(m) for m in METHODS) + r" \\",
            r"\midrule",
        ]

        for gname in graph_names:
            stats = [stats_of(rows, m, gname, seeds, key) for m in METHODS]
            pick = min if direction == "min" else max
            means = [s[0] for s in stats if s is not None]
            best = pick(means) if means else None
            rest = [mu for mu in means if abs(mu - best) >= tol]
            second = pick(rest) if rest else None
            cells = []

            for stat in stats:
                if stat is None:
                    cells.append("--")
                    continue

                mu, sd = stat
                txt = f"{mu:.{dec}f}"

                if best is not None and abs(mu - best) < tol:
                    txt = r"\textbf{" + txt + "}"
                elif second is not None and abs(mu - second) < tol:
                    txt = r"\underline{" + txt + "}"

                cells.append(txt + r"\,$\pm$\," + f"{sd:.{dec}f}")

            lines.append(esc(gname) + " & " + " & ".join(cells) + r" \\")

        lines += [r"\bottomrule", r"\end{tabular}"]
        path = tables_dir / f"model_comparison_v5_{tag}.tex"
        path.write_text("\n".join(lines) + "\n")
        print(f"wrote {path}", flush=True)


def write_rank_table(rows, graph_names, seeds, tables_dir):
    methods = [m for m in METHODS if m not in RANK_EXCLUDE]
    ranks, ngraph, pvals = {}, {}, {}

    for tag, key, _, direction, _ in TABLES:
        acc = {m: [] for m in methods}

        for gname in graph_names:
            means = {}

            for mname in methods:
                stat = stats_of(rows, mname, gname, seeds, key)

                if stat is not None:
                    means[mname] = stat[0]

            if len(means) < len(methods):  # a metric the gate skipped on this graph
                continue

            vals = np.array([means[m] for m in methods])

            for mname, rank in zip(
                methods, rankdata(vals if direction == "min" else -vals)
            ):
                acc[mname].append(float(rank))

        ranks[key] = {m: (float(np.mean(a)) if a else None) for m, a in acc.items()}
        ngraph[key] = len(acc[methods[0]])
        pvals[key] = (
            friedmanchisquare(*[np.array(acc[m]) for m in methods]).pvalue
            if ngraph[key] >= 3
            else float("nan")
        )

    overall = {
        m: float(
            np.mean(
                [ranks[k][m] for _, k, _, _, _ in TABLES if ranks[k][m] is not None]
            )
        )
        for m in methods
    }
    order = sorted(methods, key=lambda m: overall[m])

    k = len(methods)
    cd = {
        key: (
            Q05[k] * np.sqrt(k * (k + 1) / (6 * ngraph[key])) if ngraph[key] else np.inf
        )
        for _, key, _, _, _ in TABLES
    }
    best = {
        key: (
            min(v for v in ranks[key].values() if v is not None)
            if any(v is not None for v in ranks[key].values())
            else None
        )
        for _, key, _, _, _ in TABLES
    }

    lines = [
        "\\begin{tabular}{lr" + "r" * len(TABLES) + "}",
        "\\toprule",
        "method & overall & " + " & ".join(h for _, _, h, _, _ in TABLES) + " \\\\",
        "       &         & "
        + " & ".join(f"$n={ngraph[key]}$" for _, key, _, _, _ in TABLES)
        + " \\\\",
        "\\midrule",
    ]

    for mname in order:
        cells = []

        for _, key, _, _, _ in TABLES:
            rank = ranks[key][mname]

            if rank is None or best[key] is None:
                cells.append("--")
                continue

            txt = f"{rank:.2f}"

            if rank == best[key]:
                cells.append(f"$\\mathbf{{{txt}}}$")
            elif rank <= best[key] + cd[key]:  # not separable from the best by Nemenyi
                cells.append(f"$\\underline{{{txt}}}$")
            else:
                cells.append(f"${txt}$")

        lines.append(
            f"{esc(mname)} & ${overall[mname]:.2f}$ & " + " & ".join(cells) + " \\\\"
        )

    lines += ["\\bottomrule", "\\end{tabular}"]
    path = tables_dir / "model_comparison_v5_ranks.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}\n", flush=True)

    for _, key, header, _, _ in TABLES:
        print(
            f"  {header:<18} n={ngraph[key]}  Friedman p={pvals[key]:.2e}  "
            f"Nemenyi CD={cd[key]:.2f}",
            flush=True,
        )


def draw(gs, runner, methods, out, figures_dir, fig_seed, printed_w=5.5):
    fig, axes = plt.subplots(
        len(methods),
        len(gs),
        figsize=(1.9 * len(gs), 1.9 * len(methods)),
        squeeze=False,
    )
    # solve for the point size that lands at ~6.4pt on paper after LaTeX scales the figure
    pt = 6.4 / (printed_w / (1.9 * len(gs)))

    for col, (gname, graph) in enumerate(gs):
        edges = np.array(list(graph.edges()))
        mesh = gname in MESHES
        drawn = (
            edges
            if mesh
            else edges[
                np.random.default_rng(0).choice(
                    len(edges), min(len(edges), 12000), replace=False
                )
            ]
        )

        for row, mname in enumerate(methods):
            ax = axes[row, col]
            ours = mname in OURS
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_box_aspect(1)  # square axes box ...

            for spine in ax.spines.values():
                spine.set_color(_shade(BASE, 1.55) if ours else "0.85")
                spine.set_linewidth(0.7 if ours else 0.5)

            if row == 0:
                ax.set_title(gname, fontsize=pt, color="0.1", pad=8)

            if col == 0:
                # a row is one panel tall, so a long name has to shrink or the labels abut
                ax.set_ylabel(
                    mname,
                    fontsize=pt * min(1.0, 9.0 / len(mname)),
                    color=_shade(BASE, 0.55) if ours else "0.1",
                    labelpad=10,
                )

            pos = runner.layout(mname, gname, fig_seed)

            if pos is None:
                ax.text(
                    0.5,
                    0.5,
                    "n/a",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=17,
                    color="0.45",
                )
                continue

            # meshes are wireframes, so their edges carry the drawing: darker and less alpha
            ecol = (C_MESH if mesh else C_EDGE) if ours else ("0.4" if mesh else "0.62")
            ax.add_collection(
                LineCollection(
                    pos[drawn],
                    color=ecol,
                    lw=0.45 if mesh else 0.25,
                    alpha=0.5 if mesh else 0.25,
                    zorder=1,
                )
            )

            if not mesh:
                ax.scatter(
                    pos[:, 0],
                    pos[:, 1],
                    color=C_NODE if ours else "0.25",
                    s=2.2,
                    lw=0,
                    zorder=2,
                )

            ax.set_aspect("equal", adjustable="datalim")  # ... with 1:1 data units
            ax.autoscale_view()

    fig.tight_layout()
    fs.save(fig, out, outdir=figures_dir)


def write_manifest(gs, runner, args):
    def git(*cmd):
        try:
            return subprocess.run(
                ["git", *cmd], cwd=ROOT, capture_output=True, text=True, timeout=20
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return "unknown"

    man = {
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": args.device,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "seeds": args.seeds,
        "fig_seed": args.fig_seed,
        "n_landmarks": args.landmarks,
        "timeout_s": args.timeout,
        "arms": METHODS,
        "graphs": {
            nm: {"n": G.number_of_nodes(), "edges": G.number_of_edges()} for nm, G in gs
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "n_layouts_cached": len(list(runner.layouts.glob("*.npy"))),
    }
    path = runner.cache / "manifest.json"
    path.write_text(json.dumps(man, indent=1))
    print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument(
        "--graphs",
        nargs="+",
        default=None,
        help=f"subset of {ALL_GRAPHS} (default: all nine)",
    )
    ap.add_argument("--quick", action="store_true", help="the five small graphs only")
    ap.add_argument("--tables-dir", default=str(ROOT / "tables"))
    ap.add_argument("--figures-dir", default=str(ROOT / "figures"))
    ap.add_argument("--cache-dir", default=str(ROOT / "cache" / "model_comparison"))
    ap.add_argument(
        "--data-dir",
        default=None,
        help="directory holding 3elt/, dwt_1005/, jagmesh8/, facebook_combined.txt",
    )
    ap.add_argument(
        "--device",
        default="auto",
        help="torch device for the layout children (auto|cpu|cuda)",
    )
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per layout")
    ap.add_argument("--landmarks", type=int, default=32)
    ap.add_argument("--fig-seed", type=int, default=0)
    ap.add_argument(
        "--no-compute",
        action="store_true",
        help="fail instead of computing a missing cell, for restyling only",
    )
    ap.add_argument(
        "--one", nargs=3, metavar=("METHOD", "GRAPH", "SEED"), help=argparse.SUPPRESS
    )
    args = ap.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.data_dir:
        fdata.DATA = Path(args.data_dir)
    elif (ROOT / "data").exists():
        fdata.DATA = ROOT / "data"

    if args.one:
        compute_one(
            args.one[0],
            args.one[1],
            int(args.one[2]),
            args.cache_dir,
            args.landmarks,
            args.device,
        )
        return

    names = args.graphs or (QUICK_GRAPHS if args.quick else ALL_GRAPHS)
    unknown = [nm for nm in names if nm not in ALL_GRAPHS]
    assert not unknown, f"unknown graphs: {unknown}"

    tables_dir, figures_dir = Path(args.tables_dir), Path(args.figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fs.use()
    runner = Runner(args)
    gs = [(nm, build_graph(nm)) for nm in ALL_GRAPHS if nm in names]

    draw(
        gs, runner, MAIN_METHODS, "fig_model_comparison_v5", figures_dir, args.fig_seed
    )
    draw(
        gs,
        runner,
        EXTRA_METHODS,
        "fig_model_comparison_v5_extra",
        figures_dir,
        args.fig_seed,
    )

    rows = runner.metric_rows(gs)
    write_metric_tables(rows, [nm for nm, _ in gs], args.seeds, tables_dir)
    write_rank_table(rows, [nm for nm, _ in gs], args.seeds, tables_dir)

    if not args.no_compute:
        write_manifest(gs, runner, args)

    runner.prog.log("--", "--", "run finished")


if __name__ == "__main__":
    main()
