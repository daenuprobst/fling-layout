import argparse
import collections
import glob
import json
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import networkx as nx
import numpy as np
import s_gd2
import torch

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fling import Fling, FlingStress, FlingVis, neulay
from fling import figstyle as fs
from fling.baselines import tsnet, tsnet_star, _pmds_embed, _train_mlp
from fling.data import scaling_graphs
from fling.graphs import bfs_rows

import matplotlib.pyplot as plt

SIZES = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
TIME_CAP_S = 1200  # 20 min: a run past this gates the method's larger sizes

# label -> (cache method key, device). Order sets the legend order.
RUNS = [
    ("Fling (GPU)", "Fling", "cuda"),
    ("FlingStress (GPU)", "FlingStress", "cuda"),
    ("FlingVis (GPU)", "FlingVis", "cuda"),
    ("Fling (CPU)", "Fling", "cpu"),
    ("FlingStress (CPU)", "FlingStress", "cpu"),
    ("FlingVis (CPU)", "FlingVis", "cpu"),
    ("s_gd2 (sparse)", "s_gd2_sparse", "cpu"),
    ("NNP-NET", "nnpnet", "cpu"),
    ("tsNET", "tsnet", "cpu"),
    ("NeuLay (CPU)", "neulay", "cpu"),
    ("NeuLay (GPU)", "neulay", "cuda"),
]
MARKER = {
    "Fling": "o",
    "FlingStress": "D",
    "FlingVis": "^",
    "s_gd2": "s",
    "tsNET": "v",
    "NNP-NET": "P",
    "NeuLay": "X",
}
SMALL = dict(title=7.0, label=6.8, tick=6.0, legend=5.6)


def _geodesics32(adj, sources, cols=None, chunk=128):
    ncol = adj.shape[0] if cols is None else len(cols)
    dist = np.empty((len(sources), ncol), np.float32)

    for b in range(0, len(sources), chunk):
        rows = bfs_rows(adj, list(sources[b : b + chunk]))
        dist[b : b + chunk] = rows if cols is None else rows[:, cols]

    return dist


def _mem_total_b():
    with open("/proc/meminfo") as f:
        return int(f.readline().split()[1]) * 1024


def _tsnet(graph, seed):
    n = graph.number_of_nodes()

    if 20 * n * n > 0.8 * _mem_total_b():
        raise MemoryError(f"tsNET needs ~{20 * n * n / 2**30:.0f} GiB dense at N={n}")

    adj = nx.adjacency_matrix(graph)

    return tsnet(_geodesics32(adj, np.arange(adj.shape[0])), seed=seed)


def _nnpnet(graph, seed, train_cap=5000, npiv=250, emb=50):
    n = graph.number_of_nodes()
    rng = np.random.default_rng(seed)
    adj = nx.adjacency_matrix(graph)

    train = np.sort(rng.choice(n, size=min(n, train_cap), replace=False))
    tset = np.zeros(n, bool)
    tset[train] = True

    p = int(rng.choice(train))  # farthest-first pivots among train
    piv = []
    cols = np.empty((n, min(npiv, len(train))), np.float32)
    dmin = None

    for k in range(cols.shape[1]):
        col = bfs_rows(adj, [p])[0]
        cols[:, k] = col
        piv.append(p)
        dmin = col if dmin is None else np.minimum(dmin, col)
        cand = np.where(tset, dmin, -1.0)
        cand[piv] = -1.0
        p = int(np.argmax(cand))

        if p in piv:
            cols = cols[:, : k + 1]
            break

    latent = torch.tensor(_pmds_embed(cols, emb), dtype=torch.float32)
    dtr = _geodesics32(adj, train, cols=train)  # train x train geodesics only
    ref = tsnet_star(dtr, seed)
    ref = (ref - ref.mean(0)) / (ref.std(0) + 1e-9)
    net = _train_mlp(latent, ref, train, epochs=40, batch=64, lr=1e-3, seed=seed)

    with torch.no_grad():
        return net(latent).cpu().numpy().astype(np.float64)


def _method(name):
    if name == "s_gd2_sparse":

        def sparse(graph, seed):
            edges = np.array(graph.edges())
            return np.asarray(
                s_gd2.layout_sparse(
                    edges[:, 0].astype(np.int32),
                    edges[:, 1].astype(np.int32),
                    min(200, graph.number_of_nodes() - 1),
                    random_seed=seed,
                ),
                float,
            )

        return sparse

    if name == "tsnet":
        return _tsnet

    if name == "nnpnet":
        return _nnpnet

    if name == "neulay":
        return lambda graph, seed: neulay(graph, seed=seed)

    cls = {"Fling": Fling, "FlingStress": FlingStress, "FlingVis": FlingVis}[name]

    return lambda graph, seed: cls(n_components=2, seed=seed).fit_transform(graph)


class _RssSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._stop = threading.Event()
        self.peak_mb = 0.0
        self._page = os.sysconf("SC_PAGESIZE") / 2**20

    def run(self):
        while not self._stop.is_set():
            try:
                with open("/proc/self/statm") as f:
                    self.peak_mb = max(
                        self.peak_mb, int(f.read().split()[1]) * self._page
                    )
            except OSError:
                pass

            self._stop.wait(0.05)

    def stop(self):
        self._stop.set()
        self.join()


def _host_hwm_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0

    return 0.0


def worker(method, n, seed, device):
    torch.set_default_device(device)

    if device == "cpu":
        # backstop: a hungry run must MemoryError-gate rather than push the host into swap
        cap = int(_mem_total_b() * 0.85)
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    fn = _method(method)
    graph = scaling_graphs([n])[0][1]

    if device == "cuda":  # untimed warm-up: CUDA context + first-kernel cost
        _method(method)(
            nx.convert_node_labels_to_integers(nx.les_miserables_graph()), 0
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    try:  # measure the timed span, not the graph build
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")  # reset the kernel RSS high-water mark
    except OSError:
        pass

    sampler = _RssSampler()
    sampler.start()
    t0 = time.perf_counter()
    out = fn(graph, seed)

    if device == "cuda":
        torch.cuda.synchronize()

    secs = time.perf_counter() - t0
    _ = np.asarray(out[0] if isinstance(out, tuple) else out)  # force materialization

    sampler.stop()

    mem = max(_host_hwm_mb(), sampler.peak_mb)
    gpu = torch.cuda.max_memory_reserved() / 2**20 if device == "cuda" else 0.0

    print(
        "RESULT "
        + json.dumps(
            dict(
                t=secs,
                mem_mb=mem,
                gpu_mb=gpu,
                n=graph.number_of_nodes(),
                e=graph.number_of_edges(),
            )
        )
    )


def _cache_path(cache_dir, method, device, n, seed):
    return cache_dir / f"{method}_{device}_n{n}_seed{seed}.json"


def _write(path, rec):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(path)


def _log(prog, msg):
    prog.write(f"{time.strftime('%H:%M:%S')}  {msg}\n")
    prog.flush()
    print(msg, flush=True)


def measure(runs, sizes, seeds, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    prog = open(cache_dir / "progress.txt", "a")

    with open("/proc/loadavg") as f:
        _log(prog, "sweep started, loadavg: " + f.read().strip())

    for label, method, device in runs:
        for seed in seeds:
            for n in sizes:
                path = _cache_path(cache_dir, method, device, n, seed)

                if path.exists():
                    rec = json.loads(path.read_text())

                    if "gated" in rec:
                        _log(
                            prog,
                            f"{label:18s} N={n:<8} gated (cached: {rec['gated'][:80]})",
                        )
                        break

                    _log(
                        prog,
                        f"{label:18s} N={n:<8} seed={seed} cached: t={rec['t']:8.1f}s "
                        f"mem={rec['mem_mb']:7.0f}MB gpu={rec['gpu_mb']:6.0f}MB",
                    )

                    if rec["t"] > TIME_CAP_S:
                        break

                    continue

                cmd = [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--single",
                    method,
                    str(n),
                    str(seed),
                    device,
                ]

                try:
                    # headroom over the cap: the graph build is untimed but not free
                    p = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=TIME_CAP_S + 600
                    )

                    line = next(
                        (l for l in p.stdout.splitlines() if l.startswith("RESULT ")),
                        None,
                    )

                    if line is None:
                        err = (p.stderr.strip().splitlines() or ["no RESULT"])[-1][:200]
                        _write(path, dict(gated=err))
                        _log(prog, f"{label:18s} N={n:<8} GATED ({err[:80]})")
                        break

                    rec = json.loads(line[len("RESULT ") :])
                    _write(path, rec)
                    _log(
                        prog,
                        f"{label:18s} N={rec['n']:<8} seed={seed} t={rec['t']:8.1f}s "
                        f"mem={rec['mem_mb']:7.0f}MB gpu={rec['gpu_mb']:6.0f}MB",
                    )

                    if rec["t"] > TIME_CAP_S:
                        _log(
                            prog,
                            f"{label:18s} exceeded {TIME_CAP_S}s cap -> gating larger N",
                        )
                        break

                except subprocess.TimeoutExpired:
                    _write(path, dict(gated=f"timeout > {TIME_CAP_S + 600}s"))
                    _log(prog, f"{label:18s} N={n:<8} TIMEOUT -> gating")
                    break

    _log(prog, "sweep finished")
    prog.close()


def load(cache_dir):
    cells = collections.defaultdict(lambda: collections.defaultdict(list))

    for path in glob.glob(str(cache_dir / "*.json")):
        stem = os.path.basename(path)[:-5]
        head, seed = stem.rsplit("_seed", 1)
        head, n = head.rsplit("_n", 1)
        method, device = head.rsplit("_", 1)

        try:
            rec = json.loads(Path(path).read_text())
        except json.JSONDecodeError:
            continue

        if "gated" in rec or rec.get("t") is None:
            continue

        cells[f"{method}|{device}"][int(n)].append(rec)

    out = {}
    for key, by_n in cells.items():
        out[key] = [
            dict(
                n=n,
                t=float(np.mean([r["t"] for r in rs])),
                mem=float(np.mean([r["mem_mb"] + r.get("gpu_mb", 0.0) for r in rs])),
                seeds=len(rs),
            )
            for n, rs in sorted(by_n.items())
        ]

    return out


def plot(data, out_dir, width, height, name):
    fs.use()

    # 2x2 rather than 1x4: at full width each panel is twice as wide, which is what separates
    # the curves. Rows are the metric, columns the device, so a row shares its y scale.
    fig, axes = plt.subplots(2, 2, figsize=(width, height), sharex=True)

    for idx, (device, metric) in enumerate(
        (("cpu", "t"), ("cuda", "t"), ("cpu", "mem"), ("cuda", "mem"))
    ):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        for label, key, dev in RUNS:
            base = label.split(" (")[0]
            split = label.endswith("(CPU)") or label.endswith("(GPU)")

            if split and dev != device:
                continue

            rs = data.get(f"{key}|{dev}", [])

            if not rs:
                continue

            ours = fs.is_ours(base)
            # the panel title already says CPU or GPU, so both device curves share one entry
            ax.plot(
                [r["n"] for r in rs],
                [r[metric] for r in rs],
                color=fs.method_color(base),
                marker=MARKER[base],
                lw=1.2 if ours else 0.8,
                ms=2.8 if ours else 2.2,
                zorder=3 if ours else 2,
                label=base if idx == 0 else None,
            )

        if metric == "t":
            ax.set_title(
                "CPU" if device == "cpu" else "GPU",
                fontsize=SMALL["title"],
                color="0.15",
                pad=3,
            )
            ax.axhline(TIME_CAP_S, ls=":", lw=0.7, color="0.55", zorder=1)

        if col == 0:
            ax.set_ylabel(
                "wall-clock (s)" if metric == "t" else "peak memory (MB)",
                fontsize=SMALL["label"],
                labelpad=1,
            )
        if row == 1:
            ax.set_xlabel("$N$ (nodes)", fontsize=SMALL["label"], labelpad=1)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(labelsize=SMALL["tick"], pad=1.0, length=2.0)
        ax.set_box_aspect(1.0)
        fs.grid(ax, axis="both")

    axes[0, 0].text(
        1400,
        TIME_CAP_S * 1.25,
        "20-min cap",
        fontsize=SMALL["tick"],
        color="0.45",
        va="bottom",
    )
    axes[1, 1].text(
        0.04,
        0.96,
        "host + device",
        transform=axes[1, 1].transAxes,
        fontsize=SMALL["tick"],
        color="0.45",
        va="top",
    )

    handles, labels = [], []
    for hh, ll in zip(*axes[0, 0].get_legend_handles_labels()):
        if ll not in labels:
            handles.append(hh)
            labels.append(ll)

    fig.legend(
        handles,
        labels,
        ncol=max(len(labels), 1),
        fontsize=SMALL["legend"],
        handlelength=1.0,
        columnspacing=0.7,
        handletextpad=0.3,
        borderpad=0.0,
        borderaxespad=0.0,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
    )
    fig.tight_layout(pad=0.3, w_pad=1.0, h_pad=0.5, rect=(0, 0.048, 1, 1.0))

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=220)
        print(f"wrote {out_dir / name}.{ext}")

    print(f"\n{'series':<22}{'N range':<24}{'seeds'}")
    for label, key, dev in RUNS:
        rs = data.get(f"{key}|{dev}", [])
        if rs:
            print(
                f"{label:<22}{rs[0]['n']:>7,} to {rs[-1]['n']:>9,}      "
                f"{sorted({r['seeds'] for r in rs})}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--sizes", type=int, nargs="*", default=SIZES)
    ap.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="cache keys to restrict the sweep to, e.g. Fling s_gd2_sparse",
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "figures")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "scaling")
    ap.add_argument("--name", default="exp_scaling_2x2")
    ap.add_argument("--width", type=float, default=5.5)
    ap.add_argument("--height", type=float, default=5.15)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    runs = [r for r in RUNS if args.methods is None or r[1] in args.methods]

    if not args.plot_only:
        measure(runs, args.sizes, args.seeds, args.cache_dir)

    plot(load(args.cache_dir), args.out_dir, args.width, args.height, args.name)


if __name__ == "__main__":
    if len(sys.argv) >= 6 and sys.argv[1] == "--single":
        worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
        sys.exit(0)

    main()
