import argparse
import json
import pathlib
import resource
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd
import s_gd2
import scipy.sparse as sp
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fling import Fling, FlingStress, make_graph
from fling.graphs import bfs_rows, edge_array


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def sampled_stress(pos, adj, pivots):
    # exact stress is O(N^2), so score against BFS rows from a few pivots
    d = bfs_rows(adj, pivots)
    emb = np.linalg.norm(pos[None, :, :] - pos[pivots][:, None, :], axis=-1)
    ok = d > 0
    ratio = emb[ok] / d[ok]
    a = ratio.sum() / (ratio**2).sum()

    return float((((a * emb[ok] - d[ok]) / d[ok]) ** 2).mean())


def load_roadnet(path):
    frame = pd.read_csv(path, sep="\t", comment="#", header=None, names=["u", "v"])
    graph = nx.from_edgelist(frame.itertuples(index=False, name=None))
    keep = max(nx.connected_components(graph), key=len)

    return nx.convert_node_labels_to_integers(graph.subgraph(keep).copy())


def load_neurons(data, k=15):
    # 1.3M mouse brain cells from 10x. Nodes are cells and edges join each cell to its k nearest
    # in the 20 principal components 10x published, so the structure lives in expression space.
    cache = data / f"neurons_knn_k{k}.npz"

    if cache.exists():
        held = np.load(cache)
        edges, labels = held["edges"], held["labels"]
    else:
        import pynndescent

        pcs = pd.read_csv(data / "analysis/pca/20_components/projection.csv")
        labels = pd.read_csv(data / "analysis/clustering/graphclust/clusters.csv")[
            "Cluster"
        ]
        feats = np.ascontiguousarray(pcs.iloc[:, 1:].to_numpy(np.float32))
        print(
            f"  {feats.shape[0]} cells, {feats.shape[1]} PCs, building {k}-NN",
            flush=True,
        )
        index = pynndescent.NNDescent(
            feats,
            n_neighbors=k + 1,
            metric="euclidean",
            random_state=0,
            low_memory=True,
        )
        nbr = index.neighbor_graph[0][:, 1:]
        src = np.repeat(np.arange(len(nbr), dtype=np.int32), nbr.shape[1])
        edges = np.unique(
            np.sort(np.column_stack([src, nbr.ravel().astype(np.int32)]), axis=1),
            axis=0,
        )
        labels = labels.to_numpy(np.int16)
        np.savez_compressed(cache, edges=edges, labels=labels)

    graph = nx.Graph()
    graph.add_nodes_from(range(len(labels)))
    graph.add_edges_from(map(tuple, edges))
    keep = sorted(max(nx.connected_components(graph), key=len))
    np.save(data / f"neurons_labels_k{k}.npy", labels[keep])

    return nx.convert_node_labels_to_integers(
        graph.subgraph(keep).copy(), ordering="sorted"
    )


def build(kind, n, data):
    if kind == "roadnet":
        return load_roadnet(data / "roadNet-PA.txt.gz")

    if kind == "neurons":
        return load_neurons(data)

    return make_graph("delaunay", n)


def graph_tag(kind, n):
    return kind if kind != "delaunay" else f"delaunay_n{n}"


def arm_tag(arm, alpha, diff_k, landmarks, npiv=None):
    # the diffusion horizon has to reach across the graph, so it is part of the cache key
    if arm == "sgd2" or (alpha, diff_k, landmarks) == (0.05, 20, 64):
        return arm if npiv is None else f"{arm}_P{npiv}"

    tag = f"{arm}_a{alpha:g}_K{diff_k}_L{landmarks}"

    return tag if npiv is None else f"{tag}_P{npiv}"


def fit_fling(graph, seed, alpha, diff_k, landmarks, cls=Fling, npiv=None):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    torch.set_default_device("cuda")
    extra = {} if npiv is None else dict(npiv=npiv)
    model = cls(
        n_components=2,
        seed=seed,
        alpha=alpha,
        diff_K=diff_k,
        n_landmarks=landmarks,
        **extra,
    )
    pos = np.asarray(model.fit_transform(graph), float)
    torch.set_default_device("cpu")

    return pos, dict(
        t=time.perf_counter() - t0,
        host_mb=peak_mb(),
        gpu_mb=torch.cuda.max_memory_allocated() / 1e6,
        alpha=alpha,
        diff_K=diff_k,
        n_landmarks=landmarks,
        npiv=npiv,
    )


def fit_sgd2(edges, n_nodes, seed):
    t0 = time.perf_counter()
    # the sparse pivot variant, which is what scales past a few tens of thousands of nodes
    xy = s_gd2.layout_sparse(
        edges[:, 0].astype(np.int32),
        edges[:, 1].astype(np.int32),
        min(200, n_nodes - 1),
        random_seed=seed,
    )

    return np.asarray(xy, float), dict(
        t=time.perf_counter() - t0, host_mb=peak_mb(), gpu_mb=0.0
    )


def run(kind, n, seed, out, n_pivots, data, alpha, diff_k, landmarks, arms, npiv=None):
    out.mkdir(parents=True, exist_ok=True)
    tag = graph_tag(kind, n)
    todo = [
        a
        for a in arms
        if not (
            out / f"{arm_tag(a, alpha, diff_k, landmarks, npiv)}_{tag}_seed{seed}.json"
        ).exists()
    ]
    if not todo:
        print("all arms cached")
        return

    print(f"building {kind}", flush=True)
    t0 = time.perf_counter()
    graph = build(kind, n, data)
    n_nodes = graph.number_of_nodes()
    print(
        f"  {n_nodes} nodes {graph.number_of_edges()} edges "
        f"in {time.perf_counter() - t0:.0f}s",
        flush=True,
    )

    edges = edge_array(graph)
    adj = sp.csr_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n_nodes, n_nodes)
    )
    adj = adj + adj.T
    pivots = np.random.default_rng(seed).choice(n_nodes, n_pivots, replace=False)

    for arm in todo:
        name = arm_tag(arm, alpha, diff_k, landmarks, npiv)
        print(f"fitting {name}", flush=True)
        if arm == "sgd2":
            pos, cell = fit_sgd2(edges, n_nodes, seed)
        else:
            cls = FlingStress if arm == "flingstress" else Fling
            pos, cell = fit_fling(graph, seed, alpha, diff_k, landmarks, cls, npiv)

        cell["stress"] = sampled_stress(pos, adj, pivots)
        cell["n"], cell["m"] = n_nodes, graph.number_of_edges()
        np.save(out / f"{name}_{tag}_seed{seed}.npy", pos.astype(np.float32))
        (out / f"{name}_{tag}_seed{seed}.json").write_text(json.dumps(cell, indent=1))
        print(f"  {cell['t']:.0f}s  stress {cell['stress']:.4f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--graph", default="delaunay", choices=["delaunay", "roadnet", "neurons"]
    )
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--data", type=pathlib.Path, default=ROOT / "data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pivots", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--diff-k", type=int, default=20)
    p.add_argument("--landmarks", type=int, default=64)
    p.add_argument("--npiv", type=int, default=None)  # FlingStress pivot columns
    p.add_argument("--arms", nargs="*", default=["fling", "sgd2"])  # or flingstress
    p.add_argument("--out", type=pathlib.Path, default=ROOT / "cache" / "million")

    a = p.parse_args()

    run(
        a.graph,
        a.n,
        a.seed,
        a.out,
        a.pivots,
        a.data,
        a.alpha,
        a.diff_k,
        a.landmarks,
        a.arms,
        a.npiv,
    )
