import os
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.io

from fling.graphs import make_graph

# the benchmark graphs are not shipped in the wheel, so an installed copy needs FLING_DATA
DATA = Path(
    os.environ.get("FLING_DATA") or Path(__file__).resolve().parents[1] / "data"
)


def _lcc(graph):
    graph = nx.Graph(graph)
    graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()

    return nx.convert_node_labels_to_integers(graph)


def _sbm(sizes=(80, 80, 80), pin=0.12, pout=0.008, seed=1):
    n_blocks = len(sizes)
    prob = [
        [pin if row == col else pout for col in range(n_blocks)]
        for row in range(n_blocks)
    ]

    return _lcc(nx.stochastic_block_model(list(sizes), prob, seed=seed))


def _load_mtx(name):
    return _lcc(
        nx.from_scipy_sparse_array(
            scipy.io.mmread(str(DATA / name / f"{name}.mtx")).tocsr()
        )
    )


def scaling_graphs(sizes):
    return [
        (f"delaunay_{int(size)}", make_graph("delaunay", int(size))) for size in sizes
    ]


def graphs(quick=False):
    specs = [
        ("grid_400", make_graph("grid", 400)),
        ("tree_400", make_graph("tree", 400)),
        ("delaunay_600", make_graph("delaunay", 600)),
        ("sbm_240", _sbm((80, 80, 80))),
        ("lesmis", nx.les_miserables_graph()),
    ]

    if not quick:
        specs += [
            ("dwt_1005", _load_mtx("dwt_1005")),
            ("3elt", _load_mtx("3elt")),
            ("jagmesh8", _load_mtx("jagmesh8")),
            (
                "ego-Facebook",
                nx.read_edgelist(DATA / "facebook_combined.txt", nodetype=int),
            ),
        ]

    out = []

    for name, graph in specs:
        graph = nx.convert_node_labels_to_integers(
            graph.subgraph(max(nx.connected_components(graph), key=len))
        )
        out.append((name, graph))

    return out


def _lcc_with(graph, node_attr):
    keep = max(nx.connected_components(graph), key=len)

    return nx.convert_node_labels_to_integers(
        graph.subgraph(keep).copy(), label_attribute=node_attr
    )


def load_cora(path=None):
    path = Path(path) if path is not None else DATA / "cora"
    label_of = {}

    with open(path / "cora.content") as fh:
        for line in fh:
            parts = line.split()
            if parts:
                label_of[parts[0]] = parts[-1]

    graph = nx.Graph()
    graph.add_nodes_from(label_of)

    with open(path / "cora.cites") as fh:
        for line in fh:
            pair = line.split()
            if len(pair) >= 2 and pair[0] in label_of and pair[1] in label_of:
                graph.add_edge(pair[0], pair[1])

    graph = graph.subgraph([node for node in graph if node in label_of]).copy()
    lcc = _lcc_with(graph, "pid")
    classes = sorted(set(label_of.values()))
    class_id = {cls: idx for idx, cls in enumerate(classes)}

    labels = np.array(
        [
            class_id[label_of[lcc.nodes[idx]["pid"]]]
            for idx in range(lcc.number_of_nodes())
        ]
    )

    return lcc, labels, classes
