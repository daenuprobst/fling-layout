import networkx as nx
import numpy as np
from scipy.spatial import Delaunay


def make_graph(kind, n_nodes=40, extra=0, seed=0):
    rng = np.random.default_rng(seed)

    if kind == "tree":
        graph = nx.random_labeled_tree(n_nodes, seed=seed)
    elif kind == "grid":
        side = int(round(n_nodes**0.5))
        graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
    elif kind in ("delaunay", "nearly_planar"):
        pts = rng.random((n_nodes, 2))
        tri = Delaunay(pts)
        graph = nx.Graph()
        graph.add_nodes_from(range(n_nodes))

        for simplex in tri.simplices:
            for src, dst in (
                (simplex[0], simplex[1]),
                (simplex[1], simplex[2]),
                (simplex[2], simplex[0]),
            ):
                graph.add_edge(int(src), int(dst))
    else:
        raise ValueError(kind)

    if kind == "nearly_planar" or extra:
        nodes = list(graph.nodes())
        n_extra = extra or max(1, len(nodes) // 12)
        added = 0

        while added < n_extra:
            src, dst = rng.choice(nodes, 2, replace=False)

            if not graph.has_edge(src, dst):
                graph.add_edge(int(src), int(dst))
                added += 1

    graph = nx.convert_node_labels_to_integers(graph)

    return graph


def edge_array(graph):
    return np.array(graph.edges(), dtype=np.int64)


def bfs_rows(adj, sources):
    indptr, indices = adj.indptr, adj.indices
    n_nodes = adj.shape[0]
    out = np.zeros((len(sources), n_nodes))
    newmask = np.zeros(n_nodes, bool)

    for row, src in enumerate(sources):
        dist = out[row]
        visited = np.zeros(n_nodes, bool)
        visited[src] = True
        frontier = np.array([src], dtype=np.int64)
        level = 0

        while frontier.size:
            level += 1
            starts = indptr[frontier]
            counts = indptr[frontier + 1] - starts
            total = int(counts.sum())

            if total == 0:
                break

            offs = np.repeat(
                starts - np.concatenate(([0], np.cumsum(counts)[:-1])), counts
            )
            nbr = indices[offs + np.arange(total)]
            newmask[nbr] = True
            newmask &= ~visited
            frontier = np.flatnonzero(newmask)

            if frontier.size == 0:
                break

            visited[frontier] = True
            dist[frontier] = level
            newmask[frontier] = False  # newmask is all-False again for the next level

    return np.atleast_2d(out)


def _ccw(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])


def count_crossings(pos, edges):
    n_edges = len(edges)
    cnt = 0

    for i1 in range(n_edges):
        p1, p2 = pos[edges[i1, 0]], pos[edges[i1, 1]]

        for i2 in range(i1 + 1, n_edges):
            e2 = edges[i2]

            if edges[i1, 0] in e2 or edges[i1, 1] in e2:
                continue  # share a vertex, not a crossing

            p3, p4 = pos[e2[0]], pos[e2[1]]
            ccw1 = _ccw(p3, p4, p1)
            ccw2 = _ccw(p3, p4, p2)
            ccw3 = _ccw(p1, p2, p3)
            ccw4 = _ccw(p1, p2, p4)

            if ((ccw1 > 0) != (ccw2 > 0)) and ((ccw3 > 0) != (ccw4 > 0)):
                cnt += 1

    return cnt


def nonadjacent_pairs(edges):
    n_edges = len(edges)
    pairs = []

    for i1 in range(n_edges):
        for i2 in range(i1 + 1, n_edges):
            if edges[i1, 0] in edges[i2] or edges[i1, 1] in edges[i2]:
                continue

            pairs.append((i1, i2))

    return np.array(pairs, dtype=np.int64)
