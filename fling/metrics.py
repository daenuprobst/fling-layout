import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from fling.baselines import adj_recall, all_pairs_D, full_sns
from fling.graphs import bfs_rows


def fast_D(graph):
    adj = nx.adjacency_matrix(graph)
    n_nodes = adj.shape[0]
    dist = np.empty((n_nodes, n_nodes), np.float32)

    for start in range(0, n_nodes, 256):
        dist[start : start + 256] = bfs_rows(
            adj, list(range(start, min(start + 256, n_nodes)))
        )

    return dist


def normalized_stress(pos, D):
    return full_sns(pos, D)


def _neighbor_sets(graph):
    nodes = list(graph.nodes())
    idx = {node: node_i for node_i, node in enumerate(nodes)}

    return [set(idx[nbr] for nbr in graph.neighbors(node)) for node in nodes]


def knn_recall(pos, nbrs, subset=None):
    return adj_recall(pos, nbrs, subset)


def neighborhood_preservation(pos, nbrs, subset=None):
    tree = cKDTree(pos)
    n_inter = n_union = 0.0
    node_ids = range(len(nbrs)) if subset is None else subset

    for node in node_ids:
        deg = len(nbrs[node])

        if deg == 0:
            continue

        knn = set(int(hit) for hit in tree.query(pos[node], deg + 1)[1][1:])
        n_inter += len(knn & nbrs[node])
        n_union += len(knn | nbrs[node])

    return n_inter / max(n_union, 1e-9)


def edge_length_cov(pos, edges):
    lens = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)

    return float(lens.std() / (lens.mean() + 1e-9))


def _ccw(p0, p1, p2):
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])


def count_crossings(pos, edges):
    n_edges = len(edges)
    n_cross = 0

    for i1 in range(n_edges):
        p1, p2 = pos[edges[i1, 0]], pos[edges[i1, 1]]

        for i2 in range(i1 + 1, n_edges):
            edge_j = edges[i2]

            if edges[i1, 0] in edge_j or edges[i1, 1] in edge_j:
                continue  # share a vertex, not a crossing

            p3, p4 = pos[edge_j[0]], pos[edge_j[1]]

            o1 = _ccw(p3, p4, p1)
            o2 = _ccw(p3, p4, p2)
            o3 = _ccw(p1, p2, p3)
            o4 = _ccw(p1, p2, p4)

            if ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0)):
                n_cross += 1

    return n_cross


def nonadjacent_pairs(edges):
    n_edges = len(edges)
    pairs = []

    for e1 in range(n_edges):
        for e2 in range(e1 + 1, n_edges):
            if edges[e1, 0] in edges[e2] or edges[e1, 1] in edges[e2]:
                continue

            pairs.append((e1, e2))

    return np.array(pairs, dtype=np.int64)


def crosslessness(pos, edges):
    n_cross = count_crossings(pos, np.asarray(edges))
    n_pairs = len(nonadjacent_pairs(np.asarray(edges)))

    if n_pairs == 0:
        return 1.0, 0

    return float(1.0 - np.sqrt(n_cross / n_pairs)), int(n_cross)


def min_crossing_angle(pos, edges):
    edges = np.asarray(edges)
    pairs = nonadjacent_pairs(edges)
    angles = []

    for e1, e2 in pairs:
        p1, p2 = pos[edges[e1, 0]], pos[edges[e1, 1]]
        p3, p4 = pos[edges[e2, 0]], pos[edges[e2, 1]]
        o1 = _ccw(p3, p4, p1)
        o2 = _ccw(p3, p4, p2)
        o3 = _ccw(p1, p2, p3)
        o4 = _ccw(p1, p2, p4)

        if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
            dir1 = p2 - p1
            dir2 = p4 - p3
            cos = abs(dir1 @ dir2) / (
                np.linalg.norm(dir1) * np.linalg.norm(dir2) + 1e-12
            )
            angles.append(np.degrees(np.arccos(np.clip(cos, 0, 1))))

    return float(np.mean(angles)) if angles else 90.0


def evaluate(pos, graph, D=None, nbrs=None, cross_max_e=1500, angle_max_e=800):
    pos = np.asarray(pos, float)

    if D is None:
        D = all_pairs_D(graph)

    if nbrs is None:
        nbrs = _neighbor_sets(graph)

    edges = np.array(graph.edges())
    n_edges = len(edges)
    out = {
        "stress": round(normalized_stress(pos, D), 4),
        "np_jaccard": round(neighborhood_preservation(pos, nbrs), 4),
        "knn_recall": round(knn_recall(pos, nbrs), 4),
        "edge_cov": round(edge_length_cov(pos, edges), 4),
    }

    if n_edges <= cross_max_e:
        cl_score, n_cross = crosslessness(pos, edges)
        out["crosslessness"] = round(cl_score, 4)
        out["crossings"] = int(n_cross)
    else:
        out["crosslessness"] = None
        out["crossings"] = None

    out["min_angle"] = (
        round(min_crossing_angle(pos, edges), 1) if n_edges <= angle_max_e else None
    )

    return out
