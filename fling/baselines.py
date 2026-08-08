import time

import networkx as nx
import numpy as np
import s_gd2
import torch
from fa2_modified import ForceAtlas2
from scipy.spatial import cKDTree
from sklearn.manifold import TSNE

torch.set_num_threads(8)


def all_pairs_D(graph):
    idx = {node: i for i, node in enumerate(graph.nodes())}
    n_nodes = len(idx)
    dist = np.zeros((n_nodes, n_nodes), np.float32)

    for src, dist_map in nx.all_pairs_shortest_path_length(graph):
        for dst, length in dist_map.items():
            dist[idx[src], idx[dst]] = length

    return dist


def farthest_pivots(graph, n_pivots, among=None, seed=0):
    rng = np.random.default_rng(seed)
    nodes = list(graph.nodes())
    idx = {node: i for i, node in enumerate(nodes)}
    n_nodes = len(nodes)
    pool = [idx[node] for node in (among if among is not None else nodes)]
    pivots = [pool[int(rng.choice(len(pool)))]]
    cols = []

    for _ in range(n_pivots):
        dist_map = nx.single_source_shortest_path_length(graph, nodes[pivots[-1]])
        col = np.full(n_nodes, n_nodes, np.float32)

        for dst, length in dist_map.items():
            col[idx[dst]] = length

        cols.append(col)
        cand = np.min(np.stack(cols), 0).copy()
        cand[pivots] = -1

        if among is not None:
            mask = np.ones(n_nodes, bool)
            mask[pool] = False
            cand[mask] = -1

        far = int(np.argmax(cand))

        if far in pivots:
            break

        pivots.append(far)

    return np.array(pivots[: len(cols)]), np.stack(cols, 1)


def full_sns(pos, D):
    n_nodes = len(pos)
    iu = np.triu_indices(n_nodes, 1)
    dist = D[iu].astype(np.float64)
    edist = np.linalg.norm(pos[iu[0]] - pos[iu[1]], axis=1)
    mask = dist > 0
    dist, edist = dist[mask], edist[mask]
    scale = (edist / dist).sum() / ((edist / dist) ** 2).sum()

    return float(np.mean((scale * edist / dist - 1) ** 2))


def adj_recall(pos, nbrs, subset=None):
    tree = cKDTree(pos)
    hits = total = 0
    node_ids = range(len(nbrs)) if subset is None else subset

    for nid in node_ids:
        deg = len(nbrs[nid])

        if deg == 0:
            continue

        idx = tree.query(pos[nid], deg + 1)[1][1:]
        hits += len(set(int(cand) for cand in idx) & nbrs[nid])
        total += deg

    return hits / max(total, 1)


def label_purity(pos, labels, n_nbrs=10):
    tree = cKDTree(pos)
    n_nodes = len(pos)
    n_nbrs = min(n_nbrs, n_nodes - 1)
    same = 0
    idx = tree.query(pos, n_nbrs + 1)[1][:, 1:]

    for nid in range(n_nodes):
        same += np.mean(labels[idx[nid]] == labels[nid])

    return same / n_nodes


def fa2(graph, iters=400):
    layout = ForceAtlas2(verbose=False).forceatlas2_networkx_layout(
        graph, pos=None, iterations=iters
    )

    return np.array([layout[node] for node in graph.nodes()], float)


def sgd_stress(graph, seed=0):
    idx = {node: i for i, node in enumerate(graph.nodes())}
    edges = np.array([[idx[src], idx[dst]] for src, dst in graph.edges()])
    src = edges[:, 0].astype(np.int32)
    dst = edges[:, 1].astype(np.int32)

    try:
        return np.asarray(s_gd2.layout(src, dst, random_seed=seed), float)
    except TypeError:
        return np.asarray(s_gd2.layout(src, dst), float)


def tsnet(D, seed=0, perplexity=30):
    perp = max(5, min(perplexity, (len(D) - 1) // 3))

    return TSNE(
        metric="precomputed", init="random", perplexity=perp, random_state=seed
    ).fit_transform(D.astype(np.float64))


def tsnet_star(D, seed=0, perplexity=40):
    perp = max(5, min(perplexity, (len(D) - 1) // 3))
    init = _mds2(D)
    init = (init - init.mean(0)) / (init.std(0) + 1e-9) * 1e-4

    return TSNE(
        metric="precomputed",
        init=init.astype(np.float64),
        perplexity=perp,
        random_state=seed,
    ).fit_transform(D.astype(np.float64))


def _mds2(D):
    dist_sq = D.astype(float) ** 2
    gram = -0.5 * (
        dist_sq
        - dist_sq.mean(0, keepdims=True)
        - dist_sq.mean(1, keepdims=True)
        + dist_sq.mean()
    )
    uvec, sval, _ = np.linalg.svd(gram, full_matrices=False)

    return uvec[:, :2] * np.sqrt(
        np.maximum(sval[:2], 0.0)
    )  # classical scaling: U @ sqrt(Lambda)


def _pmds_embed(cols, dim):
    dist_sq = cols.astype(float) ** 2
    gram = -0.5 * (
        dist_sq
        - dist_sq.mean(0, keepdims=True)
        - dist_sq.mean(1, keepdims=True)
        + dist_sq.mean()
    )
    uvec, sval, _ = np.linalg.svd(gram, full_matrices=False)
    n_comp = min(dim, uvec.shape[1])
    emb = uvec[:, :n_comp] * sval[:n_comp]

    return (emb - emb.mean(0)) / (emb.std(0) + 1e-9)


class _MLP(torch.nn.Module):
    def __init__(self, dim_in, dim_out=2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim_in, 256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, dim_out),
        )

    def forward(self, feat):
        return self.net(feat)


def _train_mlp(feat, target, rows, epochs, batch, lr, seed):
    torch.manual_seed(seed)
    net = _MLP(feat.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rows = np.asarray(rows)
    inputs = feat[torch.as_tensor(rows, dtype=torch.long)]
    targets = torch.as_tensor(target, dtype=torch.float32)
    n_rows = len(rows)
    rng = np.random.default_rng(seed)

    for _ in range(epochs):
        perm = rng.permutation(n_rows)

        for start in range(0, n_rows, batch):
            idx = perm[start : start + batch]
            loss = ((net(inputs[idx]) - targets[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    return net


class NNPNet:
    def __init__(
        self,
        n_components=2,
        emb=50,
        pmds_pivots=250,
        perp=40,
        epochs=40,
        batch=64,
        lr=1e-3,
        seed=0,
    ):
        self.n_components = n_components
        self.emb = emb
        self.pmds_pivots = pmds_pivots
        self.perp = perp
        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        self.seed = seed

    def fit(self, graph, train_nodes=None):
        nodes = list(graph.nodes())
        idx = {node: i for i, node in enumerate(nodes)}

        train = np.array(
            [idx[node] for node in (nodes if train_nodes is None else train_nodes)]
        )

        _, cols = farthest_pivots(
            graph, min(self.pmds_pivots, len(train) - 1), among=train, seed=self.seed
        )

        feat = torch.tensor(
            _pmds_embed(cols, self.emb), dtype=torch.float32
        )  # PivotMDS input, all nodes

        dist_tr = all_pairs_D(graph)[
            np.ix_(train, train)
        ]  # tsNET* target on the train nodes

        target = tsnet_star(dist_tr, self.seed, self.perp)
        target = (target - target.mean(0)) / (target.std(0) + 1e-9)
        t_start = time.perf_counter()

        self.net_ = _train_mlp(
            feat, target, train, self.epochs, self.batch, self.lr, self.seed
        )
        self.fit_time_ = time.perf_counter() - t_start
        self.nodes_ = nodes

        self.edges_ = (
            np.array(
                [[idx[src], idx[dst]] for src, dst in graph.edges()], dtype=np.int64
            )
            if graph.number_of_edges()
            else np.empty((0, 2), np.int64)
        )

        with torch.no_grad():
            self.embedding_ = self.net_(feat).cpu().numpy().astype(np.float64)

        return self

    def fit_transform(self, graph, train_nodes=None):
        return self.fit(graph, train_nodes).embedding_


# Simonetto, Archambault, Auber and Bourqui, "ImPrEd: An Improved Force-Directed Algorithm that
# Prevents Nodes Crossing Edges", Computer Graphics Forum 30(3):1071-1080, 2011 (EuroVis),
# refining PrEd (Bertault, GD 2000). Structure follows the reference implementation accompanying
# Pupyrev's 2026 evaluation (github.com/spupyrev/planar-vibe, MIT).
def _seg_point_dist(px, py, ax, ay, bx, by):
    # distance from each point to each segment, and the closest point on it: (N,1)x(1,E)->(N,E)
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vv = vx * vx + vy * vy

    t = np.where(vv > 1e-12, (wx * vx + wy * vy) / np.maximum(vv, 1e-12), 0.0)
    t = np.clip(t, 0.0, 1.0)

    cx, cy = ax + t * vx, ay + t * vy

    return np.hypot(px - cx, py - cy), cx, cy


def _ray_seg_t(px, py, dx, dy, ax, ay, bx, by):
    # distance along ray (p, d) to segment (a, b). Returns inf if it does not hit. Broadcasts.
    ex, ey = bx - ax, by - ay
    den = dx * ey - dy * ex
    ok = np.abs(den) > 1e-12
    den = np.where(ok, den, 1.0)

    qx, qy = ax - px, ay - py

    t = (qx * ey - qy * ex) / den  # along the ray
    u = (qx * dy - qy * dx) / den  # along the segment

    hit = ok & (t > 1e-9) & (u >= 0.0) & (u <= 1.0)

    return np.where(hit, t, np.inf)


def _crossed(pos0, pos1, edges, tol=0.01):
    # (node, edge) pairs where the node moved through the edge: side flips, foot stays inside
    def sides(p):
        a, b = p[edges[:, 0]], p[edges[:, 1]]
        v = (b - a)[None, :, :]
        w = p[:, None, :] - a[None, :, :]
        cr = v[..., 0] * w[..., 1] - v[..., 1] * w[..., 0]
        vv = (v**2).sum(-1)
        t = np.where(vv > 1e-12, (w * v).sum(-1) / np.maximum(vv, 1e-12), -1.0)

        return np.sign(cr), (t > 0.0) & (t < 1.0)

    s0, in0 = sides(pos0)
    s1, in1 = sides(pos1)

    inc = np.zeros((len(pos0), len(edges)), bool)
    inc[edges[:, 0], np.arange(len(edges))] = True
    inc[edges[:, 1], np.arange(len(edges))] = True

    return (s0 != s1) & in0 & in1 & ~inc & (s0 != 0)


def impred(
    pos,
    edges,
    iters=100,
    n_sectors=8,
    gamma=None,
    seed=0,
    w_edge=1.0,
    w_node=1.0,
    w_nodeedge=1.0,
    cool=None,
    chunk=256,
    damp=0.5,
    verbose=False,
):
    # `pos` is the drawing to refine (n, 2), `edges` an (E, 2) index array. `gamma` is the
    # node-edge interaction radius, defaulting to the median edge length.
    pos = np.array(pos, float, copy=True)
    edges = np.asarray(edges, int)
    n, E = len(pos), len(edges)

    seg = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)
    scale = float(np.median(seg[seg > 0])) if np.any(seg > 0) else 1.0
    gamma = scale if gamma is None else gamma
    step_max = 0.5 * scale

    inc = np.zeros((n, E), bool)
    inc[edges[:, 0], np.arange(E)] = True
    inc[edges[:, 1], np.arange(E)] = True

    sec_ang = (np.arange(n_sectors) + 0.5) * (2 * np.pi / n_sectors) - np.pi
    sec_dx, sec_dy = np.cos(sec_ang), np.sin(sec_ang)
    sector_w = 2.0 * np.pi / n_sectors

    # cooling is tied to the iteration count, not a fixed per-step factor
    cool = (0.1 ** (1.0 / max(iters, 1))) if cool is None else cool
    temp = 1.0

    for _it in range(iters):
        prev = pos.copy()
        disp = np.zeros_like(pos)

        # (1) edge attraction toward the rest length
        d = pos[edges[:, 1]] - pos[edges[:, 0]]
        L = np.linalg.norm(d, axis=1, keepdims=True)
        u = d / np.maximum(L, 1e-12)
        f = w_edge * (L - scale) / scale * u

        np.add.at(disp, edges[:, 0], f)
        np.add.at(disp, edges[:, 1], -f)

        # (2) node-node repulsion, only within gamma
        for i, j in cKDTree(pos).query_pairs(gamma, output_type="ndarray"):
            dv = pos[i] - pos[j]
            dist = max(float(np.hypot(*dv)), 1e-9)
            g = w_node * (gamma - dist) / dist * dv

            disp[i] += g
            disp[j] -= g

        # (3) node-edge repulsion, and (4) the per-sector movement limits
        limit = np.full((n, n_sectors), step_max)

        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            px, py = pos[lo:hi, 0][:, None], pos[lo:hi, 1][:, None]
            ax, ay = pos[edges[:, 0], 0][None, :], pos[edges[:, 0], 1][None, :]
            bx, by = pos[edges[:, 1], 0][None, :], pos[edges[:, 1], 1][None, :]
            dist, cx, cy = _seg_point_dist(px, py, ax, ay, bx, by)
            dist = np.where(inc[lo:hi], np.inf, dist)

            near = (dist < gamma) & np.isfinite(dist)
            if near.any():
                ni, ei = np.nonzero(near)
                dv = np.stack([px[ni, 0] - cx[ni, ei], py[ni, 0] - cy[ni, ei]], 1)
                dd = np.maximum(np.linalg.norm(dv, axis=1, keepdims=True), 1e-9)
                g = w_nodeedge * ((gamma - dd) ** 2) / (gamma * dd) * dv
                np.add.at(disp, ni + lo, g)
                np.add.at(disp, edges[ei, 0], -0.5 * g)
                np.add.at(disp, edges[ei, 1], -0.5 * g)

            reach = (dist < 3.0 * step_max) & np.isfinite(
                dist
            )  # farther edges are unreachable

            if reach.any():
                ni, ei = np.nonzero(reach)

                for s in range(n_sectors):
                    t = _ray_seg_t(
                        pos[ni + lo, 0],
                        pos[ni + lo, 1],
                        sec_dx[s],
                        sec_dy[s],
                        pos[edges[ei, 0], 0],
                        pos[edges[ei, 0], 1],
                        pos[edges[ei, 1], 0],
                        pos[edges[ei, 1], 1],
                    )

                    np.minimum.at(limit, (ni + lo, np.full(len(ni), s)), t)

        limit = np.maximum(limit - 1e-4, 0.0)

        # (5) move, clamped to the limit of the sector the displacement points into
        mag = np.linalg.norm(disp, axis=1)

        if not (mag > 1e-12).any():
            break

        dsec = np.floor((np.arctan2(disp[:, 1], disp[:, 0]) + np.pi) / sector_w).astype(
            int
        )

        cap = np.minimum(limit[np.arange(n), dsec % n_sectors], step_max) * temp

        pos += (
            disp
            * np.where(mag > 1e-12, np.minimum(1.0, cap / np.maximum(mag, 1e-12)), 0.0)[
                :, None
            ]
        )

        # (6) the guarantee: revert any node that went through an edge, and damp it
        bad = _crossed(prev, pos, edges)
        n_bad, full = int(bad.any() and len(np.unique(np.nonzero(bad)[0]))), False

        if bad.any():
            for _ in range(6):
                vi, ei = np.nonzero(bad)
                revert = np.unique(np.concatenate([vi, edges[ei, 0], edges[ei, 1]]))
                pos[revert] = prev[revert]
                bad = _crossed(prev, pos, edges)

                if not bad.any():
                    break

            if bad.any():
                pos = prev
                full = True
                temp *= damp

        if verbose:
            print(
                f"    it{_it:>4} step={np.linalg.norm(pos - prev, axis=1).mean():.5f} "
                f"cap_med={np.median(cap):.4f} force_med={np.median(mag):.4f} "
                f"offenders={n_bad} full_rollback={full} temp={temp:.4f}",
                flush=True,
            )

        temp *= cool

    return pos
