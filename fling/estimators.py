import collections
import time

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree
from scipy.special import erfinv

from .graphs import bfs_rows
from .nets import Gabor, _has_twins, _with_tie, tie_features


def _adam(params, lr, **kw):
    # the fused kernel needs every parameter on one CUDA device and a floating dtype. On CPU
    # foreach is the quick one. Neither changes a result beyond rounding.
    params = list(params)
    on_cuda = bool(params) and all(getattr(p, "is_cuda", False) for p in params)

    try:
        return (
            torch.optim.Adam(params, lr=lr, fused=True, **kw)
            if on_cuda
            else torch.optim.Adam(params, lr=lr, foreach=True, **kw)
        )
    except (RuntimeError, ValueError):  # dtype or device the fast kernels refuse
        return torch.optim.Adam(params, lr=lr, **kw)


def _norm_adj(adj):
    adj = sp.csr_matrix(adj).astype(float)
    adj.data[:] = 1.0
    deg = np.asarray(adj.sum(1)).ravel()
    deg[deg == 0] = 1.0
    dmat = sp.diags(1.0 / np.sqrt(deg))

    return (dmat @ adj @ dmat).tocsr()


def _diffusion_landmarks(adj, land_idx, alpha=0.15, n_steps=10, chunk=32):
    ahat = _norm_adj(adj).astype(np.float32)
    n_nodes = ahat.shape[0]
    lmarks = np.asarray(land_idx)
    n_land = len(lmarks)
    out = np.empty((n_nodes, n_land), dtype=np.float32)

    for start in range(0, n_land, chunk):
        lchunk = lmarks[start : start + chunk]
        onehot = np.zeros((n_nodes, len(lchunk)), dtype=np.float32)
        onehot[lchunk, np.arange(len(lchunk))] = 1.0
        ppr = alpha * onehot
        prop = onehot

        for step in range(1, n_steps + 1):
            prop = ahat @ prop
            ppr = ppr + np.float32(alpha * (1 - alpha) ** step) * prop

        out[:, start : start + len(lchunk)] = ppr

    np.log(out + np.float32(1e-7), out=out)

    return np.negative(out, out=out)


def _tangent_frames(feat, edges, n_nodes):
    adj = collections.defaultdict(list)

    for src, dst in edges:
        adj[int(src)].append(int(dst))
        adj[int(dst)].append(int(src))

    frames = np.zeros((n_nodes, feat.shape[1], 2), np.float32)

    for node in range(n_nodes):
        nbrs = adj.get(node, [])

        if len(nbrs) < 2:
            continue

        _, _, vt = np.linalg.svd(feat[nbrs] - feat[node], full_matrices=False)
        frames[node] = vt[:2].T

    seen = np.zeros(n_nodes, bool)

    for root in range(n_nodes):  # per connected component
        if seen[root] or not adj.get(root):
            continue

        seen[root] = True
        queue = collections.deque([root])

        while queue:
            node = queue.popleft()

            for nbr in adj[node]:
                if seen[nbr]:
                    continue

                seen[nbr] = True

                if (
                    np.linalg.det(frames[node].T @ frames[nbr]) < 0
                ):  # frames disagree -> flip nbr to match node
                    frames[nbr][:, 1] *= -1.0

                queue.append(nbr)

    return frames


def _fold_penalty(net, feat, frames, rows, margin, sign="auto"):
    # frames are only made consistent by a BFS, never aligned to the drawing, so which sign of the
    # determinant means "not folded" is not known in advance. sign="auto" takes the batch's own
    # median sign as the reference so the barrier penalises disagreement, not a convention
    feat_b = feat[rows]
    _, jac1 = torch.func.jvp(lambda inp: net(inp), (feat_b,), (frames[rows, :, 0],))
    _, jac2 = torch.func.jvp(lambda inp: net(inp), (feat_b,), (frames[rows, :, 1],))

    det = jac1[:, 0] * jac2[:, 1] - jac1[:, 1] * jac2[:, 0]
    scale = det.abs().mean().detach() + 1e-12

    if sign == "auto":
        ref = torch.sign(det.detach().median())
        ref = ref if ref != 0 else torch.ones((), dtype=det.dtype, device=det.device)
    else:
        ref = torch.as_tensor(float(sign), dtype=det.dtype, device=det.device)

    return torch.relu(margin - ref * det / scale).mean()


def _attraction(pos, edges, kdist, law="fr"):
    src, dst = edges[:, 0], edges[:, 1]
    diff = pos[src] - pos[dst]
    dist = np.linalg.norm(diff, axis=1) + 1e-9

    if law == "linlog":
        force = diff / dist[:, None]  # |force| = 1
    elif law == "fa2":
        force = diff  # |force| = dist
    else:
        force = (dist / kdist)[
            :, None
        ] * diff  # |force| = dist^2 / k   (Fruchterman-Reingold)

    n_nodes = pos.shape[0]
    att = np.empty_like(pos)

    for dim in range(pos.shape[1]):
        att[:, dim] = np.bincount(dst, force[:, dim], minlength=n_nodes) - np.bincount(
            src, force[:, dim], minlength=n_nodes
        )

    return att


def _repulsion_at(
    pos, anc, kdist, law="fr", deg=None, cutoff=0.0, near=0.0, metw=None, emask=None
):
    ancpos = pos[anc]
    dist2 = (
        (ancpos**2).sum(1)[:, None] + (pos**2).sum(1)[None, :] - 2.0 * ancpos @ pos.T
    )

    dist2 = np.maximum(dist2, 1e-18)
    weight = 1.0 / dist2  # force ~ 1/d along the unit vector

    if cutoff > 0:
        weight[dist2 > (cutoff * kdist) ** 2] = 0.0

    if near > 0:  # near pairs handled exactly (P3M split)
        weight[dist2 < (near * kdist) ** 2] = 0.0

    if metw is not None:  # metric-aware: repel in proportion to the
        weight = weight * metw  # graph distance the pair should sit at

    if emask is not None:  # pairs handled exactly at the call site
        rows, cols = emask.nonzero()
        weight[rows, cols] = 0.0

    if law == "fa2" and deg is not None:  # hubs push harder
        weight = weight * ((deg[anc] + 1.0)[:, None] * (deg + 1.0)[None, :])

    weight[np.arange(len(anc)), anc] = 0.0
    scale = kdist**2 if law != "linlog" else 1.0

    return scale * (ancpos * weight.sum(1)[:, None] - weight @ pos)


def _near_repulsion(pos, kdist, radius, law="fr", deg=None, rng=None, cap=50):
    pairs = cKDTree(pos).query_pairs(radius, output_type="ndarray")
    out = np.zeros_like(pos)

    if len(pairs) == 0:
        return out

    if (
        len(pairs) > cap * len(pos) and rng is not None
    ):  # collapsed clump: O(clump^2) pairs, subsample
        pairs = pairs[rng.choice(len(pairs), cap * len(pos), replace=False)]

    src, dst = pairs[:, 0], pairs[:, 1]
    diff = pos[src] - pos[dst]
    dist2 = np.maximum((diff**2).sum(1), 1e-18)
    weight = 1.0 / dist2

    if law == "fa2" and deg is not None:
        weight = weight * (deg[src] + 1.0) * (deg[dst] + 1.0)

    force = (kdist**2 if law != "linlog" else 1.0) * weight[:, None] * diff

    for dim in range(pos.shape[1]):
        out[:, dim] = np.bincount(src, force[:, dim], minlength=len(pos)) - np.bincount(
            dst, force[:, dim], minlength=len(pos)
        )

    return out


def _fr_energy(pos, edges, kdist):
    edist = (pos[edges[:, 0]] - pos[edges[:, 1]]).norm(dim=-1)
    E_attr = (edist**3 / (3 * kdist)).sum()

    # all-pairs log-repulsion via pdist (one fused kernel) in float32: the condensed distance
    # vector is the whole O(N^2) cost
    dists = torch.pdist(pos.float())
    E_rep = kdist**2 * torch.log(dists.clamp_min(1e-6)).sum()
    return (E_attr - E_rep.to(pos.dtype)) / pos.shape[0]


class _BaseFling:
    def __init__(
        self,
        n_components=3,
        w0=None,
        act="gelu",
        width=128,
        depth=2,
        tb_dim=None,
        tb_alpha=0.03,
        seed=0,
        local_cache=False,
    ):
        self.n_components = n_components
        self.w0 = w0
        self.act = act
        self.width = width
        self.depth = depth
        self.tb_dim = tb_dim
        self.tb_alpha = tb_alpha
        self.seed = seed

        # local_cache keeps the fit's raw feature block so transform(local=True) can place
        # newcomers from their neighbours' columns. It must be set BEFORE fit
        self.local_cache = local_cache

    # feature construction (identical at fit and transform, so OOS matches training)
    def _resolve_tb(self, graph):
        return (2 if _has_twins(graph) else 0) if self.tb_dim is None else self.tb_dim

    def _hop_csr(self, graph):
        allnodes = list(graph.nodes())

        return nx.adjacency_matrix(graph, nodelist=allnodes), {
            node: row for row, node in enumerate(allnodes)
        }

    def _pivot_columns(self, graph, pivots, nodes):
        adj, rowof = self._hop_csr(graph)
        dist = bfs_rows(adj, [rowof[piv] for piv in pivots])

        return np.ascontiguousarray(
            dist[:, np.array([rowof[node] for node in nodes])].T
        )

    def _farthest_pivots(self, graph, nodes, npiv):
        rng = np.random.default_rng(self.seed)
        adj, rowof = self._hop_csr(graph)
        rows = np.array([rowof[node] for node in nodes])
        pivots = [nodes[int(rng.integers(len(nodes)))]]
        cols = []
        dmin = None  # running min over columns == min of the stack

        for _ in range(npiv):
            col = bfs_rows(adj, [rowof[pivots[-1]]])[0][rows]
            cols.append(col)
            dmin = col if dmin is None else np.minimum(dmin, col)
            far = nodes[int(np.argmax(dmin))]

            if far in pivots:
                break

            pivots.append(far)

        dist = np.stack(cols, 1)

        return (
            pivots[: dist.shape[1]],
            dist,
        )  # one label per column (drop the last unused)

    def _maybe_cache_block(self, block, nodes):
        # subclasses that override _input_block must still call this, or local placement breaks
        if (
            getattr(self, "local_cache", False)
            and getattr(self, "_feat_cache_", None) is None
            and list(nodes) == list(getattr(self, "nodes_", []))
        ):
            self._feat_cache_ = np.array(block, dtype=np.float32)

    def _input_block(self, graph, nodes):
        out = (
            self._diffusion_block(graph, nodes)
            if getattr(self, "features", "pivot") == "diffusion"
            else self._pivot_columns(graph, self._feat_pivots_, nodes)
        )

        rp = getattr(self, "_rp_dim_", getattr(self, "rp_dim", 0))

        if rp:
            out = np.concatenate([out, self._rp_block(graph, nodes, rp)], 1)

        self._maybe_cache_block(out, nodes)

        return out

    def _diff_steps(self, graph):
        # walk length, fixed at fit time so transform matches. Derived from an eccentricity so the
        # walk reaches the far side of the graph rather than leaving those columns unordered.
        if self.diff_K is not None:
            return self.diff_K

        if getattr(self, "_diff_steps_", None) is None:
            src = next(iter(graph.nodes()))
            ecc = max(nx.single_source_shortest_path_length(graph, src).values())
            self._diff_steps_ = int(min(2 * ecc + 4, 200))

        return self._diff_steps_

    def _diffusion_block(self, graph, nodes):
        steps = self._diff_steps(graph)

        if getattr(self, "_diff_sat_", None) is None:  # first call = fit
            idx = {node: row for row, node in enumerate(nodes)}
            adj = nx.adjacency_matrix(graph, nodelist=nodes).astype(float)

            pot = _diffusion_landmarks(
                adj, [idx[node] for node in self._land_], self.alpha, steps
            )

            # backstop against a degenerate constant block: fall back to hop columns
            self._diff_sat_ = bool((np.abs(pot + np.log(1e-7)) < 1e-3).mean() > 0.995)

            if not self._diff_sat_:
                return pot

        if self._diff_sat_:
            return self._pivot_columns(graph, self._land_, nodes)

        idx = {node: row for row, node in enumerate(nodes)}
        adj = nx.adjacency_matrix(graph, nodelist=nodes).astype(float)

        return _diffusion_landmarks(
            adj, [idx[node] for node in self._land_], self.alpha, steps
        )

    def _rp_block(self, graph, nodes, dim, depths=(1, 2, 3, 5, 8)):
        # Gaussian probes smoothed by the normalised adjacency, read at several depths: these break
        # the ties between nodes equidistant from every landmark
        adj = nx.adjacency_matrix(graph, nodelist=nodes).astype(float)
        ahat = _norm_adj(adj).astype(np.float32)
        per = max(1, dim // len(depths))

        # splitmix64 on (label, column, seed), so each node's probe is a pure function of its own
        # LABEL and adding a node leaves every other node's features untouched. hash() is signed,
        # and the mask is the usual reinterpretation into uint64.
        key = np.asarray(
            [hash(node) & 0xFFFFFFFFFFFFFFFF for node in nodes], dtype=np.uint64
        )
        cols = []
        for column in range(per):
            z = key + np.uint64(0x9E3779B97F4A7C15) * np.uint64(
                column + 1 + 977 * self.seed
            )
            z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            z = z ^ (z >> np.uint64(31))

            uni = (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)
            cols.append(
                np.sqrt(2.0) * erfinv(np.clip(2 * uni - 1, -1 + 1e-12, 1 - 1e-12))
            )

        cur = np.stack(cols, 1).astype(np.float32) / np.sqrt(per)
        blocks = []

        for step in range(1, max(depths) + 1):
            cur = ahat @ cur

            if step in depths:
                blocks.append(cur.copy())

        return np.concatenate(blocks, 1)

    def _input_block_local(self, graph, new_nodes):
        if getattr(self, "_feat_cache_", None) is None:
            raise RuntimeError(
                "local placement needs the fit's feature block: set local_cache=True "
                "(constructor or attribute) BEFORE fit, then re-fit"
            )

        cache = self._feat_cache_
        rowof = {node: row for row, node in enumerate(self.nodes_)}
        new = list(new_nodes)
        newrow = {node: row for row, node in enumerate(new)}
        n_cols = cache.shape[1]

        hop = (
            bool(getattr(self, "_diff_sat_", False))
            or getattr(self, "features", "pivot") != "diffusion"
        )

        if hop:
            BIG = np.float64(1e6)
            out = np.full((len(new), n_cols), BIG)

            for _ in range(len(new) + 1):  # relax until new-new edges settle
                changed = False

                for node in new:
                    best = out[newrow[node]]

                    for nbr in graph[node]:
                        src = (
                            cache[rowof[nbr]]
                            if nbr in rowof
                            else (out[newrow[nbr]] if nbr in newrow else None)
                        )

                        if src is None:
                            continue

                        cand = src + 1.0

                        if (cand < best).any():
                            best = np.minimum(best, cand)
                            changed = True

                    out[newrow[node]] = best

                if not changed:
                    break

            return np.where(
                out >= BIG, 0.0, out
            )  # unreachable -> 0.0, as bfs_rows returns
        ppr = (
            np.exp(-cache.astype(np.float64)) - 1e-7
        )  # cached columns are potentials -log(p)

        deg = dict(graph.degree())
        acc = np.zeros((len(new), n_cols))

        for _ in range(2):  # one pass + one relaxation for new-new
            for node in new:
                psum = np.zeros(n_cols)

                for nbr in graph[node]:
                    pnbr = (
                        ppr[rowof[nbr]]
                        if nbr in rowof
                        else (acc[newrow[nbr]] if nbr in newrow else None)
                    )

                    if pnbr is None:
                        continue

                    psum += pnbr / np.sqrt(max(deg[node], 1) * max(deg[nbr], 1))

                acc[newrow[node]] = (1.0 - self.alpha) * psum

        return -np.log(acc + 1e-7)

    def _features_local(self, graph, new_nodes):
        feat = (self._input_block_local(graph, new_nodes) - self._fmean) / self._fstd

        if self._tb_dim_ > 0:
            feat = np.concatenate(
                [
                    feat,
                    tie_features(
                        len(new_nodes), self._tb_dim_, self.tb_alpha, self.seed
                    ),
                ],
                1,
            )

        if getattr(self, "_code_pad_", 0):
            feat = np.concatenate(
                [feat, np.zeros((len(new_nodes), self._code_pad_))], 1
            )

        return feat

    def _features(self, graph, nodes):
        feat = (self._input_block(graph, nodes) - self._fmean) / self._fstd

        if self._tb_dim_ > 0:  # inductive RNI: each node draws its own cols
            feat = np.concatenate(
                [
                    feat,
                    tie_features(len(nodes), self._tb_dim_, self.tb_alpha, self.seed),
                ],
                1,
            )

        if getattr(
            self, "_code_pad_", 0
        ):  # codes are transductive: unseen nodes get zeros
            feat = np.concatenate([feat, np.zeros((len(nodes), self._code_pad_))], 1)

        return feat

    # sklearn surface
    def fit_transform(self, graph):
        self.fit(graph)
        return self.embedding_

    def transform(self, graph, nodes=None, relax=0, local=False):
        if not hasattr(self, "net_"):
            raise RuntimeError("call fit before transform")

        nodes = list(graph.nodes()) if nodes is None else list(nodes)

        if local:
            known = {node: row for row, node in enumerate(self.nodes_)}
            new = [node for node in nodes if node not in known]
            coords = np.empty((len(nodes), self.embedding_.shape[1]))

            for row, node in enumerate(nodes):  # fitted nodes: frozen, verbatim
                if node in known:
                    coords[row] = self.embedding_[known[node]]

            if new:
                nfeat = self._features_local(graph, new)
                dt = next(self.net_.parameters()).dtype

                with torch.no_grad():
                    ncoords = (
                        self.net_(torch.tensor(nfeat, dtype=dt))
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )

                if relax:  # de-overlap newcomers only
                    ncoords = self._relax_new(graph, new, ncoords, iters=int(relax))

                pos = {node: nrow for nrow, node in enumerate(new)}

                for row, node in enumerate(nodes):
                    if node not in known:
                        coords[row] = ncoords[pos[node]]

            return coords

        feat = self._features(graph, nodes)
        dt = next(self.net_.parameters()).dtype

        with torch.no_grad():
            coords = (
                self.net_(torch.tensor(feat, dtype=dt)).cpu().numpy().astype(np.float64)
            )

        return (
            self._relax_new(graph, nodes, coords, iters=int(relax)) if relax else coords
        )

    def _relax_new(self, graph, new_nodes, coords, iters=120, rep=1.0, attr=1.0):
        pos = {node: self.embedding_[row] for row, node in enumerate(self.nodes_)}

        for node, co in zip(new_nodes, coords):
            pos[node] = np.asarray(co, float)

        order = [node for node in graph.nodes() if node in pos]
        pts = np.array([pos[node] for node in order])
        rowof = {node: row for row, node in enumerate(order)}
        mov = np.array([rowof[node] for node in new_nodes if node in rowof])

        epairs = [
            (rowof[src], rowof[dst])
            for src, dst in graph.edges()
            if src in rowof and dst in rowof
        ]

        scale = np.abs(self.embedding_ - self.embedding_.mean(0)).max() + 1e-9
        kdist = 0.06 * scale
        temp = 0.12 * scale
        soft = (0.02 * scale) ** 2

        pts[mov] += np.random.default_rng(0).normal(
            scale=0.01 * scale, size=(len(mov), pts.shape[1])
        )  # break coincidence

        movof = {gr: mr for mr, gr in enumerate(mov)}

        for _ in range(iters):
            diff = pts[mov][:, None, :] - pts[None, :, :]
            dist2 = np.maximum((diff**2).sum(2), soft)
            dist2[np.arange(len(mov)), mov] = np.inf

            frep = (
                rep * kdist**2 * (diff / dist2[:, :, None]).sum(1)
            )  # repel from all nodes

            fatt = np.zeros((len(mov), pts.shape[1]))

            for src, dst in epairs:  # attract along incident edges
                for end, ref in ((src, dst), (dst, src)):
                    if end in movof:
                        dvec = pts[ref] - pts[end]
                        dlen = np.linalg.norm(dvec) + 1e-9
                        fatt[movof[end]] += attr * (dlen / kdist) * dvec

            vel = frep + fatt
            mag = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-12
            pts[mov] += np.minimum(mag, temp) * vel / mag
            temp = max(temp * 0.97, 1e-3 * scale)

        return np.array([pts[rowof[node]] for node in new_nodes if node in rowof])

    # train / held-out plumbing shared by all variants
    def _train_split(self, graph, train_nodes):
        nodes = list(graph.nodes())

        idx = self._record_graph(graph, nodes)

        self._diff_sat_ = None  # re-decide the diffusion saturation fallback per fit
        self._feat_cache_ = None  # raw fit feature block, refilled by _features

        # the random probe columns are dropped on a sampled fit: with only M rows in the energy
        # the network fits them instead of averaging them out
        self._rp_dim_ = 0 if train_nodes is not None else getattr(self, "rp_dim", 0)

        train = nodes if train_nodes is None else list(train_nodes)
        tix = np.array([idx[node] for node in train], dtype=np.int64)
        loc = {node: row for row, node in enumerate(train)}

        return nodes, len(nodes), idx, train, tix, loc

    def _train_edges(self, graph, loc):
        elist = [
            [loc[src], loc[dst]]
            for src, dst in graph.edges()
            if src in loc and dst in loc
        ]

        return np.array(elist, dtype=np.int64) if elist else np.empty((0, 2), np.int64)

    def _record_graph(self, graph, nodes):
        idx = {node: row for row, node in enumerate(nodes)}
        self.nodes_ = nodes

        self.edges_ = (
            np.array(
                [[idx[src], idx[dst]] for src, dst in graph.edges()], dtype=np.int64
            )
            if graph.number_of_edges()
            else np.empty((0, 2), np.int64)
        )

        return idx


class FlingStress(_BaseFling):
    def __init__(
        self,
        n_components=3,
        iters=500,
        npiv=None,
        batch=128,
        lr=5e-3,
        fcols=16,
        knn_w=0.0,
        knn_K=4,
        w0=2.0,
        act="gelu",
        width=128,
        depth=2,
        tb_dim=None,
        tb_alpha=0.03,
        seed=0,
        features="diffusion",
        n_landmarks=64,
        alpha=0.05,
        diff_K=None,
        rp_dim=32,
        rep_w=0.0,
        rep_K=10,
        solver="sgd",
        code_dim=0,
        code_init=0.01,
        local_w=0.0,
        local_ramp=0.0,
        hard_neg=0.0,
        finish=0,
        record_every=0,
        anneal=0.05,
        local_cache=False,
        fold_w=0.0,
        fold_margin=0.1,
        fold_sign="auto",
        fold_cols=64,
        fold_start=0.0,
        fold_rand=False,
    ):
        super().__init__(
            n_components, w0, act, width, depth, tb_dim, tb_alpha, seed, local_cache
        )

        self.iters = iters
        self.npiv = npiv
        self.batch = batch
        self.lr = lr
        self.fcols = fcols
        self.knn_w = knn_w
        self.knn_K = knn_K

        # the tangent-Jacobian barrier, which is off by default, since it changes the layouts.
        # fold_cols caps which feature columns may define a tangent frame (the structural ones).
        # fold_start delays it as a fraction of the schedule
        self.fold_w = fold_w
        self.fold_margin = fold_margin
        self.fold_sign = fold_sign
        self.fold_cols = fold_cols
        self.fold_start = fold_start
        self.fold_rand = fold_rand

        # rep_w>0 = maxent-stress is the entropy repulsion over random non-neighbours spreads the drawing
        self.rep_w = rep_w
        self.rep_K = rep_K
        self.solver = solver

        if solver != "sgd" and rep_w > 0:
            raise ValueError("maxent (rep_w>0) requires solver='sgd'")

        self.features = features  # "pivot" hop distances | "diffusion" potentials
        self.n_landmarks = n_landmarks
        self.alpha = alpha
        self.diff_K = diff_K
        self.rp_dim = rp_dim

        self.record_every = (
            record_every  # if >0 stores the layout every k iterations in frames_
        )

        # anneal>0 cosine-decays the lr to anneal*lr. The objective is stochastic (fresh pivot batch
        # each step), so a constant rate orbits the optimum instead of landing on it
        self.anneal = anneal

        # code_dim>0 concatenates a trainable per-node code to the field input (transductive, zeros
        # out of sample). Sibling leaves / intra-cluster twins have near-identical features
        self.code_dim = code_dim
        self.code_init = code_init
        self.local_w = local_w

        # local_ramp>0 fades the local term in over that fraction of iterations. hard_neg>0 draws
        # that fraction of InfoNCE negatives from the node's own 2-3-hop set
        self.local_ramp = local_ramp
        self.hard_neg = hard_neg
        self.finish = finish

    def fit(self, graph, train_nodes=None):
        nodes, n_nodes, idx, train, tix, loc = self._train_split(graph, train_nodes)
        n_train = len(train)

        npiv = min(self.npiv or 400, n_train - 1)
        rng_pv = np.random.default_rng(self.seed)
        pivots = [train[row] for row in rng_pv.choice(n_train, npiv, replace=False)]
        dist0 = self._pivot_columns(graph, pivots, train)  # one batched BFS sweep

        # transductive fit. train == nodes, so the farthest-first columns ARE the stress targets
        dist = (
            dist0 if train_nodes is None else self._pivot_columns(graph, pivots, nodes)
        )

        npiv = dist.shape[1]
        self.pivots_ = pivots
        fcols = min(self.fcols, npiv)
        self._feat_pivots_ = pivots[:fcols]  # only these columns feed the net input

        if self.features == "diffusion":
            self._land_ = self._farthest_pivots(
                graph, train, min(self.n_landmarks, n_train - 1)
            )[0]
            feat_in = self._input_block(
                graph, nodes
            )  # net input: the stress targets stay hop distances
        else:
            feat_in = dist[:, :fcols]  # reuse the pivot columns already computed above

            if self.local_cache:
                self._feat_cache_ = np.array(feat_in, dtype=np.float32)

        self._fmean = feat_in[tix].mean(0)
        self._fstd = feat_in[tix].std(0) + 1e-9
        self._tb_dim_ = self._resolve_tb(graph)

        feats = _with_tie(
            (feat_in - self._fmean) / self._fstd,
            self._tb_dim_,
            self.tb_alpha,
            self.seed,
        )

        dt = torch.float32
        feat = torch.tensor(feats, dtype=dt)
        feat_tr = feat[torch.tensor(tix)]

        frames = None

        if self.fold_w > 0:  # features fixed: frames built once
            struct = min(self.fold_cols or feats.shape[1], feats.shape[1])

            if self.fold_rand:  # the null: same barrier, frames that know no geometry
                rs = np.random.default_rng(self.seed + 7717)
                fr = rs.standard_normal((n_train, struct, 2)).astype(np.float32)
                fr /= np.linalg.norm(fr, axis=1, keepdims=True) + 1e-9
            else:
                fr = _tangent_frames(
                    np.asarray(feats[tix, :struct], np.float32),
                    self._train_edges(graph, loc),
                    n_train,
                )

            if struct < feats.shape[1]:  # probe only inside the structural subspace
                fr = np.concatenate(
                    [fr, np.zeros((n_train, feats.shape[1] - struct, 2), np.float32)], 1
                )

            frames = torch.tensor(fr, dtype=dt)

        # hop counts are exact in int16, so the pure-sgd path stores the target block 4x smaller and
        # recovers validity from each gathered slice. The majorization path keeps fp32 + a valid
        # mask, with invalid entries set to 1.0 so the loss can divide by the target
        if self.solver == "sgd":
            dist_t = torch.tensor(dist[tix].astype(np.int16))
            valid = None
            nv_col = (dist_t > 0).sum(0)  # exact valid count per pivot column
        else:
            dist_t = torch.tensor(dist[tix], dtype=dt)
            valid = dist_t > 0
            nv_col = valid.sum(0)
            dist_t.masked_fill_(~valid, 1.0)

        lp_i = lp_j = lp_d = np.empty(0)

        if (
            self.solver in ("majorize", "hybrid")
            or self.local_w > 0
            or self.hard_neg > 0
        ):
            # Ortmann's sparse-stress pair set is pivot pairs PLUS local pairs: exact 2- and 3-hop
            # pairs, capped per node so a hub's quadratic neighbourhood cannot explode the set
            elist = self._train_edges(graph, loc)

            adj1 = sp.csr_matrix(
                (np.ones(len(elist)), (elist[:, 0], elist[:, 1])),
                shape=(n_train, n_train),
            )

            adj1 = ((adj1 + adj1.T) > 0).tocsr()
            rng_l = np.random.default_rng(self.seed)
            i_list, j_list, d_list = [], [], []
            reach = (adj1 + sp.identity(n_train, format="csr")) > 0
            frontier = adj1

            for hop in (2, 3):
                newhop = (
                    (frontier @ adj1) > 0
                ).tocsr()  # bool matmul (int counts overflow)

                newhop = (
                    newhop.astype(np.int8) - newhop.multiply(reach).astype(np.int8)
                ).tocsr()

                newhop.eliminate_zeros()  # CSR-native masking of already-reached pairs
                newhop = newhop.astype(bool)
                coo = sp.triu(newhop).tocoo()
                pick = np.arange(len(coo.row))

                if (
                    len(pick) > 16 * n_train
                ):  # cap: hubs otherwise square the pair count
                    pick = rng_l.choice(pick, 16 * n_train, replace=False)

                i_list.append(coo.row[pick])
                j_list.append(coo.col[pick])
                d_list.append(np.full(len(pick), float(hop)))
                reach = (reach + newhop) > 0
                frontier = newhop

            lp_i = np.concatenate(i_list)
            lp_j = np.concatenate(j_list)
            lp_d = np.concatenate(d_list)

            # mid-range pairs get the ALT lower bound from the pivot columns already computed. A
            # distance estimate for ANY pair at no extra BFS
            rnd_j = rng_l.integers(0, n_train, 8 * n_train)
            rnd_i = np.repeat(np.arange(n_train), 8)
            keep = rnd_i != rnd_j
            rnd_i, rnd_j = rnd_i[keep], rnd_j[keep]
            dist_tr = dist[tix]

            # chunked ALT bound. Hop distances are exact in int16, so the result is bit-identical
            dhop = dist_tr.astype(np.int16)
            rnd_d = np.empty(len(rnd_i))

            for start in range(0, len(rnd_i), 262144):
                span = slice(start, start + 262144)
                rnd_d[span] = np.abs(dhop[rnd_i[span]] - dhop[rnd_j[span]]).max(1)

            keep = rnd_d >= 3.5  # local scales are covered exactly above
            lp_i = np.concatenate([lp_i, rnd_i[keep]])
            lp_j = np.concatenate([lp_j, rnd_j[keep]])
            lp_d = np.concatenate([lp_d, rnd_d[keep]])

        negp_t = None

        if self.hard_neg > 0 and len(lp_i):
            sel = lp_d <= 3.0  # 2-3-hop partners only (mid-range excluded)

            hn_i = np.concatenate([lp_i[sel], lp_j[sel]]).astype(np.int64)
            hn_j = np.concatenate([lp_j[sel], lp_i[sel]]).astype(np.int64)

            negpool = np.random.default_rng(self.seed).integers(
                0, n_train, (n_train, 8)
            )

            order = np.argsort(hn_i, kind="stable")
            hi_s, hj_s = hn_i[order], hn_j[order]

            seg_s = np.searchsorted(hi_s, np.arange(n_train))
            seg_e = np.searchsorted(hi_s, np.arange(n_train) + 1)

            for node in range(n_train):
                cands = hj_s[seg_s[node] : seg_e[node]][:8]

                if len(cands):
                    negpool[node, : len(cands)] = cands

            negp_t = torch.tensor(negpool, dtype=torch.long)

        pv = torch.tensor(
            [loc[piv] for piv in pivots], dtype=torch.long
        )  # pivot row within the train block

        edges = torch.as_tensor(
            self._train_edges(graph, loc), dtype=torch.long
        )  # train-internal edges, local

        src_e, dst_e = edges[:, 0], edges[:, 1]
        ones = torch.ones(edges.shape[0], dtype=dt)

        lp_i_t = torch.as_tensor(
            lp_i, dtype=torch.int32
        )  # device copies of the local-pair set (int32

        lp_j_t = torch.as_tensor(
            lp_j, dtype=torch.int32
        )  # halves the resident index memory)

        lp_d_t = torch.tensor(lp_d, dtype=dt)
        torch.manual_seed(self.seed)

        use_code = (
            self.code_dim > 0 and train_nodes is None
        )  # codes are transductive, drop them for OOS

        codes = (
            torch.nn.Parameter(
                torch.randn(n_train, self.code_dim, dtype=dt) * self.code_init
            )
            if use_code
            else None
        )

        self._code_pad_ = (
            self.code_dim if use_code else 0
        )  # transform pads zeros for these columns

        inr = Gabor(
            feat.shape[1] + self._code_pad_,
            self.n_components,
            width=self.width,
            depth=self.depth,
            w0=self.w0,
            act=self.act,
        ).float()

        opt = torch.optim.Adam(
            list(inr.parameters()) + ([codes] if codes is not None else []), lr=self.lr
        )

        rng = np.random.default_rng(self.seed)
        bat = self.batch if (self.batch and self.batch < npiv) else npiv
        t0 = time.perf_counter()
        self.frames_ = []

        for step in range(self.iters):
            if self.anneal > 0:  # cosine decay to anneal*lr
                fdec = self.anneal + (1 - self.anneal) * 0.5 * (
                    1 + np.cos(np.pi * step / self.iters)
                )

                for grp in opt.param_groups:
                    grp["lr"] = self.lr * fdec

            if self.record_every and step % self.record_every == 0:
                with torch.no_grad():
                    self.frames_.append(
                        inr(feat if codes is None else torch.cat([feat, codes], 1))
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )

            coords = inr(feat_tr if codes is None else torch.cat([feat_tr, codes], 1))

            # solver="hybrid". Majorization first, then the sgd loss so the InfoNCE term can
            # re-sharpen local neighbourhoods from a well-placed start
            maj_now = self.solver == "majorize" or (
                self.solver == "hybrid" and step < int(0.75 * self.iters)
            )

            if maj_now:
                with torch.no_grad():  # target built on-device, same formulas
                    coords_d = coords.detach()
                    coords_p = coords_d[pv]

                    # per-node sums are row-independent, so row blocks give the identical target at a
                    # fraction of the peak memory. Small graphs stay single-chunk (cached)
                    chunk = n_train if n_train * npiv <= 50_000_000 else 131072
                    cache, s1, s2 = [], 0.0, 0.0

                    for start in range(0, n_train, chunk):
                        diff = (
                            coords_d[start : start + chunk, None, :]
                            - coords_p[None, :, :]
                        )  # (chunk, npiv, dim)

                        dcur = diff.square().sum(-1).clamp_min(1e-18).sqrt()
                        valid_c, dist_c = (
                            valid[start : start + chunk],
                            dist_t[start : start + chunk],
                        )
                        s1 = s1 + (dcur[valid_c] / dist_c[valid_c]).sum()
                        s2 = (
                            s2
                            + (dcur[valid_c].square() / dist_c[valid_c].square()).sum()
                        )

                        if chunk >= n_train:
                            cache.append((diff, dcur))

                    scale = (s1 / (s2 + 1e-12)).clamp_min(1e-12)
                    num = torch.empty_like(coords_d)
                    den = torch.empty(n_train, dtype=dt)

                    for chunk_i, start in enumerate(range(0, n_train, chunk)):
                        if cache:
                            diff, dcur = cache[chunk_i]
                        else:
                            diff = (
                                coords_d[start : start + chunk, None, :]
                                - coords_p[None, :, :]
                            )
                            dcur = diff.square().sum(-1).clamp_min(1e-18).sqrt()

                        rest = (
                            dist_t[start : start + chunk] / scale
                        )  # rest lengths in layout units

                        weight = torch.where(
                            valid[start : start + chunk],
                            1.0 / rest.clamp_min(1e-9).square(),
                            torch.zeros((), dtype=dt),
                        )

                        tgt = (
                            coords_p[None, :, :] + (rest / dcur)[:, :, None] * diff
                        )  # where each constraint wants i

                        num[start : start + chunk] = (weight[:, :, None] * tgt).sum(1)
                        den[start : start + chunk] = weight.sum(1)

                    if edges.shape[0]:  # edge constraints, rest 1/a
                        dist_e = (
                            (coords_d[src_e] - coords_d[dst_e])
                            .square()
                            .sum(-1)
                            .clamp_min(1e-18)
                            .sqrt()
                        )

                        rest_e = 1.0 / scale
                        weight_e = 1.0 / rest_e**2

                        tgt_a = coords_d[dst_e] + (rest_e / dist_e)[:, None] * (
                            coords_d[src_e] - coords_d[dst_e]
                        )
                        tgt_b = coords_d[src_e] + (rest_e / dist_e)[:, None] * (
                            coords_d[dst_e] - coords_d[src_e]
                        )

                        num.index_add_(0, src_e, weight_e * tgt_a)
                        den.index_add_(0, src_e, weight_e * ones)
                        num.index_add_(0, dst_e, weight_e * tgt_b)
                        den.index_add_(0, dst_e, weight_e * ones)

                    # local (2-3 hop) pairs: mesh rigidity. index_add_ accumulates via atomics, so
                    # chunking only reorders the same adds
                    for pstart in range(0, len(lp_i), 4_000_000):
                        src_p, dst_p = (
                            lp_i_t[pstart : pstart + 4_000_000],
                            lp_j_t[pstart : pstart + 4_000_000],
                        )

                        dist_l = (
                            (coords_d[src_p] - coords_d[dst_p])
                            .square()
                            .sum(-1)
                            .clamp_min(1e-18)
                            .sqrt()
                        )

                        rest_l = lp_d_t[pstart : pstart + 4_000_000] / scale
                        weight_l = 1.0 / rest_l.square()

                        tgt_a = coords_d[dst_p] + (rest_l / dist_l)[:, None] * (
                            coords_d[src_p] - coords_d[dst_p]
                        )
                        tgt_b = coords_d[src_p] + (rest_l / dist_l)[:, None] * (
                            coords_d[dst_p] - coords_d[src_p]
                        )

                        num.index_add_(0, src_p, weight_l[:, None] * tgt_a)
                        den.index_add_(0, src_p, weight_l)
                        num.index_add_(0, dst_p, weight_l[:, None] * tgt_b)
                        den.index_add_(0, dst_p, weight_l)

                    target = num / den.clamp_min(1e-12)[:, None]

                loss = ((coords - target) ** 2).mean() * 10.0
            else:
                cols = (
                    rng.choice(npiv, bat, replace=False) if bat < npiv else slice(None)
                )

                # mask-as-weight instead of masked_select. torch.where zeroes the ~bat invalid terms
                # in place instead of paying a select + reduce + scatter backward per iteration
                dpred = torch.cdist(coords, coords[pv[cols]]).clamp_min(1e-9)
                dtarg = dist_t[:, cols]

                if valid is None:  # int16 hop targets (see dist_t setup above)
                    valid_c = dtarg > 0
                    dist_c = dtarg.clamp_min(1).to(dt)
                else:
                    valid_c = valid[:, cols]
                    dist_c = dtarg

                ratio = torch.where(valid_c, dpred / dist_c, torch.zeros((), dtype=dt))
                s1 = ratio.sum()
                s2 = (ratio * ratio).sum()

                cnt = (
                    nv_col[cols].sum() + edges.shape[0]
                )  # stays a device tensor: no host sync

                if edges.shape[0]:
                    dist_e = (coords[src_e] - coords[dst_e]).norm(dim=1).clamp_min(1e-9)
                    s1 = s1 + dist_e.sum()
                    s2 = s2 + (dist_e * dist_e).sum()
                scale = s1 / (s2 + 1e-12)

                res = torch.where(
                    valid_c,
                    (scale.detach() * dpred - dist_c) / dist_c,
                    torch.zeros((), dtype=dt),
                )

                loss = (res * res).sum()

                if edges.shape[0]:
                    loss = loss + ((scale.detach() * dist_e - 1.0) ** 2).sum()

                loss = loss / cnt

                lw_t = (
                    self.local_w
                    if self.local_ramp <= 0
                    else self.local_w
                    * min(1.0, step / max(1.0, self.local_ramp * self.iters))
                )

                if lw_t > 0 and len(lp_i):
                    lbat = rng.integers(0, len(lp_i), min(4096, len(lp_i)))

                    cur_lp = (
                        (
                            coords[torch.as_tensor(lp_i[lbat])]
                            - coords[torch.as_tensor(lp_j[lbat])]
                        )
                        .norm(dim=1)
                        .clamp_min(1e-9)
                    )

                    tgt_lp = torch.tensor(lp_d[lbat], dtype=dt)
                    scl_lp = (cur_lp / tgt_lp).sum() / (
                        (cur_lp**2 / tgt_lp**2).sum() + 1e-12
                    )

                    loss = (
                        loss
                        + lw_t
                        * (((scl_lp.detach() * cur_lp - tgt_lp) / tgt_lp) ** 2).mean()
                    )

            if self.rep_w > 0:  # maxent: entropy repulsion, non-neighbours
                neg = torch.randint(0, n_train, (n_train, self.rep_K))
                dneg = (coords[:, None, :] - coords[neg]).norm(dim=-1).clamp_min(1e-9)
                loss = loss - self.rep_w * torch.log(scale.detach() * dneg).mean()

            if self.knn_w > 0 and edges.shape[0]:
                # InfoNCE: neighbours in, random negatives out
                ebat = rng.choice(
                    edges.shape[0], min(256, edges.shape[0]), replace=False
                )
                src_i, dst_i = src_e[ebat], dst_e[ebat]
                neg = torch.randint(0, n_train, (len(ebat), self.knn_K))

                if negp_t is not None:
                    n_hard = max(1, int(round(self.hard_neg * self.knn_K)))
                    neg[:, :n_hard] = negp_t[
                        src_i.unsqueeze(1), torch.randint(0, 8, (len(ebat), n_hard))
                    ]

                qpos = 1.0 / (1.0 + ((coords[src_i] - coords[dst_i]) ** 2).sum(1))
                qneg = 1.0 / (
                    1.0 + ((coords[src_i].unsqueeze(1) - coords[neg]) ** 2).sum(2)
                )

                loss = (
                    loss
                    + self.knn_w
                    * (-torch.log(qpos / (qpos + qneg.sum(1) + 1e-9) + 1e-9)).mean()
                )

            if (
                frames is not None
                and self.fold_w > 0
                and step >= self.fold_start * self.iters
            ):
                rows = torch.as_tensor(
                    rng.choice(n_train, min(256, n_train), replace=False)
                )

                loss = loss + self.fold_w * _fold_penalty(
                    inr, feat_tr, frames, rows, self.fold_margin, self.fold_sign
                )

            opt.zero_grad()
            loss.backward()
            opt.step()

        self.net_ = inr
        self.codes_ = None if codes is None else codes.detach()

        emb_in = (
            feat if codes is None else torch.cat([feat, codes], 1)
        )  # transductive fit. n_train == n_nodes

        self.embedding_ = (
            inr(emb_in).detach().cpu().numpy().astype(np.float64)
        )  # all nodes (held-out via forward pass)

        if self.record_every:
            self.frames_.append(self.embedding_)

        self.delta_ = None

        if self.finish > 0 and train_nodes is None and n_nodes <= 6000:
            dfull = np.asarray(
                self._pivot_columns(graph, nodes, nodes), np.float32
            )  # full BFS. Every node a pivot

            coords_o = self._smacof(
                self.embedding_[:, :2] if self.n_components > 2 else self.embedding_,
                dfull,
                self.finish,
            )

            if self.n_components > 2:  # finish is planar: keep extra dims untouched
                self.delta_ = np.concatenate(
                    [
                        coords_o - self.embedding_[:, :2],
                        np.zeros((n_nodes, self.n_components - 2)),
                    ],
                    1,
                )
            else:
                self.delta_ = coords_o - self.embedding_

            self.embedding_ = self.embedding_ + self.delta_

        self._piv_dist_ = {
            nodes[row]: dist[row].astype(np.float64) for row in range(n_nodes)
        }  # cache for incremental updates

        self.fit_time_ = time.perf_counter() - t0

        return self

    @staticmethod
    def _smacof(coords, D, iters):
        coords = np.asarray(coords, np.float32).copy()

        D = np.asarray(D, np.float32)
        weight = np.where(D > 0, 1.0 / np.maximum(D, 1e-9) ** 2, 0.0).astype(np.float32)

        np.fill_diagonal(weight, 0.0)
        wsum = weight.sum(1)[:, None]
        wd = weight * D

        for _ in range(int(iters)):
            dcur = np.sqrt(
                np.maximum(((coords[:, None] - coords[None]) ** 2).sum(-1), 1e-12)
            )
            ratio = wd / dcur
            np.fill_diagonal(ratio, 0.0)
            coords = (
                weight @ coords + coords * ratio.sum(1)[:, None] - ratio @ coords
            ) / wsum

        return coords.astype(np.float64)

    def _pivot_columns_incremental(self, graph, new_nodes, nodes):
        npiv = len(self.pivots_)
        newset = set(new_nodes)
        dist = {
            node: self._piv_dist_[node] for node in nodes if node in self._piv_dist_
        }

        for node in new_nodes:
            best = np.full(npiv, 1e6)

            for nbr in graph[node]:
                if nbr in dist and nbr not in newset:
                    best = np.minimum(best, dist[nbr] + 1.0)

            dist[node] = best

        for _ in range(len(new_nodes) + 1):  # relax over new-new edges
            changed = False

            for node in new_nodes:
                row = dist[node]

                for nbr in graph[node]:
                    if nbr in dist:
                        cand = dist[nbr] + 1.0

                        if (cand < row).any():
                            row = np.minimum(row, cand)
                            changed = True

                dist[node] = row

            if not changed:
                break

        return np.array([dist[node] for node in nodes])

    def partial_fit(
        self,
        graph,
        new_nodes=None,
        steps=200,
        lr=None,
        anchor=8.0,
        rep_w=0.1,
        n_neg=8,
        local=0,
        incremental=False,
        refresh_pivots=False,
    ):
        if not hasattr(self, "net_"):
            raise RuntimeError("call fit before partial_fit")

        if getattr(self, "_code_pad_", 0):
            raise RuntimeError(
                "partial_fit does not support per-node codes (they are transductive): "
                "refit, or fit with code_dim=0"
            )

        nodes = list(graph.nodes())
        n_nodes = len(nodes)
        idx = {node: row for row, node in enumerate(nodes)}
        train = set(self.nodes_)

        new_nodes = (
            [node for node in nodes if node not in train]
            if new_nodes is None
            else list(new_nodes)
        )

        diffusion = getattr(self, "features", "pivot") == "diffusion"

        if refresh_pivots:  # re-select landmarks over the GROWN
            piv, dpiv = self._farthest_pivots(
                graph, nodes, min(self.npiv or len(self.pivots_), n_nodes - 1)
            )  # graph
            self.pivots_ = piv

            if not diffusion:
                fcols = min(self.fcols, dpiv.shape[1])
                self._feat_pivots_ = piv[:fcols]
                self._fmean = dpiv[:, :fcols].mean(0)
                self._fstd = dpiv[:, :fcols].std(0) + 1e-9
        elif incremental and hasattr(self, "_piv_dist_") and not diffusion:
            # cheap distances (no full BFS)
            dpiv = self._pivot_columns_incremental(graph, new_nodes, nodes)
        else:
            dpiv = self._pivot_columns(
                graph, self.pivots_, nodes
            )  # stress targets to all pivots

        if diffusion:
            # landmarks are fixed at fit: re-propagate over the grown graph
            feats = self._features(graph, nodes)
        else:
            fcols = len(self._feat_pivots_)  # first fcols of pivots_ feed the net
            feats = (dpiv[:, :fcols] - self._fmean) / self._fstd

            if self._tb_dim_ > 0:
                feats = np.concatenate(
                    [
                        feats,
                        tie_features(n_nodes, self._tb_dim_, self.tb_alpha, self.seed),
                    ],
                    1,
                )

        dt = next(self.net_.parameters()).dtype
        feat_all = torch.tensor(feats, dtype=dt)

        cur = {
            node: self.embedding_[row] for row, node in enumerate(self.nodes_)
        }  # OUTPUT-space anchor targets

        if local > 0:  # working set = new + k-hop + landmarks
            keep = set(new_nodes)
            front = set(new_nodes)

            for _ in range(local):
                front = {
                    nbr for node in front for nbr in graph[node] if nbr not in keep
                }
                keep |= front

            keep |= set(self.pivots_)
            work = [node for node in nodes if node in keep]
        else:
            work = nodes

        wrow = {node: row for row, node in enumerate(work)}
        wglob = [idx[node] for node in work]
        feat = feat_all[wglob]
        dist_t = torch.tensor(dpiv[wglob], dtype=dt)
        valid = dist_t > 0
        pv = torch.tensor([wrow[piv] for piv in self.pivots_], dtype=torch.long)
        wset = set(work)

        elist = [
            [wrow[src], wrow[dst]]
            for src, dst in graph.edges()
            if src in wset and dst in wset
        ]
        edges = (
            torch.tensor(elist, dtype=torch.long)
            if elist
            else torch.empty((0, 2), dtype=torch.long)
        )

        src_e, dst_e = edges[:, 0], edges[:, 1]
        ones = torch.ones(edges.shape[0], dtype=dt)
        amask = torch.tensor([node in cur for node in work])
        apos = torch.tensor(
            np.array([cur.get(node, np.zeros(self.n_components)) for node in work]),
            dtype=dt,
        )

        new_w = torch.tensor(
            [wrow[node] for node in new_nodes if node in wrow], dtype=torch.long
        )  # new nodes within work

        scale = float(np.abs(self.embedding_ - self.embedding_.mean(0)).max() + 1e-9)
        opt = torch.optim.Adam(self.net_.parameters(), lr=lr or self.lr)
        rng = np.random.default_rng(self.seed + 1)
        npiv = dist_t.shape[1]
        bat = self.batch if (self.batch and self.batch < npiv) else npiv
        n_work = len(work)

        for _ in range(steps):
            coords = self.net_(feat)
            cols = rng.choice(npiv, bat, replace=False) if bat < npiv else slice(None)
            dpred = torch.cdist(coords, coords[pv[cols]]).clamp_min(1e-9)
            valid_c = valid[:, cols]
            cur_d = dpred[valid_c]
            tgt_d = dist_t[:, cols][valid_c]

            if edges.shape[0]:
                dist_e = (coords[src_e] - coords[dst_e]).norm(dim=1).clamp_min(1e-9)
                cur_d = torch.cat([cur_d, dist_e])
                tgt_d = torch.cat([tgt_d, ones])

            scl = (cur_d / tgt_d).sum() / ((cur_d**2 / tgt_d**2).sum() + 1e-12)
            loss = (((scl.detach() * cur_d - tgt_d) / tgt_d) ** 2).mean()

            if rep_w > 0 and len(new_w):  # spread the NEW nodes only (they are
                neg = torch.as_tensor(
                    rng.integers(0, n_work, (len(new_w), n_neg)), dtype=torch.long
                )  # unanchored, so
                dneg = (
                    (coords[new_w][:, None, :] - coords[neg])
                    .norm(dim=-1)
                    .clamp_min(1e-9)
                )  # the base stays put
                loss = loss - rep_w * torch.log(scl.detach() * dneg).mean()

            if anchor > 0:  # output-space stability (mental map)
                loss = (
                    loss
                    + anchor * (((coords[amask] - apos[amask]) / scale) ** 2).mean()
                )

            opt.zero_grad()
            loss.backward()
            opt.step()

        self.embedding_ = (
            self.net_(feat_all).detach().cpu().numpy().astype(np.float64)
        )  # recompute all: invariant holds

        self.nodes_ = nodes
        self._piv_dist_ = {
            node: dpiv[row] for row, node in enumerate(nodes)
        }  # refresh cache with the new rows

        self.edges_ = (
            np.array(
                [[idx[src], idx[dst]] for src, dst in graph.edges()], dtype=np.int64
            )
            if graph.number_of_edges()
            else np.empty((0, 2), np.int64)
        )

        return (
            np.array([self.embedding_[idx[node]] for node in new_nodes])
            if new_nodes
            else self.embedding_
        )


class Fling(_BaseFling):
    def __init__(
        self,
        n_components=3,
        iters=400,
        n_pivots=16,
        m_anchor=80,
        fit_steps=6,
        lr=5e-3,
        grav=0.1,
        w0=4.0,
        act="gelu",
        width=128,
        depth=2,
        tb_dim=None,
        tb_alpha=0.03,
        seed=0,
        features="diffusion",
        n_landmarks=64,
        alpha=0.05,
        diff_K=20,
        rep_act=None,
        rep_w0=8.0,
        rep_width=None,
        rep_depth=None,
        force="shell",
        rep_cutoff=0.0,
        near_rep=0.0,
        record_every=0,
        anneal=0.0,
        local_cache=False,
        fold_w=0.0,
        fold_margin=0.1,
    ):
        super().__init__(
            n_components, w0, act, width, depth, tb_dim, tb_alpha, seed, local_cache
        )

        # fold_w>0 adds the tangent-Jacobian barrier (see _fold_penalty): the field is penalised
        # where it turns a patch of the graph over on itself. Off by default: it changes the layouts
        self.fold_w = fold_w
        self.fold_margin = fold_margin
        self.iters = iters
        self.n_pivots = n_pivots
        self.m_anchor = m_anchor
        self.fit_steps = fit_steps
        self.lr = lr
        self.grav = grav

        # the repulsion field's best architecture depends on the law: gelu/128/1 for the FR-family
        # laws, gabor/48/2 otherwise. None resolves per law at fit time
        self.rep_act = rep_act
        self.rep_w0 = rep_w0
        self.rep_width = rep_width
        self.rep_depth = rep_depth

        # force: 'fr' | 'linlog' | 'fa2' | 'metric' (repulsion weighted by feature distance) |
        # 'shell' (the majorization default). rep_cutoff>0 truncates the repulsion at cutoff*k
        self.force = force
        self.rep_cutoff = rep_cutoff

        # near_rep>0 adds the exact repulsion between pairs closer than near_rep*k (the
        # particle-particle half of P3M), which the mean-field anchor net cannot represent
        self.near_rep = near_rep
        self.features = features  # "pivot" hop distances | "diffusion" potentials
        self.n_landmarks = n_landmarks
        self.alpha = alpha
        self.diff_K = diff_K

        # record_every>0 stores the layout of every node each k iterations in frames_
        self.record_every = record_every

        # the majorisation masses are fit at m RESAMPLED anchors each step, so the update is
        # stochastic and the drawing keeps jittering. anneal>0 cosine-decays the lr to anneal*lr
        self.anneal = anneal

    def fit(self, graph, train_nodes=None):
        nodes, n_nodes, idx, train, tix, loc = self._train_split(graph, train_nodes)
        n_train = len(train)
        npiv = min(self.n_pivots, n_train - 1)

        pivots, dist0 = self._farthest_pivots(
            graph, train, npiv
        )  # pivots drawn from the training nodes

        self.pivots_ = pivots
        self._feat_pivots_ = pivots  # all pivots feed the net input

        if self.features == "diffusion":
            self._land_ = self._farthest_pivots(
                graph, train, min(self.n_landmarks, n_train - 1)
            )[0]

        dist = self._input_block(graph, nodes)  # net input for every node

        self._fmean = dist[tix].mean(0)
        self._fstd = dist[tix].std(0) + 1e-9
        self._tb_dim_ = self._resolve_tb(graph)

        feats = _with_tie(
            (dist - self._fmean) / self._fstd, self._tb_dim_, self.tb_alpha, self.seed
        )

        shell_ = self.force == "shell"

        # the shell path runs in float32 (double is 5-8x slower on CPU for no layout benefit). The
        # FR-family force laws keep the Gabor default double: in float32 their dynamics collapse
        # to a needle
        feat = torch.tensor(feats, dtype=torch.float32 if shell_ else torch.float64)
        feat_tr = feat[torch.as_tensor(tix)]

        edges = self._train_edges(graph, loc)  # train-internal edges, local indices
        kdist = np.sqrt(1.0 / n_train)
        dim = self.n_components
        fr_law = self.force in ("fr", "metric", "shell")  # laws with the k^2 scale

        self.rep_act_ = self.rep_act or ("gelu" if fr_law else "gabor")
        self.rep_width_ = self.rep_width or (128 if fr_law else 48)
        self.rep_depth_ = self.rep_depth or (1 if fr_law else 2)

        metric = self.force in ("metric", "shell")  # feature distance ~ graph distance
        shell = self.force == "shell"

        if metric:  # diffusion features are smooth, so the
            fcent = feats[tix] - feats[tix].mean(
                0
            )  # potentials are low-rank: PCA-8 keeps the
            _, _, vt = np.linalg.svd(
                fcent, full_matrices=False
            )  # metric and shrinks every force tensor ~8x
            fpca = np.ascontiguousarray(fcent @ vt[:8].T, np.float32)
            fsq = (fpca**2).sum(1)

            samp = fpca[
                np.random.default_rng(self.seed).choice(
                    n_train, min(256, n_train), replace=False
                )
            ]

            rbar = (
                float(
                    np.sqrt(
                        np.maximum(((samp[:, None] - samp[None]) ** 2).sum(-1), 0)
                    ).mean()
                )
                + 1e-12
            )

            # per-node scale: diffusion under-visits low-degree nodes and inflates ALL their
            # distances. Dividing r_ij by sqrt(s_i s_j) (each node's mean distance to the sample,
            # normalised to mean 1) deflates that inflation continuously
            nscale = np.empty(n_train, np.float32)

            for start in range(
                0, n_train, 20000
            ):  # chunked: (blk, 256, 8) temporaries only
                blk = fpca[start : start + 20000]
                nscale[start : start + 20000] = np.sqrt(
                    np.maximum(((blk[:, None] - samp[None]) ** 2).sum(-1), 0)
                ).mean(1)

            nscale = np.sqrt(nscale / (nscale.mean() + 1e-12)).astype(
                np.float32
            )  # sqrt(s_i), mean-1 scale

        if shell:  # edges are handled exactly: the mean field
            # cannot resolve pair-specific near forces
            src_e, dst_e = edges[:, 0], edges[:, 1]
            ones_e = np.ones(len(edges))

            adj = sp.csr_matrix((ones_e, (src_e, dst_e)), shape=(n_train, n_train))
            adj = ((adj + adj.T) > 0).tocsr()

            # rest lengths come from the hop metric, not the diffusion metric (a star metric on
            # hairballs). The landmark bound max_l |h_il - h_jl| over the pivots already computed is
            # a lower bound on the shortest-path distance, uniform (<=1) on edges, long on bridges
            hpiv = (
                np.ascontiguousarray(dist0, np.float32)
                if train_nodes is None
                else np.ascontiguousarray(
                    self._pivot_columns(graph, pivots, nodes), np.float32
                )[tix]
            )

            hop_e = np.maximum(
                np.abs(hpiv[src_e] - hpiv[dst_e]).max(1), 0.9
            )  # raw hop units (edges are 1 hop)

            hcent = hpiv - hpiv.mean(0)
            _, _, vt8 = np.linalg.svd(hcent, full_matrices=False)
            hpca = np.ascontiguousarray(
                hcent @ vt8[:8].T, np.float32
            )  # field input: hop-PCA

            if hpca.shape[1] < 8:  # tiny graphs: fewer pivots than 8
                hpca = np.pad(hpca, ((0, 0), (0, 8 - hpca.shape[1])))
            hpca /= hpca.std() + 1e-9

            # device copies: the shell update runs on-device end to end
            hpiv_t = torch.tensor(hpiv)
            hpca_t = torch.tensor(hpca)
            hop_e_t = torch.tensor(hop_e, dtype=torch.float32)
            src_e_t = torch.as_tensor(src_e, dtype=torch.long)
            dst_e_t = torch.as_tensor(dst_e, dtype=torch.long)

        torch.manual_seed(self.seed)

        inr = Gabor(
            feat.shape[1],
            dim,
            width=self.width,
            depth=self.depth,
            w0=self.w0,
            act=self.act,
        )

        rep_net = Gabor(
            dim + (8 if metric else 0),
            dim + (1 if shell else 0),
            width=self.rep_width_,
            depth=self.rep_depth_,
            w0=self.rep_w0,
            act=self.rep_act_,
            zero_last=True,
        )

        if shell:
            inr = inr.float()  # see dtype note at feat above
            rep_net = rep_net.float()
        elif metric:
            # float32. The wide-input float64 ops fall into torch's threaded small-op pathology
            rep_net = rep_net.float()

        opt_i = _adam(inr.parameters(), self.lr)
        opt_r = _adam(rep_net.parameters(), 5e-3)

        n_anc = min(self.m_anchor, n_train)
        rng = np.random.default_rng(self.seed)
        temp = 0.1
        frames = None

        if self.fold_w > 0:
            # frames are fixed: features never change
            frames = torch.tensor(
                _tangent_frames(np.asarray(feats[tix], np.float32), edges, n_train),
                dtype=feat.dtype,
            )

        deg = np.array(
            [graph.degree(node) for node in train], float
        )  # ForceAtlas2 law only

        def rescale(pos):
            # robust scale: one flung node must not
            pos = pos - pos.mean(0)  # inflate the max and collapse the rest

            return pos / (np.percentile(np.linalg.norm(pos, axis=1), 98) + 1e-9)

        def rescale_t(pos):  # torch twin, kept on device
            pos = pos - pos.mean(0)

            return pos / (torch.quantile(pos.norm(dim=1), 0.98) + 1e-9)

        t0 = time.perf_counter()
        self.frames_ = []

        for step in range(self.iters):
            if self.anneal > 0:  # cosine decay to anneal*lr
                fdec = self.anneal + (1 - self.anneal) * 0.5 * (
                    1 + np.cos(np.pi * step / self.iters)
                )

                for grp in opt_i.param_groups:
                    grp["lr"] = self.lr * fdec

            if self.record_every and step % self.record_every == 0:
                self.frames_.append(inr(feat).detach().cpu().numpy().astype(np.float64))

            coords = inr(feat_tr)
            anc = rng.choice(n_train, n_anc, replace=False)

            if shell:
                # mean-field stress majorization (Gansner et al.). x <- (sum_j w (x_j + l u_ij)) /
                # (sum_j w), w = 1/l^2. Numerator and denominator are pair sums, so the anchor-field
                # trick applies to them verbatim. Edges are accumulated exactly and excluded at the
                # anchors.
                with torch.no_grad():
                    pos = rescale_t(
                        coords.detach()
                    )  # positions of the training nodes, on device

                    anc_t = torch.as_tensor(anc)
                    ancpos = pos[anc_t]
                    dmat = (
                        (
                            ancpos.square().sum(1)[:, None]
                            + pos.square().sum(1)[None, :]
                            - 2.0 * ancpos @ pos.T
                        )
                        .clamp_min(1e-12)
                        .sqrt()
                    )

                    hpiv_a = hpiv_t[anc_t]
                    altmax = torch.zeros_like(dmat)

                    for col in range(hpiv_t.shape[1]):
                        altmax = torch.maximum(
                            altmax,
                            (hpiv_a[:, col][:, None] - hpiv_t[:, col][None, :]).abs(),
                        )

                    rest_h = altmax.clamp_min(0.9)
                    hscale = (dmat * rest_h).sum() / (
                        rest_h.square().sum() + 1e-12
                    )  # hop -> layout scale

                    rest = hscale * rest_h
                    weight = 1.0 / (rest * rest)
                    weight[torch.arange(len(anc)), anc_t] = 0.0
                    erow, ecol = adj[anc].nonzero()  # edges handled exactly below

                    weight[
                        torch.as_tensor(erow, dtype=torch.long),
                        torch.as_tensor(ecol, dtype=torch.long),
                    ] = 0.0

                    ratio = weight * rest / dmat

                    num_ex = (
                        weight @ pos
                        + ancpos * ratio.sum(1)[:, None]
                        - ratio @ pos
                        - ancpos * weight.sum(1)[:, None]
                    ) / n_train

                    den_ex = weight.sum(1) / n_train
                    x_anc = torch.cat([ancpos, hpca_t[anc_t]], 1)
                    y_anc = torch.cat([num_ex, den_ex[:, None]], 1)

                for _ in range(self.fit_steps):
                    opt_r.zero_grad()
                    ((rep_net(x_anc) - y_anc) ** 2).mean().backward()
                    opt_r.step()

                with torch.no_grad():
                    out = rep_net(torch.cat([pos, hpca_t], 1))
                    num_f, den_f = out[:, :dim], out[:, dim]
                    rest_e = hscale * hop_e_t
                    weight_e = 1.0 / (rest_e * rest_e)
                    dist_e = (pos[src_e_t] - pos[dst_e_t]).norm(dim=1) + 1e-12
                    egrad = (weight_e * (1.0 - rest_e / dist_e))[:, None] * (
                        pos[dst_e_t] - pos[src_e_t]
                    )  # A-contribution to src_e

                    num_e = torch.zeros_like(pos)
                    den_e = torch.zeros(n_train)
                    num_e.index_add_(0, src_e_t, egrad)
                    num_e.index_add_(0, dst_e_t, -egrad)
                    den_e.index_add_(0, src_e_t, weight_e)
                    den_e.index_add_(0, dst_e_t, weight_e)
                    num_e = num_e / n_train
                    den_e = den_e / n_train
                    den_tot = torch.maximum(den_f + den_e, 0.05 * den_ex.mean() + 1e-12)

                    vel = (num_f + num_e) / den_tot[:, None]
            else:
                pos = rescale(
                    coords.detach().cpu().numpy()
                )  # numpy path: FR-family laws stay as-is

                if metric:
                    # graph-far pairs repel harder (anti-fold)
                    fdist = np.sqrt(
                        np.maximum(
                            fsq[anc][:, None] + fsq[None, :] - 2.0 * fpca[anc] @ fpca.T,
                            0.0,
                        )
                    )

                    metw = np.maximum(
                        fdist / rbar / (nscale[anc][:, None] * nscale[None, :]), 0.05
                    )

                    x_anc = torch.tensor(
                        np.concatenate([pos[anc].astype(np.float32), fpca[anc]], 1)
                    )
                else:
                    metw = None
                    x_anc = torch.tensor(pos[anc])

                y_anc_np = _repulsion_at(
                    pos,
                    anc,
                    kdist,
                    self.force,
                    deg,
                    self.rep_cutoff,
                    near=self.near_rep,
                    metw=metw,
                )

                y_anc = torch.tensor(
                    y_anc_np.astype(np.float32) if metric else y_anc_np
                )

                for _ in range(self.fit_steps):
                    opt_r.zero_grad()
                    ((rep_net(x_anc) - y_anc) ** 2).mean().backward()
                    opt_r.step()

                with torch.no_grad():  # metric: the far field is itself feature-
                    rep_in = (
                        np.concatenate([pos.astype(np.float32), fpca], 1)
                        if metric
                        else pos
                    )

                    rep = rep_net(torch.tensor(rep_in)).cpu().numpy().astype(np.float64)

                att = _attraction(pos, edges, kdist, self.force)

                if self.near_rep > 0:
                    rep = rep + _near_repulsion(
                        pos, kdist, self.near_rep * kdist, self.force, deg, rng
                    )

                vel = rep + att - self.grav * pos

            if shell:  # majorization's stability comes from the
                with torch.no_grad():  # FULL Jacobi step. A capped step recreates
                    tgt = rescale_t(
                        pos + vel
                    )  # the slow flow the update exists to escape
            else:
                mag = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-9
                target = rescale(pos + np.minimum(mag, temp) * vel / mag)
                tgt = torch.tensor(target, dtype=coords.dtype)

            loss = ((coords - tgt) ** 2).mean()

            if frames is not None:
                rows = torch.as_tensor(
                    rng.choice(n_train, min(256, n_train), replace=False)
                )

                loss = loss + self.fold_w * _fold_penalty(
                    inr, feat_tr, frames, rows, self.fold_margin
                )

            opt_i.zero_grad()
            loss.backward()
            opt_i.step()
            temp = max(temp * 0.99, 1e-3)

        self.net_ = inr
        self.embedding_ = (
            inr(feat).detach().cpu().numpy().astype(np.float64)
        )  # all nodes (held-out via forward pass)

        if self.record_every:
            self.frames_.append(self.embedding_)

        self.fit_time_ = time.perf_counter() - t0

        return self


def neulay(graph, iters=600, hid=16, width=64, lr=5e-3, seed=0):
    graph = nx.convert_node_labels_to_integers(graph)
    edges = torch.tensor(np.array(graph.edges()), dtype=torch.long)

    n_nodes = graph.number_of_nodes()

    kdist = 1.0

    torch.manual_seed(seed)
    codes = torch.nn.Parameter(torch.randn(n_nodes, hid, dtype=torch.float64))

    net = torch.nn.Sequential(
        torch.nn.Linear(hid, width),
        torch.nn.ReLU(),
        torch.nn.Linear(width, width),
        torch.nn.ReLU(),
        torch.nn.Linear(width, 2),
    ).double()

    opt = torch.optim.Adam([codes] + list(net.parameters()), lr=lr)
    t0 = time.perf_counter()

    for _ in range(iters):
        opt.zero_grad()
        _fr_energy(net(codes), edges, kdist).backward()
        opt.step()

    return net(codes).detach().cpu().numpy(), time.perf_counter() - t0, None
