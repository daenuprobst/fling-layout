import time

import numpy as np
import torch

from .estimators import _adam, _BaseFling
from .nets import Gabor, _with_tie
from .optim import build as _muon


def seg_dist(pts, seg_a, seg_b):
    ab = seg_b - seg_a
    t = ((pts - seg_a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)
    proj = seg_a + t.clamp(0, 1).unsqueeze(-1) * ab

    return (pts - proj).norm(dim=-1)


def near_edges(pos, edges, node_idx, k_cand, pool=None):
    # k candidate edges per node, ranked by distance to the edge's two ends and middle
    use = edges if pool is None else edges[pool]
    ends = pos[use]
    mid = ends.mean(1)
    query = pos[node_idx]

    dist = torch.minimum(
        torch.cdist(query, mid),
        torch.minimum(torch.cdist(query, ends[:, 0]), torch.cdist(query, ends[:, 1])),
    )

    inc = (use[None, :, 0] == node_idx[:, None]) | (
        use[None, :, 1] == node_idx[:, None]
    )

    dist = dist.masked_fill(inc, float("inf"))  # incident edges are not occlusions
    idx = dist.topk(min(k_cand, use.shape[0]), dim=1, largest=False).indices

    return idx if pool is None else pool[idx]


def node_edge_loss(pos, edges, node_idx, cand, clearance):
    # one-sided hinge. Pay only when a node is closer to a non-incident edge than asked
    ends = pos[edges[cand]]
    dist = seg_dist(pos[node_idx][:, None, :], ends[..., 0, :], ends[..., 1, :])

    return (torch.relu(1.0 - dist / clearance) ** 2).mean()


def cross_pairs(pos, edges, n_pair, k_cand, gen, pool=None):
    # edge pairs near each other sharing no endpoint, the only ones that can cross
    use = torch.arange(edges.shape[0], device=pos.device) if pool is None else pool
    mid = pos[edges[use]].mean(1)
    loc = torch.randint(0, use.shape[0], (n_pair,), device=pos.device, generator=gen)
    dist = torch.cdist(mid[loc], mid)
    dist[torch.arange(n_pair, device=pos.device), loc] = float("inf")
    cand = dist.topk(min(k_cand, use.shape[0]), dim=1, largest=False).indices
    pick = torch.randint(0, cand.shape[1], (n_pair,), device=pos.device, generator=gen)
    sel = use[loc]
    other = use[cand[torch.arange(n_pair, device=pos.device), pick]]

    shares = (
        (edges[sel, 0] == edges[other, 0])
        | (edges[sel, 0] == edges[other, 1])
        | (edges[sel, 1] == edges[other, 0])
        | (edges[sel, 1] == edges[other, 1])
    )

    return sel[~shares], other[~shares]


def cross_loss(pos, edges, pair_a, pair_b, tau):
    # smooth crossing indicator, the product of the two orientation tests. `tau` must carry FOUR
    # powers of length: each ccw is twice a signed triangle area, so it already carries length^2,
    # and the indicator multiplies two of them.
    p1, p2 = pos[edges[pair_a, 0]], pos[edges[pair_a, 1]]
    p3, p4 = pos[edges[pair_b, 0]], pos[edges[pair_b, 1]]

    def ccw(a, b, c):
        return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
            c[:, 0] - a[:, 0]
        )

    d1, d2 = ccw(p1, p2, p3), ccw(p1, p2, p4)
    d3, d4 = ccw(p3, p4, p1), ccw(p3, p4, p2)

    return (torch.sigmoid(-d1 * d2 / tau) * torch.sigmoid(-d3 * d4 / tau)).mean()


def ne_loss(
    pos,
    edges,
    n_neg,
    gen,
    exag=1.0,
    w_rep=1.0,
    min_dist=0.0,
    ends=None,
    want_delta=False,
):
    # Neighbour embedding, LargeVis form. Attraction along edges against sampled negatives.
    # w_rep is LargeVis's gamma. Both terms are plain means, so w_rep=1 charges one negative pair
    # per edge. min_dist is UMAP's, a fraction of the drawing's RMS radius, and puts a dead zone in
    # the attraction so two adjacent nodes closer than it feel no pull.

    # `ends` is (edges[:, 0], edges[:, 1]) made contiguous once by the caller. Slicing a column
    # out of the (E, 2) block gives a strided index tensor and the gather is measurably slower.
    src_i, dst_i = (edges[:, 0], edges[:, 1]) if ends is None else ends
    delta = pos[src_i] - pos[dst_i]

    if min_dist > 0:
        # the norm is not differentiable where two nodes coincide, so keep the squared form
        # unless the dead zone is actually asked for
        scale = pos.pow(2).sum(1).mean().sqrt().detach()
        d_pos = (
            torch.relu((delta**2).sum(-1).clamp_min(1e-12).sqrt() - min_dist * scale)
            ** 2
        )
    else:
        d_pos = (delta**2).sum(-1)

    attract = exag * torch.log1p(d_pos).mean()

    n_nodes = pos.shape[0]
    src = torch.randint(0, n_nodes, (n_neg,), device=pos.device, generator=gen)
    dst = torch.randint(0, n_nodes, (n_neg,), device=pos.device, generator=gen)
    d_neg = ((pos[src] - pos[dst]) ** 2).sum(-1)
    repel = -torch.log(d_neg / (1.0 + d_neg) + 1e-6).mean()

    loss = attract + w_rep * repel

    return (loss, delta) if want_delta else loss


class FlingVis(_BaseFling):
    def __init__(
        self,
        n_components=2,
        iters=2000,
        aes_iters=None,
        lr=5e-3,
        lr2=0.5,
        w_ne=0.3,
        w_cross=0.3,
        code_dim=0,
        code_init=0.01,
        n_neg=4096,
        exaggeration=4.0,
        exag_frac=0.3,
        w_rep=1.0,
        min_dist=0.0,
        clear=0.6,
        m_node=512,
        k_cand=12,
        n_pair=2048,
        cand_every=2,
        cand_pool=8000,
        n_landmarks=64,
        alpha=0.05,
        diff_K=None,
        rp_dim=32,
        w0=2.0,
        act="gelu",
        width=128,
        depth=2,
        tb_dim=None,
        tb_alpha=0.03,
        optimizer="muon",
        muon_ratio=2.0,
        # tau = cross_tau * median_edge_length ** cross_tau_pow. The power is 4, see cross_loss
        cross_tau=0.25,
        cross_tau_pow=4,
        seed=0,
        device=None,
        # keep the fit's raw feature block so transform(local=True) can run. Set it before fit
        local_cache=False,
    ):
        super().__init__(
            n_components, w0, act, width, depth, tb_dim, tb_alpha, seed, local_cache
        )
        self.iters = iters
        self.aes_iters = aes_iters
        self.optimizer = optimizer
        self.muon_ratio = muon_ratio
        self.cross_tau = cross_tau
        self.cross_tau_pow = cross_tau_pow
        self.lr = lr
        self.lr2 = lr2
        self.w_ne = w_ne
        self.w_cross = w_cross
        self.code_dim = code_dim
        self.code_init = code_init
        self.n_neg = n_neg
        self.exaggeration = exaggeration
        self.exag_frac = exag_frac
        self.w_rep = w_rep
        self.min_dist = min_dist
        self.clear = clear
        self.m_node = m_node
        self.k_cand = k_cand
        self.n_pair = n_pair
        self.cand_every = cand_every
        self.cand_pool = cand_pool
        self.n_landmarks = n_landmarks
        self.alpha = alpha
        self.diff_K = diff_K
        self.rp_dim = rp_dim
        self.features = "diffusion"
        self.device = device

    def _input_block(self, graph, nodes):
        out = self._diffusion_block(graph, nodes)  # incl. the saturation fallback
        self._maybe_cache_block(out, nodes)

        return out

    def _features(self, graph, nodes):
        feat = _with_tie(
            (self._input_block(graph, nodes) - self._fmean) / self._fstd,
            self._tb_dim_,
            self.tb_alpha,
            self.seed,
        )

        if getattr(
            self, "_code_pad_", 0
        ):  # codes are transductive, unseen nodes get zeros
            feat = np.concatenate([feat, np.zeros((len(nodes), self._code_pad_))], 1)

        return feat

    def fit(self, graph, train_nodes=None):
        if train_nodes is not None:
            raise NotImplementedError(
                "FlingVis is fitted transductively, place unseen nodes with transform()"
            )

        # follow the torch default device, as the other estimators do, so a caller that has not
        # asked for the GPU does not end up with the net and its inputs on different devices
        dev = torch.device(self.device) if self.device else torch.empty(0).device
        nodes = list(graph.nodes())
        n_nodes = len(nodes)
        idx = {node: row for row, node in enumerate(nodes)}
        self.nodes_ = nodes

        self._land_ = self._farthest_pivots(
            graph, nodes, min(self.n_landmarks, n_nodes - 1)
        )[0]
        feat_in = self._input_block(graph, nodes)
        self._fmean = feat_in.mean(0)
        self._fstd = feat_in.std(0) + 1e-9
        self._tb_dim_ = self._resolve_tb(graph)

        feats = _with_tie(
            (feat_in - self._fmean) / self._fstd,
            self._tb_dim_,
            self.tb_alpha,
            self.seed,
        )

        edges = np.array([[idx[u], idx[v]] for u, v in graph.edges()], dtype=np.int64)
        feat = torch.tensor(np.asarray(feats, np.float32), device=dev)
        ed = torch.tensor(edges, device=dev)

        torch.manual_seed(self.seed)
        gen = torch.Generator(device=dev).manual_seed(self.seed)

        use_code = self.code_dim > 0
        codes = None

        if use_code:
            gen_c = torch.Generator(device="cpu").manual_seed(self.seed)
            codes = (
                (
                    self.code_init
                    * torch.randn(n_nodes, self.code_dim, generator=gen_c, device="cpu")
                )
                .to(dev)
                .requires_grad_()
            )

        self._code_pad_ = self.code_dim if use_code else 0

        inr = (
            Gabor(
                feat.shape[1] + self._code_pad_,
                self.n_components,
                width=self.width,
                depth=self.depth,
                w0=self.w0,
                act=self.act,
            )
            .float()
            .to(dev)
        )

        params = list(inr.parameters()) + ([codes] if codes is not None else [])

        def make_opt(lr):
            if self.optimizer == "muon":
                return _muon(
                    inr,
                    lr,
                    extra=[codes] if codes is not None else [],
                    ratio=self.muon_ratio,
                )

            return _adam(params, lr)

        def positions():
            return inr(feat if codes is None else torch.cat([feat, codes], 1))

        t0 = time.perf_counter()

        # phase 1, the neighbour embedding alone, with the attraction exaggerated early
        opt = make_opt(self.lr)

        for step in range(self.iters):
            strength = self.exaggeration if step < self.exag_frac * self.iters else 1.0

            loss = ne_loss(
                positions(),
                ed,
                self.n_neg,
                gen,
                exag=strength,
                w_rep=self.w_rep,
                min_dist=self.min_dist,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            for grp in opt.param_groups:
                grp["lr"] = self.lr * (
                    0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * step / max(1, self.iters)))
                )

        # phase 2, the same energy plus the terms, on a fresh schedule. A single ramp would decay
        # the learning rate away exactly as the terms fade in, which starves them.
        aes_iters = self.iters if self.aes_iters is None else self.aes_iters

        if aes_iters > 0 and (self.w_ne > 0 or self.w_cross > 0):
            opt = make_opt(self.lr * self.lr2)
            cache = {}

            for step in range(aes_iters):
                pos = positions()
                loss = ne_loss(
                    pos,
                    ed,
                    self.n_neg,
                    gen,
                    exag=1.0,
                    w_rep=self.w_rep,
                    min_dist=self.min_dist,
                )

                ramp = min(1.0, step / max(1, 0.2 * aes_iters))
                med = torch.median(
                    (pos[ed[:, 0]] - pos[ed[:, 1]]).norm(dim=-1)
                ).detach()

                if step % max(1, self.cand_every) == 0:
                    with torch.no_grad():
                        pool = None

                        if self.cand_pool and self.cand_pool < ed.shape[0]:
                            pool = torch.randperm(
                                ed.shape[0], device=dev, generator=gen
                            )[: self.cand_pool]

                        if self.w_ne > 0:
                            cache["nodes"] = torch.randint(
                                0,
                                n_nodes,
                                (min(self.m_node, n_nodes),),
                                device=dev,
                                generator=gen,
                            )
                            cache["cand"] = near_edges(
                                pos, ed, cache["nodes"], self.k_cand, pool
                            )

                        if self.w_cross > 0:
                            cache["pairs"] = cross_pairs(
                                pos, ed, self.n_pair, self.k_cand, gen, pool
                            )

                if self.w_ne > 0:
                    loss = loss + ramp * self.w_ne * node_edge_loss(
                        pos, ed, cache["nodes"], cache["cand"], self.clear * med
                    )

                if self.w_cross > 0:
                    pair_a, pair_b = cache["pairs"]
                    if pair_a.numel():
                        loss = loss + ramp * self.w_cross * cross_loss(
                            pos,
                            ed,
                            pair_a,
                            pair_b,
                            self.cross_tau * med**self.cross_tau_pow,
                        )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                for grp in opt.param_groups:
                    grp["lr"] = (self.lr * self.lr2) * (
                        0.05
                        + 0.95 * 0.5 * (1 + np.cos(np.pi * step / max(1, aes_iters)))
                    )

        with torch.no_grad():
            self.embedding_ = positions().detach().cpu().numpy().astype(np.float64)

        # the net stays on the device it was trained on, matching the other estimators, so that
        # transform() builds its input on the same default device
        self.net_ = inr.eval()
        self.codes_ = None if codes is None else codes.detach()
        self.edges_ = edges  # the field readings draw the graph over the sampled plane
        self.fit_seconds_ = time.perf_counter() - t0
        self.delta_ = None

        return self

    def fit_transform(self, graph):
        self.fit(graph)

        return self.embedding_
