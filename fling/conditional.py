import time

import numpy as np
import torch

from .estimators import _adam, _BaseFling
from .nets import Gabor
from .optim import build as _muon
from .vis import ne_loss


def pivot_stress(pos, dist, piv_rows, cols):
    # normalised at the closed-form optimal scale, or the blend weight could lower this term by
    # shrinking the drawing, which is what the neighbour-embedding end rewards
    tgt = dist[:, cols]
    emb = torch.cdist(pos, pos[piv_rows[cols]])
    ok = tgt > 0
    tgt = torch.where(ok, tgt, torch.ones_like(tgt))
    ratio = emb / tgt
    scale = (ratio * ok).sum() / ((ratio**2 * ok).sum() + 1e-12)
    err = (scale * emb - tgt) / tgt

    return (err**2 * ok).sum() / ok.sum()


class CondField(torch.nn.Module):
    def __init__(
        self,
        din,
        dout=2,
        width=128,
        depth=2,
        w0=2.0,
        act="gelu",
        conditioning="film",
        n_freq=4,
        hyper=64,
    ):
        super().__init__()

        if conditioning not in ("film", "concat"):
            raise ValueError("conditioning must be 'film' or 'concat'")

        self.conditioning = conditioning
        self.n_freq = n_freq
        self.depth = depth
        self.width = width
        self.lam = 0.0

        self.body = Gabor(
            din + (1 if conditioning == "concat" else 0),
            dout,
            width=width,
            depth=depth,
            w0=w0,
            act=act,
        )

        # alias so optim.build finds the output layer and keeps it off the orthogonaliser.
        # parameters() deduplicates by identity, so it is not counted twice
        self.out = self.body.out

        if conditioning == "film":
            self.film = torch.nn.Sequential(
                torch.nn.Linear(2 * n_freq, hyper),
                torch.nn.GELU(),
                torch.nn.Linear(hyper, 2 * width * depth),
            )

            # zeroed at init, so training starts from the unconditional field
            torch.nn.init.zeros_(self.film[-1].weight)
            torch.nn.init.zeros_(self.film[-1].bias)

    def embed(self, lam):  # Fourier features of lambda at octave frequencies
        freqs = 2.0 ** torch.arange(self.n_freq, device=lam.device, dtype=lam.dtype)

        return torch.cat(
            [torch.sin(np.pi * freqs * lam), torch.cos(np.pi * freqs * lam)], -1
        )

    def forward(self, feat, lam=None):
        lam = self.lam if lam is None else lam
        lam_t = torch.as_tensor(lam, device=feat.device, dtype=feat.dtype).reshape(1)

        if self.conditioning == "concat":
            col = lam_t.expand(feat.shape[0], 1)

            return self.body(torch.cat([feat, col], -1))

        mod = self.film(self.embed(lam_t)).reshape(self.depth, 2, self.width)

        for layer, lin in enumerate(self.body.hidden):
            gain, shift = mod[layer, 0], mod[layer, 1]
            feat = self.body.g(lin(feat) * (1 + gain) + shift)

        return self.body.out(feat)


class FlingBlend(_BaseFling):
    def __init__(
        self,
        n_components=2,
        iters=1500,
        lr=5e-3,
        npiv=200,
        pcols=32,
        n_neg=4096,
        n_lam=4,
        lam_prior="beta",
        lam_default=0.5,
        lam=None,
        conditioning="film",
        n_freq=4,
        hyper=64,
        exaggeration=1.0,
        w_rep=1.0,
        min_dist=0.0,
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
        optimizer="adam",
        muon_ratio=2.0,
        cal_iters=None,
        scales=None,
        anneal=0.0,
        seed=0,
        device=None,
        local_cache=False,
    ):
        super().__init__(
            n_components, w0, act, width, depth, tb_dim, tb_alpha, seed, local_cache
        )

        self.iters = iters
        self.lr = lr
        self.npiv = npiv
        self.pcols = pcols
        self.n_neg = n_neg

        # n_lam samples of lambda per step, each contributing its own blended loss, so a step is
        # n_lam times the work of an unconditional one
        self.n_lam = n_lam
        self.lam_prior = lam_prior
        self.lam_default = lam_default

        # lam=None conditions the field on the weight. A float pins it, training an ordinary
        # unconditional field on that one blend
        self.lam = lam

        self.conditioning = conditioning
        self.n_freq = n_freq
        self.hyper = hyper

        self.exaggeration = exaggeration
        self.w_rep = w_rep
        self.min_dist = min_dist

        self.features = "diffusion"
        self.n_landmarks = n_landmarks
        self.alpha = alpha
        self.diff_K = diff_K
        self.rp_dim = rp_dim

        self.optimizer = optimizer
        self.muon_ratio = muon_ratio

        # cal_iters=None calibrates at the full `iters`. scales=(stress, ne) skips calibration
        self.cal_iters = cal_iters
        self.scales = scales
        self.anneal = anneal
        self.device = device

    def _sample_lams(self, gen, dev):
        unif = torch.rand(self.n_lam, device=dev, generator=gen)

        if self.lam_prior == "uniform":
            return unif
        if self.lam_prior != "beta":
            raise ValueError("lam_prior must be 'beta' or 'uniform'")

        return torch.sin(0.5 * np.pi * unif) ** 2

    def _make_net(self, din, dev):
        return (
            CondField(
                din,
                self.n_components,
                width=self.width,
                depth=self.depth,
                w0=self.w0,
                act=self.act,
                conditioning=self.conditioning,
                n_freq=self.n_freq,
                hyper=self.hyper,
            )
            .float()
            .to(dev)
        )

    def _make_opt(self, net, lr):
        if self.optimizer == "muon":
            return _muon(net, lr, ratio=self.muon_ratio)

        return _adam(list(net.parameters()), lr)

    def _train(self, feat, dist, piv_rows, edges, gen, dev, iters, scales, lam=None):
        # lam=None conditions the field. A float pins it to a specialist
        net = self._make_net(feat.shape[1], dev)
        opt = self._make_opt(net, self.lr)
        n_piv = dist.shape[1]

        for step in range(iters):
            cols = torch.randint(
                0, n_piv, (min(self.pcols, n_piv),), device=dev, generator=gen
            )
            lams = (
                self._sample_lams(gen, dev)
                if lam is None
                else torch.full((1,), float(lam), device=dev)
            )

            opt.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=dev)

            for lval in lams:
                pos = net(feat, lval)
                l_st = pivot_stress(pos, dist, piv_rows, cols) / scales[0]
                l_ne = (
                    ne_loss(
                        pos,
                        edges,
                        self.n_neg,
                        gen,
                        exag=self.exaggeration,
                        w_rep=self.w_rep,
                        min_dist=self.min_dist,
                    )
                    / scales[1]
                )

                loss = loss + (1 - lval) * l_st + lval * l_ne

            (loss / len(lams)).backward()
            opt.step()

            if self.anneal > 0:
                fdec = self.anneal + (1 - self.anneal) * 0.5 * (
                    1 + np.cos(np.pi * step / max(1, iters))
                )

                for grp in opt.param_groups:
                    grp["lr"] = self.lr * fdec

        return net.eval()

    def _raw_terms(self, net, feat, dist, piv_rows, edges, lam, dev):
        # unnormalised value of each energy, on all pivot columns and many negatives
        gen = torch.Generator(device=dev).manual_seed(self.seed)
        cols = torch.arange(dist.shape[1], device=dev)

        with torch.no_grad():
            pos = net(feat, lam)

            return (
                float(pivot_stress(pos, dist, piv_rows, cols)),
                float(
                    ne_loss(
                        pos,
                        edges,
                        20000,
                        gen,
                        exag=self.exaggeration,
                        w_rep=self.w_rep,
                        min_dist=self.min_dist,
                    )
                ),
            )

    def fit(self, graph, train_nodes=None):
        if train_nodes is not None:
            raise NotImplementedError(
                "FlingBlend is fitted transductively, place unseen nodes with transform()"
            )

        dev = torch.device(self.device) if self.device else torch.empty(0).device
        nodes, n_nodes, idx, _, _, _ = self._train_split(graph, train_nodes)

        self._land_ = self._farthest_pivots(
            graph, nodes, min(self.n_landmarks, n_nodes - 1)
        )[0]

        feat_in = self._input_block(graph, nodes)
        self._fmean = feat_in.mean(0)
        self._fstd = feat_in.std(0) + 1e-9
        self._tb_dim_ = self._resolve_tb(graph)
        self._code_pad_ = 0
        feats = self._features(graph, nodes)

        # stress targets are hop distances to a farthest-first pivot set, as in FlingStress
        pivots, _ = self._farthest_pivots(graph, nodes, min(self.npiv, n_nodes - 1))
        dist_np = self._pivot_columns(graph, pivots, nodes).astype(np.float32)

        feat = torch.tensor(np.asarray(feats, np.float32), device=dev)
        dist = torch.tensor(dist_np, device=dev)
        piv_rows = torch.tensor(np.array([idx[p] for p in pivots]), device=dev)
        edges = torch.tensor(self.edges_, device=dev)

        torch.manual_seed(self.seed)
        t0 = time.perf_counter()

        # calibration. each energy in units of what a field trained on it alone reaches, so the blend
        # weight means the same thing on graphs whose two energies differ in scale. A field pinned
        # to an endpoint carries no weight on the other term, so it needs no calibration.
        pinned_end = self.lam is not None and float(self.lam) in (0.0, 1.0)

        if self.scales is None and pinned_end:
            self.scales_ = (1.0, 1.0)
        elif self.scales is None:
            cal_iters = self.iters if self.cal_iters is None else self.cal_iters
            raw = {}

            for lam, key in ((0.0, "stress"), (1.0, "ne")):
                gen = torch.Generator(device=dev).manual_seed(self.seed)
                spec = self._train(
                    feat,
                    dist,
                    piv_rows,
                    edges,
                    gen,
                    dev,
                    cal_iters,
                    (1.0, 1.0),
                    lam=lam,
                )
                raw[key] = self._raw_terms(spec, feat, dist, piv_rows, edges, lam, dev)

            self.scales_ = (abs(raw["stress"][0]) + 1e-9, abs(raw["ne"][1]) + 1e-9)
        else:
            self.scales_ = tuple(self.scales)

        gen = torch.Generator(device=dev).manual_seed(self.seed)
        net = self._train(
            feat,
            dist,
            piv_rows,
            edges,
            gen,
            dev,
            self.iters,
            self.scales_,
            lam=self.lam,
        )

        self.net_ = net
        self._feat_t_ = feat
        self.pivots_ = pivots
        self.fit_seconds_ = time.perf_counter() - t0
        self.embedding_ = self.draw(self.lam_default if self.lam is None else self.lam)

        return self

    def draw(self, lam):
        # also sets the net's default lambda, so a following transform() places unseen nodes into
        # this member of the family
        if not hasattr(self, "net_"):
            raise RuntimeError("call fit before draw")

        if self.lam is not None and float(lam) != float(self.lam):
            raise ValueError(
                f"this field was pinned to lam={self.lam} and was never trained at "
                f"lam={lam}, fit with lam=None for the family"
            )

        self.net_.lam = float(lam)

        with torch.no_grad():
            return (
                self.net_(self._feat_t_, float(lam))
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

    def family(self, lams=(0.0, 0.25, 0.5, 0.75, 1.0)):  # (len(lams), N, n_components)
        return np.stack([self.draw(lam) for lam in lams])

    def fit_transform(self, graph, lam=None):
        self.fit(graph)

        return self.embedding_ if lam is None else self.draw(lam)
