import numpy as np
import torch


class Gabor(torch.nn.Module):
    def __init__(
        self,
        din,
        dout,
        width=64,
        depth=2,
        sigma=0.5,
        w0=8.0,
        zero_last=False,
        act="gabor",
        k_bias=0.707,
    ):
        super().__init__()

        self.w0, self.sigma, self.act = w0, sigma, act
        dims = [din] + [width] * depth
        self.hidden = torch.nn.ModuleList(
            torch.nn.Linear(n_in, n_out) for n_in, n_out in zip(dims[:-1], dims[1:])
        )
        self.out = torch.nn.Linear(width, dout)

        if zero_last:
            torch.nn.init.zeros_(self.out.weight)
            torch.nn.init.zeros_(self.out.bias)
        if act in ("siren", "finer"):  # SIREN init: first layer U(-1/din,1/din),
            with torch.no_grad():  # rest U(+-sqrt(6/fan_in)/w0)
                self.hidden[0].weight.uniform_(-1 / din, 1 / din)

                for lin in list(self.hidden)[1:]:
                    bound = (6 / lin.in_features) ** 0.5 / w0
                    lin.weight.uniform_(-bound, bound)

        if act == "finer":  # FINER's knob: wide first-layer bias init
            with torch.no_grad():
                self.hidden[0].bias.uniform_(-k_bias, k_bias)

        self.double()

    def g(self, acts):
        if self.act == "siren":
            return torch.sin(self.w0 * acts)

        if self.act == "finer":
            return torch.sin(self.w0 * (acts.abs() + 1.0) * acts)

        if self.act == "relu":
            return torch.relu(acts)

        if self.act == "gelu":
            return torch.nn.functional.gelu(acts)

        return torch.exp(-(acts**2) / (2 * self.sigma**2)) * torch.cos(self.w0 * acts)

    def forward(self, feat):
        for lin in self.hidden:
            feat = self.g(lin(feat))

        return self.out(feat)


def tie_features(n_nodes, dim=2, alpha=0.03, seed=0):
    if dim <= 0:
        return np.zeros((n_nodes, 0))

    return alpha * np.random.default_rng(seed).standard_normal((n_nodes, dim))


def _with_tie(feat, tb_dim, tb_alpha, seed=0):
    feat = np.asarray(feat, float)

    if tb_dim <= 0:
        return feat

    return np.concatenate(
        [feat, tie_features(feat.shape[0], tb_dim, tb_alpha, seed)], 1
    )


def _has_twins(graph):
    open_, closed = set(), set()

    for node in graph:
        nbrs = frozenset(graph[node])

        if nbrs in open_ or (nbrs | {node}) in closed:
            return True

        open_.add(nbrs)
        closed.add(nbrs | {node})

    return False
