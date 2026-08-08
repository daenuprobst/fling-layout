import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fling import FlingStress
from fling.metrics import normalized_stress
from fling.nets import _with_tie

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HELD_FRAC = 0.2  # share of the train nodes kept out of the energy, to select budgets on


def make_est(seed, iters, lr, npiv, n_land):
    return FlingStress(
        n_components=2,
        iters=iters,
        lr=lr,
        npiv=npiv,
        fcols=n_land,
        features="hops",
        seed=seed,
    )


def stress_rows(pos, tgt, piv_rows, cols):
    emb = torch.cdist(pos, pos[piv_rows[cols]])
    tt = tgt[:, cols]
    ok = tt > 0
    tt = torch.where(ok, tt, torch.ones_like(tt))
    ratio = emb / tt
    scale = (ratio * ok).sum() / ((ratio**2 * ok).sum() + 1e-12)

    return (((scale * emb - tt) / tt) ** 2 * ok).sum() / ok.sum()


def held_stress(pos, rows, tgt, piv):
    # the held-out pivots are addressed in the node set while the training rows are a subset,
    # so this cannot share indexing with stress_rows
    emb = torch.cdist(pos[rows], pos[piv])
    tt = tgt[rows]
    ok = tt > 0
    tt = torch.where(ok, tt, torch.ones_like(tt))
    ratio = emb / tt
    scale = (ratio * ok).sum() / ((ratio**2 * ok).sum() + 1e-12)

    return float((((scale * emb - tt) / tt) ** 2 * ok).sum() / ok.sum())


def split(graph, m_train, npiv, seed):
    # mirrors FlingStress.fit: pivots are default_rng(seed).choice over the train list, so the
    # complement is known without touching the estimator's internals
    nodes = list(graph.nodes())
    rng = np.random.default_rng(1000 + seed)
    tix = np.sort(rng.choice(len(nodes), min(m_train, len(nodes)), replace=False))
    train = [nodes[i] for i in tix]

    npiv_eff = int(min(npiv, len(train) - 1, round((1 - HELD_FRAC) * len(train))))
    chosen = np.random.default_rng(seed).choice(len(train), npiv_eff, replace=False)
    held = [train[i] for i in sorted(set(range(len(train))) - set(chosen.tolist()))]

    # selection has to see nodes OUTSIDE the sample: holding out columns alone lets a narrow
    # kernel place its anchors perfectly and everything else arbitrarily and still score best
    rest = np.setdiff1d(np.arange(len(nodes)), tix)
    oos = (
        np.sort(rng.choice(rest, min(len(rest), 1000), replace=False))
        if len(rest)
        else tix
    )

    return train, tix, held, npiv_eff, oos


def blocks(est, graph, held_pivots):
    # rebuilt the way FlingStress.fit builds it: with features='hops' the public _features path
    # recomputes diffusion potentials and returns a different width entirely
    nodes = list(graph.nodes())
    idx = {node: row for row, node in enumerate(nodes)}
    dist_tr = np.asarray(est._pivot_columns(graph, est.pivots_, nodes), np.float32)
    feat_in = dist_tr[:, : min(est.fcols, dist_tr.shape[1])]
    feat = np.asarray(
        _with_tie(
            (feat_in - est._fmean) / est._fstd, est._tb_dim_, est.tb_alpha, est.seed
        ),
        np.float32,
    )
    dist_ho = np.asarray(est._pivot_columns(graph, held_pivots, nodes), np.float32)
    piv_rows = np.array([idx[p] for p in est.pivots_])
    ho_rows = np.array([idx[p] for p in held_pivots])

    return feat, dist_tr, dist_ho, piv_rows, ho_rows


def field_curve(graph, train, held_pivots, oos, dmat, seed, lr, iters, npiv, n_land):
    est = make_est(seed, iters, lr, npiv, n_land)
    est.fit(graph, train_nodes=train)
    pos = est.embedding_
    nodes = list(graph.nodes())
    idx = {node: row for row, node in enumerate(nodes)}
    ho_rows = np.array([idx[p] for p in held_pivots])
    dist_ho = np.asarray(est._pivot_columns(graph, held_pivots, nodes), np.float32)
    tix = np.array([idx[t] for t in train])

    pt = torch.as_tensor(pos, device=DEV, dtype=torch.float32)
    held = held_stress(
        pt,
        torch.as_tensor(oos, device=DEV),
        torch.as_tensor(dist_ho, device=DEV),
        torch.as_tensor(ho_rows, device=DEV),
    )

    return held, float(normalized_stress(pos, dmat)), est, tix
