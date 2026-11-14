"""HealthRouter: a per-sample router that selects and weights a collection of models.

The router sees only cheap shared features z(x) (a small frozen encoder) and outputs one
score per candidate model. It is trained on out-of-fold candidate outputs with two terms:

  * the negative log-likelihood of the router-weighted mixture of candidate predictions, and
  * a per-candidate utility term, KL(q_i || softmax(s_i)) with q_i = softmax(log p_c(y_i|x_i) / tau),
    which tells the router directly which candidates assign high probability to the truth
    on inputs like x_i.

At test time it keeps, per sample, the smallest set of top-weighted candidates whose weight
mass reaches rho (at most k_max, optionally penalising cost), and fuses only those. The router
never receives a task or domain identifier; it only learns which models do well where.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

EPS = 1e-4
torch.set_num_threads(2)


class Gate(nn.Module):
    def __init__(self, d_in: int, n_out: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, n_out))

    def forward(self, z):
        return self.net(z)


class Standardizer:
    def fit(self, z):
        self.mu, self.sd = z.mean(0), z.std(0) + 1e-6
        return self

    def __call__(self, z):
        return ((z - self.mu) / self.sd).astype(np.float32)


def train_gate(Z, U, alpha=1.0, tau=1.0, hidden=256, epochs=300, lr=1e-3, wd=1e-4,
               dropout=0.1, seed=0, topk_train=None, utility_only=False, mse=False):
    """Fit a gate on router features Z (n, d) and true-label candidate probabilities U (n, C).

    alpha=0, topk_train=None  -> soft mixture-of-experts gate (end to end on the fused NLL)
    topk_train=k              -> sparse top-k MoE gate (fused NLL over the top-k only)
    utility_only=True         -> only the per-candidate utility term
    mse=True                  -> regress log-utility directly (predictive-router baseline)
    """
    torch.manual_seed(seed)
    z = torch.from_numpy(Z)
    u = torch.from_numpy(np.clip(U, EPS, 1.0).astype(np.float32))
    logu = u.log()
    q = torch.softmax(logu / tau, dim=1)
    gate = Gate(Z.shape[1], U.shape[1], hidden, dropout)
    opt = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        gate.train()
        s = gate(z)
        if mse:
            loss = ((s - logu) ** 2).mean()
        else:
            if topk_train:
                top = s.topk(topk_train, dim=1).indices
                mask = torch.full_like(s, float("-inf")).scatter(1, top, 0.0)
                w = torch.softmax(s + mask, dim=1)
            else:
                w = torch.softmax(s, dim=1)
            fused = -(w * u).sum(1).clamp_min(EPS).log().mean()
            util = (q * (q.clamp_min(1e-12).log() - torch.log_softmax(s, 1))).sum(1).mean()
            loss = util if utility_only else fused + alpha * util
        opt.zero_grad()
        loss.backward()
        opt.step()
    gate.eval()
    return gate


def scores(gate, Z) -> np.ndarray:
    with torch.no_grad():
        return gate(torch.from_numpy(Z)).numpy()


def select(S: np.ndarray, T: float, rho: float, k_max: int, cost=None, lam: float = 0.0):
    """Adaptive-k selection. Returns renormalised weights (n, C) with zeros off the set."""
    logits = S / T
    logits = logits - logits.max(1, keepdims=True)
    pi = np.exp(logits)
    pi /= pi.sum(1, keepdims=True)
    key = np.log(pi + 1e-12) - (lam * cost[None, :] if cost is not None else 0.0)
    order = np.argsort(-key, axis=1)
    sorted_pi = np.take_along_axis(pi, order, 1)
    cum = np.cumsum(sorted_pi, 1)
    # smallest prefix reaching rho: count of prefixes with cum < rho, plus one
    k = np.minimum((cum < rho - 1e-9).sum(1) + 1, k_max)
    keep = np.arange(S.shape[1])[None, :] < k[:, None]
    W = np.zeros_like(pi)
    np.put_along_axis(W, order, np.where(keep, sorted_pi, 0.0), 1)
    W /= W.sum(1, keepdims=True)
    return W


def fuse(P: np.ndarray, W: np.ndarray) -> np.ndarray:
    """P (n, C, K) candidate probabilities, W (n, C) weights -> (n, K)."""
    return np.einsum("nck,nc->nk", P, W)


def fuse_geometric(P: np.ndarray, W: np.ndarray) -> np.ndarray:
    logp = np.log(np.clip(P, EPS, 1.0))
    out = np.exp(np.einsum("nck,nc->nk", logp, W))
    return out / out.sum(1, keepdims=True)


# ----------------------------------------------------------------------------- collection-level routing
def collection_library(n_cand: int, max_size: int) -> np.ndarray:
    """Binary membership matrix (n_collections, n_cand) of every subset of size 1..max_size."""
    from itertools import combinations
    rows = []
    for k in range(1, max_size + 1):
        for comb in combinations(range(n_cand), k):
            r = np.zeros(n_cand, np.float32)
            r[list(comb)] = 1.0
            rows.append(r)
    return np.stack(rows)


def collection_utility(U: np.ndarray, lib: np.ndarray) -> np.ndarray:
    """log of the uniform-average true-label probability of each collection, (n, n_collections)."""
    avg = (U @ lib.T) / lib.sum(1)[None, :]
    return np.log(np.clip(avg, EPS, 1.0)).astype(np.float32)


def train_collection_gate(Z, U, lib, tau=1.0, beta=1.0, hidden=256, epochs=300, lr=1e-3,
                          wd=1e-4, dropout=0.1, seed=0):
    """Gate over collections: listwise KL to softmax(u_S / tau) plus beta * fused NLL of the
    gate-weighted mixture of collections (each collection averages its members uniformly)."""
    torch.manual_seed(seed)
    z = torch.from_numpy(Z)
    u = torch.from_numpy(np.clip(U, EPS, 1.0).astype(np.float32))
    L = torch.from_numpy(lib)
    us = torch.log(((u @ L.T) / L.sum(1)[None, :]).clamp_min(EPS))
    q = torch.softmax(us / tau, 1)
    gate = Gate(Z.shape[1], lib.shape[0], hidden, dropout)
    opt = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        gate.train()
        s = gate(z)
        logw = torch.log_softmax(s, 1)
        util = (q * (q.clamp_min(1e-12).log() - logw)).sum(1).mean()
        fused = -torch.logsumexp(logw + us, 1).mean()
        loss = util + beta * fused
        opt.zero_grad()
        loss.backward()
        opt.step()
    gate.eval()
    return gate


def select_collection(S: np.ndarray, lib: np.ndarray, T: float, m: int, cost=None,
                      lam: float = 0.0) -> np.ndarray:
    """Per sample, average the top-m scoring collections (weights softmax(S/T) over them);
    returns candidate weights (n, C)."""
    key = S / T
    if cost is not None and lam > 0:
        key = key - lam * (lib @ cost)[None, :]
    top = np.argsort(-key, 1)[:, :m]
    sc = np.take_along_axis(key, top, 1)
    w = np.exp(sc - sc.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    member = lib / lib.sum(1, keepdims=True)
    return np.einsum("nm,nmc->nc", w, member[top])
