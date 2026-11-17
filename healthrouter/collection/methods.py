"""HealthRouter and every baseline under identical candidates, features and splits.

Each method family returns a list of (config, outputs) where outputs[split][task] = (P, M):
P (n, K) fused probabilities and M (n, C) boolean mask of the candidates that must run.
All families train only on the meta-training data (out-of-fold candidate predictions on the
inner-train part); configurations are compared on validation by the caller.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import metrics, router
from .router import EPS


# ----------------------------------------------------------------------------- helpers
def true_prob(P, y):
    return P[np.arange(len(y)), :, y]


def pooled(run, split, key="Z"):
    return np.concatenate([run[t][f"{key}_{split}"] for t in run["tasks"]])


def split_by_task(run, split, arr):
    out, i = {}, 0
    for t in run["tasks"]:
        n = len(run[t][f"y_{split}"])
        out[t] = arr[i:i + n]
        i += n
    return out


def from_weights(run, splits, W_by_split, geometric=False):
    out = {}
    for sp in splits:
        Wt = split_by_task(run, sp, W_by_split[sp])
        out[sp] = {}
        for t in run["tasks"]:
            P = run[t][f"P_{sp}"]
            fused = router.fuse_geometric(P, Wt[t]) if geometric else router.fuse(P, Wt[t])
            out[sp][t] = (fused, Wt[t] > 0)
    return out


def topk_uniform(S, k):
    W = np.zeros_like(S, dtype=float)
    idx = np.argsort(-S, 1)[:, :k]
    np.put_along_axis(W, idx, 1.0 / k, 1)
    return W


def per_task_const(run, splits, weights_by_task):
    """A fixed weight vector per task (task-level methods)."""
    out = {}
    for sp in splits:
        out[sp] = {}
        for t in run["tasks"]:
            P = run[t][f"P_{sp}"]
            w = np.asarray(weights_by_task[t], float)
            W = np.repeat(w[None] / w.sum(), len(P), 0)
            out[sp][t] = (router.fuse(P, W), W > 0)
    return out


def val_metric_per_candidate(run, t, source="va"):
    P, y = run[t][f"P_{source}"], run[t][f"y_{source}"]
    return np.array([metrics.primary(t, y, P[:, c]) for c in range(P.shape[1])])


# ----------------------------------------------------------------------------- ours
def healthrouter(run, splits, grid, Zkey="Z", cost=None, lam_grid=(0.0,), fusion="arith",
                 fixed_k=None, uniform_weights=False, task_onehot=False, onehot_only=False):
    Ztr = pooled(run, "tr", Zkey)
    std = router.Standardizer().fit(Ztr)

    def feats(sp):
        z = std(pooled(run, sp, Zkey))
        if task_onehot or onehot_only:
            oh = np.concatenate([np.repeat(np.eye(len(run["tasks"]))[[i]], len(run[t][f"y_{sp}"]), 0)
                                 for i, t in enumerate(run["tasks"])]).astype(np.float32)
            z = oh if onehot_only else np.concatenate([z, oh], 1)
        return z

    U = np.concatenate([true_prob(run[t]["P_tr"], run[t]["y_tr"]) for t in run["tasks"]])
    results = []
    trained = {}
    for alpha in grid["alpha"]:
        for tau in (grid["tau_u"] if alpha > 0 else [1.0]):
            gate = router.train_gate(feats("tr"), U, alpha=alpha, tau=tau, hidden=grid["hidden"],
                                     epochs=grid["epochs"], lr=grid["lr"], wd=grid["weight_decay"],
                                     dropout=grid["dropout"], seed=run["seed"])
            trained[(alpha, tau)] = {sp: router.scores(gate, feats(sp)) for sp in splits}
    ks = [fixed_k] if fixed_k else grid["k_max"]
    rhos = [1.0] if fixed_k else grid["rho"]
    for (alpha, tau), S in trained.items():
        for T in grid["T"]:
            for rho in rhos:
                for k in ks:
                    for lam in lam_grid:
                        W = {}
                        for sp in splits:
                            w = router.select(S[sp], T, rho, k, cost, lam)
                            if uniform_weights:
                                w = (w > 0) / (w > 0).sum(1, keepdims=True)
                            W[sp] = w
                        cfg = {"alpha": alpha, "tau_u": tau, "T": T, "rho": rho, "k_max": k,
                               "lambda": lam}
                        results.append((cfg, from_weights(run, splits, W, fusion == "geo")))
    return results


# ----------------------------------------------------------------------------- task-level
def best_single_per_task(run, splits, source="va"):
    w = {}
    for t in run["tasks"]:
        v = val_metric_per_candidate(run, t, source)
        w[t] = np.eye(len(v))[int(np.nanargmax(v))]
    return [({}, per_task_const(run, splits, w))]


def best_single_overall(run, splits):
    v = np.mean([val_metric_per_candidate(run, t) for t in run["tasks"]], 0)
    c = int(np.nanargmax(v))
    return [({"candidate": c}, per_task_const(run, splits, {t: np.eye(len(v))[c] for t in run["tasks"]}))]


def uniform(run, splits):
    C = run["n_cand"]
    return [({}, per_task_const(run, splits, {t: np.ones(C) for t in run["tasks"]}))]


def stacking(run, splits):
    res = []
    for Creg in (0.01, 0.1, 1.0, 10.0):
        out = {sp: {} for sp in splits}
        for t in run["tasks"]:
            f = lambda P: np.log(np.clip(P, EPS, 1)).reshape(len(P), -1)
            sc = StandardScaler().fit(f(run[t]["P_tr"]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = LogisticRegression(C=Creg, max_iter=3000).fit(sc.transform(f(run[t]["P_tr"])),
                                                                  run[t]["y_tr"])
            K = run[t]["P_tr"].shape[2]
            for sp in splits:
                P = np.zeros((len(run[t][f"P_{sp}"]), K))
                P[:, m.classes_] = m.predict_proba(sc.transform(f(run[t][f"P_{sp}"])))
                out[sp][t] = (P, np.ones((len(P), run["n_cand"]), bool))
        res.append(({"C": Creg}, out))
    return res


def greedy_ensemble(run, splits, steps=20):
    w = {}
    for t in run["tasks"]:
        P, y = run[t]["P_tr"], run[t]["y_tr"]
        counts = np.zeros(P.shape[1])
        cur = None
        for _ in range(steps):
            best = None
            for c in range(P.shape[1]):
                cand = P[:, c] if cur is None else (cur * counts.sum() + P[:, c]) / (counts.sum() + 1)
                loss = metrics.nll(y, cand)
                if best is None or loss < best[0]:
                    best = (loss, c, cand)
            counts[best[1]] += 1
            cur = best[2]
        w[t] = counts
    return [({"steps": steps}, per_task_const(run, splits, w))]


def _lane(run, splits, task_pred_by_split, k, source="va"):
    """Domain router: the predicted task's top-k candidates (ranked by the task metric on
    `source`: validation, or the meta-training out-of-fold predictions), uniform average."""
    rank = {t: np.argsort(-np.nan_to_num(val_metric_per_candidate(run, t, source), nan=-1))
            for t in run["tasks"]}
    out = {}
    for sp in splits:
        pred = split_by_task(run, sp, task_pred_by_split[sp])
        out[sp] = {}
        for t in run["tasks"]:
            P = run[t][f"P_{sp}"]
            W = np.zeros(P.shape[:2])
            for i, tp in enumerate(pred[t]):
                W[i, rank[run["tasks"][tp]][:k]] = 1.0 / k
            out[sp][t] = (router.fuse(P, W), W > 0)
    return out


def modality_router_supervised(run, splits, source="va"):
    tid = lambda sp: np.concatenate([np.full(len(run[t][f"y_{sp}"]), i)
                                     for i, t in enumerate(run["tasks"])])
    sc = StandardScaler().fit(pooled(run, "tr"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(pooled(run, "tr")), tid("tr"))
    pred = {sp: g.predict(sc.transform(pooled(run, sp))) for sp in splits}
    return [({"k": k}, _lane(run, splits, pred, k, source)) for k in (1, 2, 3, 5)]


def modality_router_zeroshot(run, splits, text_emb, source="va"):
    """text_emb: (n_tasks, d) normalised BiomedCLIP prompt embeddings, same task order."""
    pred = {}
    for sp in splits:
        z = pooled(run, sp, "B")
        z = z / np.linalg.norm(z, axis=1, keepdims=True)
        pred[sp] = (z @ text_emb.T).argmax(1)
    return [({"k": k}, _lane(run, splits, pred, k, source)) for k in (1, 2, 3, 5)]


# ----------------------------------------------------------------------------- MoE gates
def soft_moe(run, splits, grid):
    Ztr = pooled(run, "tr")
    std = router.Standardizer().fit(Ztr)
    U = np.concatenate([true_prob(run[t]["P_tr"], run[t]["y_tr"]) for t in run["tasks"]])
    gate = router.train_gate(std(Ztr), U, alpha=0.0, hidden=grid["hidden"], epochs=grid["epochs"],
                             lr=grid["lr"], wd=grid["weight_decay"], dropout=grid["dropout"],
                             seed=run["seed"])
    res = []
    for T in grid["T"]:
        W = {}
        for sp in splits:
            s = router.scores(gate, std(pooled(run, sp))) / T
            e = np.exp(s - s.max(1, keepdims=True))
            W[sp] = e / e.sum(1, keepdims=True)
        res.append(({"T": T}, from_weights(run, splits, W)))
    return res


def sparse_moe(run, splits, grid):
    Ztr = pooled(run, "tr")
    std = router.Standardizer().fit(Ztr)
    U = np.concatenate([true_prob(run[t]["P_tr"], run[t]["y_tr"]) for t in run["tasks"]])
    res = []
    for k in (1, 2, 3):
        gate = router.train_gate(std(Ztr), U, alpha=0.0, hidden=grid["hidden"],
                                 epochs=grid["epochs"], lr=grid["lr"], wd=grid["weight_decay"],
                                 dropout=grid["dropout"], seed=run["seed"], topk_train=k)
        W = {}
        for sp in splits:
            s = router.scores(gate, std(pooled(run, sp)))
            top = np.argsort(-s, 1)[:, :k]
            m = np.full_like(s, -np.inf)
            np.put_along_axis(m, top, 0.0, 1)
            e = np.exp(s + m - (s + m).max(1, keepdims=True))
            W[sp] = e / e.sum(1, keepdims=True)
        res.append(({"k": k}, from_weights(run, splits, W)))
    return res


# ----------------------------------------------------------------------------- neighbour / predictive routers
def knn_router(run, splits):
    Ztr = pooled(run, "tr")
    std = router.Standardizer().fit(Ztr)
    A = std(Ztr)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    L = np.log(np.clip(np.concatenate([true_prob(run[t]["P_tr"], run[t]["y_tr"])
                                       for t in run["tasks"]]), EPS, 1))
    res = []
    sims = {}
    for sp in splits:
        B = std(pooled(run, sp))
        B /= np.linalg.norm(B, axis=1, keepdims=True)
        sims[sp] = np.argsort(-(B @ A.T), 1)[:, :50]
    for nn in (5, 10, 25, 50):
        for k in (1, 2, 3, 5):
            W = {sp: topk_uniform(L[sims[sp][:, :nn]].mean(1), k) for sp in splits}
            res.append(({"neighbours": nn, "k": k}, from_weights(run, splits, W)))
    return res


def routerbench_mlp(run, splits, grid):
    Ztr = pooled(run, "tr")
    std = router.Standardizer().fit(Ztr)
    U = np.concatenate([true_prob(run[t]["P_tr"], run[t]["y_tr"]) for t in run["tasks"]])
    gate = router.train_gate(std(Ztr), U, hidden=grid["hidden"], epochs=grid["epochs"],
                             lr=grid["lr"], wd=grid["weight_decay"], dropout=grid["dropout"],
                             seed=run["seed"], mse=True)
    S = {sp: router.scores(gate, std(pooled(run, sp))) for sp in splits}
    return [({"k": k}, from_weights(run, splits, {sp: topk_uniform(S[sp], k) for sp in splits}))
            for k in (1, 2, 3, 5)]


def routellm_mf(run, splits, grid, rank=64):
    import torch
    torch.manual_seed(run["seed"])
    Ztr = pooled(run, "tr")
    std = router.Standardizer().fit(Ztr)
    z = torch.from_numpy(std(Ztr))
    L = torch.from_numpy(np.log(np.clip(np.concatenate(
        [true_prob(run[t]["P_tr"], run[t]["y_tr"]) for t in run["tasks"]]), EPS, 1)).astype(np.float32))
    C = L.shape[1]
    proj = torch.nn.Linear(z.shape[1], rank, bias=False)
    emb = torch.nn.Parameter(torch.randn(C, rank) * 0.1)
    opt = torch.optim.AdamW(list(proj.parameters()) + [emb], lr=grid["lr"], weight_decay=grid["weight_decay"])
    diff = L[:, :, None] - L[:, None, :]           # (n, C, C) utility differences
    target = (diff > 0).float()
    valid = (diff.abs() > 1e-6).float()
    for _ in range(grid["epochs"]):
        s = proj(z) @ emb.T
        sd = s[:, :, None] - s[:, None, :]
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(sd, target, reduction="none")
                * valid).sum() / valid.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        S = {sp: (proj(torch.from_numpy(std(pooled(run, sp)))) @ emb.T).numpy() for sp in splits}
    return [({"k": k}, from_weights(run, splits, {sp: topk_uniform(S[sp], k) for sp in splits}))
            for k in (1, 2, 3, 5)]


def logme_score(f, y):
    """LogME (You et al., 2021): mean per-class log marginal evidence of a linear model."""
    f = f.astype(np.float64)
    n, d = f.shape
    u, s, _ = np.linalg.svd(f, full_matrices=False)
    s = s ** 2
    s_full = np.concatenate([s, np.zeros(max(0, d - len(s)))])
    ev = []
    for k in np.unique(y):
        yk = (y == k).astype(np.float64)
        x = u.T @ yk
        x2 = x ** 2
        res_x2 = (yk ** 2).sum() - x2.sum()
        alpha, beta = 1.0, 1.0
        for _ in range(11):
            t = alpha / beta
            gamma = (s / (s + t)).sum()
            m2 = (s * x2 / ((t + s) ** 2)).sum()
            res2 = (x2 / ((1 + s / t) ** 2)).sum() + res_x2
            new_alpha, new_beta = gamma / (m2 + 1e-5), (n - gamma) / (res2 + 1e-5)
            done = abs(new_alpha / new_beta - t) / t < 0.01
            alpha, beta = new_alpha, new_beta
            if done:
                break
        e = (d / 2 * np.log(alpha) + n / 2 * np.log(beta) - 0.5 * np.log(alpha + beta * s_full).sum()
             - beta / 2 * res2 - alpha / 2 * m2 - n / 2 * np.log(2 * np.pi))
        ev.append(e / n)
    return float(np.mean(ev))


def logme_selection(run, splits):
    rank = {}
    for t in run["tasks"]:
        sc = np.array([logme_score(f, run[t]["y_tr"]) for f in run[t]["X_tr"]])
        rank[t] = np.argsort(-sc)
    res = []
    for k in (1, 2, 3):
        w = {t: np.isin(np.arange(run["n_cand"]), rank[t][:k]).astype(float) for t in run["tasks"]}
        res.append(({"k": k}, per_task_const(run, splits, w)))
    return res


# ----------------------------------------------------------------------------- dynamic ensemble selection
def _patch_deslib():
    from deslib.base import BaseDS
    from sklearn.utils.validation import validate_data

    def _vd(self, X="no_validation", y="no_validation", **kw):
        if isinstance(y, str):
            return validate_data(self, X, **kw)
        return validate_data(self, X, y, **kw)
    BaseDS._validate_data = _vd


from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402


class _Lookup(ClassifierMixin, BaseEstimator):
    """A fitted 'classifier' that returns cached candidate probabilities by sample id."""

    def __init__(self, table=None, scale=1e-9):
        self.table, self.scale = table, scale
        if table is not None:
            self.classes_ = np.arange(table.shape[1])

    def _ids(self, X):
        return np.rint(np.asarray(X)[:, -1] / self.scale).astype(int)

    def predict_proba(self, X):
        return self.table[self._ids(X)]

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def fit(self, X, y):
        return self

    def __sklearn_is_fitted__(self):
        return True


def des(run, splits, which, ks=(7,)):
    """DESlib dynamic selection. ks: neighbourhood sizes; one result per k, chosen on validation
    by the common rule (plan amendment of 2026-11-17: ks = (5, 7, 10, 25, 50, 100); before it, k = 7)."""
    return [r for k in ks for r in _des_k(run, splits, which, k)]


def _des_k(run, splits, which, k):
    _patch_deslib()
    from deslib.des import KNORAE, KNORAU, METADES
    cls = {"knora_u": KNORAU, "knora_e": KNORAE, "meta_des": METADES}[which]
    scale = 1e-9
    out = {sp: {} for sp in splits}
    std = router.Standardizer().fit(pooled(run, "tr"))
    for t in run["tasks"]:
        parts = ["tr"] + list(splits)
        P_all = np.concatenate([run[t][f"P_{sp}"] for sp in parts])     # (N, C, K)
        offs = np.cumsum([0] + [len(run[t][f"P_{sp}"]) for sp in parts])
        pool = [_Lookup(P_all[:, c], scale) for c in range(P_all.shape[1])]
        X = lambda sp, j: np.hstack([std(run[t][f"Z_{sp}"]).astype(np.float64),
                                     ((offs[j] + np.arange(len(run[t][f"P_{sp}"]))) * scale)[:, None]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = cls(pool, k=k, random_state=run["seed"]) if which != "meta_des" else \
                cls(pool, k=k, Kp=5, random_state=run["seed"])
            m.fit(X("tr", 0), run[t]["y_tr"])
            for j, sp in enumerate(splits, start=1):
                Xs = X(sp, j)
                P = m.predict_proba(Xs)
                bad = ~np.isfinite(P).all(1)
                if bad.any():  # no competent classifier (zero weights): use all candidates, as KNORA does
                    P[bad] = run[t][f"P_{sp}"][bad].mean(1)
                if which == "meta_des":
                    M = np.ones((len(P), len(pool)), bool)
                else:
                    _, nbr = m.get_competence_region(Xs)
                    hits = m.DSEL_processed_[nbr]                         # (n, k, C)
                    if which == "knora_u":
                        M = hits.any(1)
                    else:
                        M = np.zeros((len(P), len(pool)), bool)
                        for i in range(len(P)):
                            for kk in range(hits.shape[1], 0, -1):
                                sel = hits[i, :kk].all(0)
                                if sel.any():
                                    M[i] = sel
                                    break
                            if not M[i].any():
                                M[i] = True
                    M[~M.any(1)] = True
                out[sp][t] = (P, M)
    return [({"k": k}, out)]


# ----------------------------------------------------------------------------- oracle
def oracle_single(run, splits):
    out = {}
    for sp in splits:
        out[sp] = {}
        for t in run["tasks"]:
            P, y = run[t][f"P_{sp}"], run[t][f"y_{sp}"]
            c = true_prob(P, y).argmax(1)
            M = np.eye(P.shape[1], dtype=bool)[c]
            out[sp][t] = (P[np.arange(len(y)), c], M)
    return [({}, out)]


# ----------------------------------------------------------------------------- ours: collection-level router
def collection_router(run, splits, estimator="knn", sizes=(2, 3), neighbours=(25, 50, 100),
                      picks=(1, 3), cost=None, lams=(0.0,), Zkey="Z", grid=None, fusion="arith"):
    """HealthRouter (collection form): estimate, for every collection S of up to `size`
    candidates, the expected log-likelihood u_S(x) = log mean_{c in S} p_c(y|x) on inputs like
    x, then keep the best-scoring collection(s) and average their members.

    estimator="knn": cosine-neighbourhood average of out-of-fold collection utilities in the
    router-feature space; estimator="mlp": a gate trained on the same targets."""
    Ztr = pooled(run, "tr", Zkey)
    std = router.Standardizer().fit(Ztr)
    U = np.concatenate([true_prob(run[t]["P_tr"], run[t]["y_tr"]) for t in run["tasks"]])
    res = []
    if estimator == "knn":
        A = std(Ztr)
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        order = {}
        for sp in splits:
            B = std(pooled(run, sp, Zkey))
            B /= np.linalg.norm(B, axis=1, keepdims=True)
            order[sp] = np.argsort(-(B @ A.T), 1)[:, :max(neighbours)]
    for size in sizes:
        lib = router.collection_library(run["n_cand"], size)
        CU = router.collection_utility(U, lib)
        if Zkey == "T":   # oracle domain identity: the task's mean collection utility
            tid_tr = np.concatenate([np.full(len(run[t]["y_tr"]), i) for i, t in enumerate(run["tasks"])])
            means = np.stack([CU[tid_tr == i].mean(0) for i in range(len(run["tasks"]))])
            ests = {"task_mean": {sp: means[np.concatenate([np.full(len(run[t][f"y_{sp}"]), i)
                                                             for i, t in enumerate(run["tasks"])])]
                                  for sp in splits}}
        elif estimator == "knn":
            ests = {nn: {sp: CU[order[sp][:, :nn]].mean(1) for sp in splits} for nn in neighbours}
        else:
            ests = {}
            for tau in (0.25, 1.0):
                g = router.train_collection_gate(std(Ztr), U, lib, tau=tau, beta=1.0,
                                                 hidden=grid["hidden"], epochs=grid["epochs"],
                                                 lr=grid["lr"], wd=grid["weight_decay"],
                                                 dropout=grid["dropout"], seed=run["seed"])
                ests[tau] = {sp: router.scores(g, std(pooled(run, sp, Zkey))) for sp in splits}
        for nn, est in ests.items():
            for m in picks:
                for lam in lams:
                    W = {sp: router.select_collection(est[sp], lib, 1e-3 if m == 1 else 0.05, m,
                                                      cost, lam) for sp in splits}
                    cfg = {"estimator": estimator, "max_size": size,
                           ("neighbours" if estimator == "knn" else "tau"): nn, "picks": m, "lambda": lam}
                    res.append((cfg, from_weights(run, splits, W, fusion == "geo")))
    return res


# ----------------------------------------------------------------------------- smart harness (sequential calling)
def smart_harness(run, splits, neighbours=(25, 50, 100), thetas=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01),
                  k_maxes=(2, 3, 5, 10), Zkey="Z"):
    """HealthRouter harness: rank candidates per sample by the neighbourhood estimate of their
    out-of-fold utility (HealthRouter's size-1 collection scores), call the top-ranked model,
    and keep adding the next-ranked model (uniform re-fusion) while the fused max-probability
    is below theta, up to k_max models. theta and k_max are chosen on validation."""
    Ztr = pooled(run, "tr", Zkey)
    std = router.Standardizer().fit(Ztr)
    A = std(Ztr)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    L = np.log(np.clip(np.concatenate([true_prob(run[t]["P_tr"], run[t]["y_tr"])
                                       for t in run["tasks"]]), EPS, 1))
    order = {}
    for sp in splits:
        B = std(pooled(run, sp, Zkey))
        B /= np.linalg.norm(B, axis=1, keepdims=True)
        order[sp] = np.argsort(-(B @ A.T), 1)[:, :max(neighbours)]
    res = []
    for nn in neighbours:
        rank = {sp: split_by_task(run, sp, np.argsort(-L[order[sp][:, :nn]].mean(1), 1)) for sp in splits}
        # cumulative uniform fusions along each sample's ranking: (n, C, K)
        cums = {}
        for sp in splits:
            cums[sp] = {}
            for t in run["tasks"]:
                P = run[t][f"P_{sp}"]
                ranked = np.take_along_axis(P, rank[sp][t][:, :, None], 1)
                cums[sp][t] = np.cumsum(ranked, 1) / np.arange(1, P.shape[1] + 1)[None, :, None]
        for k_max in k_maxes:
            for th in thetas:
                out = {}
                for sp in splits:
                    out[sp] = {}
                    for t in run["tasks"]:
                        cum = cums[sp][t][:, :k_max]                     # (n, k_max, K)
                        conf = cum.max(2)
                        ok = conf >= th
                        n_call = np.where(ok.any(1), ok.argmax(1) + 1, k_max)
                        fused = cum[np.arange(len(cum)), n_call - 1]
                        M = np.zeros((len(cum), run["n_cand"]), bool)
                        for j in range(k_max):
                            sel = n_call > j
                            M[np.where(sel)[0], rank[sp][t][sel, j]] = True
                        out[sp][t] = (fused, M)
                res.append(({"neighbours": nn, "k_max": k_max, "theta": th}, out))
    return res


def fixed_order_harness(run, splits, thetas=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01), k_maxes=(2, 3, 5, 10)):
    """Harness control (plan amendment of 2026-11-17): the harness's confidence stop rule, but candidates
    are called in a FIXED per-task order (descending validation score on that task) instead of
    HealthRouter's per-image order. No router features are needed. theta and k_max chosen on validation."""
    rank = {t: np.argsort(-np.nan_to_num(val_metric_per_candidate(run, t, "va"), nan=-1.0)) for t in run["tasks"]}
    cums = {sp: {} for sp in splits}
    for sp in splits:
        for t in run["tasks"]:
            P = run[t][f"P_{sp}"][:, rank[t]]
            cums[sp][t] = np.cumsum(P, 1) / np.arange(1, P.shape[1] + 1)[None, :, None]
    res = []
    for k_max in k_maxes:
        for th in thetas:
            out = {}
            for sp in splits:
                out[sp] = {}
                for t in run["tasks"]:
                    cum = cums[sp][t][:, :k_max]
                    ok = cum.max(2) >= th
                    n_call = np.where(ok.any(1), ok.argmax(1) + 1, k_max)
                    fused = cum[np.arange(len(cum)), n_call - 1]
                    M = np.zeros((len(cum), run["n_cand"]), bool)
                    for j in range(k_max):
                        M[n_call > j, rank[t][j]] = True
                    out[sp][t] = (fused, M)
            res.append(({"order": "fixed per task (validation)", "k_max": k_max, "theta": th}, out))
    return res


# ----------------------------------------------------------------------------- distance fallback (added after the external results)
def with_fallback(run, splits, base_results, K=50, qs=(90, 95, 99), Zkey="Z"):
    """Route an input to the uniform ten-model ensemble when its mean cosine distance to its K
    nearest meta-training neighbours (router features) exceeds the q-th percentile of the same
    distance on the in-source validation split; otherwise keep the base method's output.
    The threshold uses validation data only; q is chosen on validation with every other setting."""
    Ztr = pooled(run, "tr", Zkey)
    std = router.Standardizer().fit(Ztr)
    A = std(Ztr)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    dist = {}
    for sp in set(splits) | {"va"}:
        B = std(pooled(run, sp, Zkey))
        B /= np.linalg.norm(B, axis=1, keepdims=True)
        sims = np.sort(B @ A.T, 1)[:, -K:]
        dist[sp] = split_by_task(run, sp, 1.0 - sims.mean(1))
    val_all = np.concatenate([dist["va"][t] for t in run["tasks"]])
    uni = uniform(run, splits)[0][1]
    res = []
    for q in qs:
        thr = float(np.percentile(val_all, q))
        for cfg, out in base_results:
            new = {}
            for sp in splits:
                new[sp] = {}
                for t in run["tasks"]:
                    far = dist[sp][t] > thr
                    P = np.where(far[:, None], uni[sp][t][0], out[sp][t][0])
                    M = np.where(far[:, None], True, out[sp][t][1])
                    new[sp][t] = (P, M)
            res.append(({**cfg, "fallback_q": q}, new))
    return res
