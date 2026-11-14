"""Query-blind presence probe (benchmark leak audit). Tests whether the episode construction lets a classifier
tell E=1 (target present) from E=0 (absent) using ONLY candidate-set structure, with the query removed. If the
within-K AUC stays ~0.50 the construction carries no presence shortcut.

Features are computed over the FULL constructed candidate set (identity count, N) plus candidate-candidate affinity
on a RANDOM (never score-ranked) subset, so the legitimate query-cosine ranking cannot leak presence into them.
Logistic regression + AUC are implemented in numpy. Usage:  python sm_probe.py <enc> <dataset> <train|test>"""
import os
import sys

import numpy as np
from scipy.stats import rankdata

import sm_v5 as V

CAP = 64              # affinity subset size (the model sees the encoded top-64 shortlist)
NQ = 250             # queries sampled per cell
_prng = np.random.default_rng(7)


def auc(score, y):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return np.nan
    r = rankdata(score); n1 = y.sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def logreg_cv_auc(X, y, groups, folds=5, iters=400, lr=0.3, l2=1e-2):
    """L2 logistic regression with GROUP k-fold CV (episodes of the same query stay in one fold, so train/val are
    independent) and standardization fit on the TRAIN fold only (no validation leakage). Pooled out-of-fold AUC."""
    X = np.asarray(X, float); y = np.asarray(y, float); groups = np.asarray(groups)
    uq = np.unique(groups); np.random.default_rng(1).shuffle(uq)
    fold_of = {g: k % folds for k, g in enumerate(uq)}; fo = np.array([fold_of[g] for g in groups])
    oof = np.full(len(y), np.nan)
    for f in range(folds):
        te = np.flatnonzero(fo == f); tr = np.flatnonzero(fo != f)
        if len(te) == 0 or len(np.unique(y[tr])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9                    # fit standardization on train fold only
        Xtr = (X[tr] - mu) / sd; Xte = (X[te] - mu) / sd; ytr = y[tr]; w = np.zeros(X.shape[1]); b = 0.0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(Xtr @ w + b))); gdt = p - ytr
            w -= lr * (Xtr.T @ gdt / len(tr) + l2 * w); b -= lr * gdt.mean()
        oof[te] = Xte @ w + b
    m = ~np.isnan(oof)
    return auc(oof[m], y[m])


def feats(D, it):
    """query-blind features of one candidate set: counts over the FULL set, affinity over a RANDOM subset"""
    itf = np.asarray(it); n = len(itf)
    sub = itf if n <= CAP else itf[_prng.permutation(n)[:CAP]]
    C = D.Fg[sub]; A = C @ C.T; m = len(sub)
    offd = A[~np.eye(m, dtype=bool)] if m > 1 else np.array([0.0])
    rowmax = (A - np.eye(m)).max(1) if m > 1 else np.array([0.0])
    pid = D.gpid[itf]; nid = len(np.unique(pid))
    return [n, nid, nid / max(n, 1), offd.mean(), offd.std(), offd.max(), np.quantile(offd, 0.95),
            (offd > 0.5).mean(), (offd > 0.9).mean(), rowmax.mean(),
            np.bincount(np.unique(pid, return_inverse=True)[1]).max() / max(n, 1)]


def main():
    enc, ds, split = sys.argv[1], sys.argv[2], sys.argv[3]
    D = V.market(enc, split, ds) if ds in ("market", "msmt", "rpif") else V.scene(enc, split, ds)
    if split == "train":                                             # probe the ACTUAL training sizes, not the test-expansion sizes
        Ks = tuple(k for k in V.TRK.get(ds, ()) if 0 <= k <= 20000)
    else:
        Ks = tuple(k for k in (getattr(D, "Gs", None) or V.TEK.get(ds, ())) if 0 <= k <= 20000)
    eps = V.episodes(D, Ks, NQ, seed=0)
    byK = {}
    for (i, it, tm, s, E, K, ngt) in eps:
        byK.setdefault(int(K), ([], [], []))
        rec = byK[int(K)]; rec[0].append(feats(D, it)); rec[1].append(int(E)); rec[2].append(int(i))   # [2] = query id -> group for group-CV
    print(f"==== {ds}/{enc}/{split}  (query-blind presence AUC, query-group CV; 0.50 = no shortcut) ====", flush=True)
    for K in sorted(byK):
        X, y, g = np.array(byK[K][0], float), np.array(byK[K][1]), np.array(byK[K][2])
        if len(np.unique(y)) < 2 or len(y) < 40:
            continue
        print(f"  K={K:<6} n={len(y):<5} E1={int(y.sum()):<4} AUC={logreg_cv_auc(X, y, g):.3f}", flush=True)


if __name__ == "__main__":
    main()
