"""Candidate-set-shift analysis: performance as a FUNCTION of gallery size K (the phenomenon the paper is about),
instead of a single size-aggregated scalar. For cosine / GS / SetWiseMatch it reports, per gallery size K
(averaged over the ready cells):
  - strict accuracy at the frozen (train-chosen) threshold,
  - forced-choice selection accuracy among present queries (cosine top-1 vs SetWiseMatch re-ranking),
plus SM size-aware frozen thresholding (t = b0 + b1*log N) so acceptance tracks the gallery size. Reads the v5
prediction dumps; writes results/setmatch/shift.json. CPU only (numpy/scipy)."""
import glob
import json
import os
from collections import defaultdict

import numpy as np

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V5 = os.path.join(R, "results/setmatch/v5")
SMT = os.environ.get("SM_TAG", "_s10main")
lg = lambda x: np.log(np.clip(x, 1e-7, 1 - 1e-7)) - np.log1p(-np.clip(x, 1e-7, 1 - 1e-7))


def sc(x, c, E, grid):                                                 # strict curves: P(x>t & correct | E=1), P(x<=t | E=0)
    a = np.sort(x[(E == 1) & (c > 0)]); z = np.sort(x[E == 0]); n1 = max((E == 1).sum(), 1); n0 = max(len(z), 1)
    return (len(a) - np.searchsorted(a, grid, side="right")) / n1, np.searchsorted(z, grid, side="right") / n0


def frozen_per_k(xtr, ctr, Etr, xte, cte, E, K, space=None):          # strict accuracy per K at a single train-chosen threshold
    if space == "prob":                                              # only posteriors are logit-transformed; GS 'sdse' is ALREADY a logit (do NOT transform it again)
        xtr, xte = lg(xtr), lg(xte)
    grid = np.unique(np.quantile(xtr, np.linspace(0, 1, 401))); g1, g0 = sc(xtr, ctr, Etr, grid); j = int(np.argmax(0.5 * g1 + 0.5 * g0))
    out = {}
    for k in np.unique(K):
        m = K == k; h1, h0 = sc(xte[m], cte[m], E[m], grid); out[int(k)] = float((0.5 * h1 + 0.5 * h0)[j])
    return out


def main():
    accc, accg, accs = defaultdict(list), defaultdict(list), defaultdict(list)
    selcos, selsm = defaultdict(list), defaultdict(list)
    for meta in sorted(glob.glob(os.path.join(V5, "*_meta.npz"))):
        key = os.path.basename(meta)[:-9]; a = os.path.join(V5, f"{key}_attn{SMT}.npz")
        if not os.path.exists(a):
            continue
        z = dict(np.load(meta)); E, K = z["E"], z["K"]; y = dict(np.load(a)); S = [s for s in range(20) if f"conf_te_s{s}" in y]
        ac = frozen_per_k(z["raw_tr"], z["Vtr"], z["Etr"], z["raw_te"], z["Vte"], E, K)
        ag = frozen_per_k(z["sdse_tr"], z["Vtr"], z["Etr"], z["sdse_te"], z["Vte"], E, K, "logit")
        asm = defaultdict(list)
        for s in S:
            for k, v in frozen_per_k(y[f"conf_tr_s{s}"], y[f"corr_tr_s{s}"], y["Etr"], y[f"conf_te_s{s}"], y[f"corr_te_s{s}"], E, K, "prob").items():
                asm[k].append(v)
        for k in ac:
            accc[k].append(ac[k]); accg[k].append(ag[k]); accs[k].append(float(np.mean(asm[k])))
        for k in np.unique(K):                                        # forced-choice selection among present: cosine top-1 (Vte) vs SM choice (corr)
            mp = (K == k) & (E == 1)
            if mp.sum() < 20:
                continue
            selcos[int(k)].append(float(z["Vte"][mp].mean())); selsm[int(k)].append(float(np.mean([y[f"corr_te_s{s}"][mp].mean() for s in S])))
    out = {"strict_acc_vs_K": {}, "selection_acc_vs_K": {}}
    print(f"{'K':>6} {'cells':>5} | {'cos':>6} {'GS':>6} {'SM':>6} {'SM-GS':>7} | {'sel_cos':>7} {'sel_SM':>7}")
    for k in sorted(accc):
        row = (float(np.mean(accc[k])), float(np.mean(accg[k])), float(np.mean(accs[k])))
        sl = (float(np.mean(selcos[k])), float(np.mean(selsm[k]))) if k in selcos else (float("nan"), float("nan"))
        out["strict_acc_vs_K"][k] = {"cos": row[0], "gs": row[1], "sm": row[2], "cells": len(accc[k])}
        out["selection_acc_vs_K"][k] = {"cos_top1": sl[0], "sm": sl[1]}
        print(f"{k:>6} {len(accc[k]):>5} | {row[0]:6.3f} {row[1]:6.3f} {row[2]:6.3f} {row[2]-row[1]:+7.3f} | {sl[0]:7.3f} {sl[1]:7.3f}")
    json.dump(out, open(os.path.join(R, "results/setmatch/shift.json"), "w"), indent=1)
    print("wrote results/setmatch/shift.json")


if __name__ == "__main__":
    main()
