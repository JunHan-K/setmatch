"""11-05 CHO (PREREG 'CHO'; saved predictions only): does the joint candidate/no-match output improve the choice itself?
Per setting, choice accuracy R1 = P(choice is the true match | present) averaged over test sizes, SetMatch (attn_s20)
minus the SCH selector (selch_s10sch: same architecture, inputs, episodes and budget, no-match output removed), with an
identity-cluster bootstrap (1000; seeds resampled); also within the training size range vs extrapolated sizes (22
settings, as LOC). Writes results/setmatch/cho.json."""
import json
import os
import sys

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "sm_swap.py")).read(); exec(src[src.index("\nSMT = "):src.index("\nres = {}")])   # noqa: S102 (own files: definitions only)
sys.path.insert(0, H)
import sm_v5 as V5M
from sm_v5 import TRK
src = open(os.path.join(H, "sm_idb.py")).read(); exec(src[src.index("\nB = 1000"):src.index("\nres = {}")])   # noqa: S102


def r1k(corr, E, K, W, ks):
    return np.mean([(W[:, (K == k) & (E == 1)] @ (corr[(K == k) & (E == 1)] > 0).astype(np.float32)) / W[:, (K == k) & (E == 1)].sum(1) for k in ks], 0)


res = {}
for key in CELLS:
    d = key.split("_")[0]; z = dict(np.load(os.path.join(V5, f"{key}_meta.npz"))); E, K = z["E"], z["K"]
    pid, c = test_pids(key); assert (c["E"] == E).all(), key
    _, g = np.unique(pid, return_inverse=True); W = rng.multinomial(g.max() + 1, np.full(g.max() + 1, 1 / (g.max() + 1)), size=B).astype(np.float32)[:, g]
    ks = list(np.unique(K)); ext = 0 not in TRK[d]; kin = [k for k in ks if k <= max(TRK[d])]; kex = [k for k in ks if k > max(TRK[d])] if ext else []
    out = {}
    for m, v in (("sm", "attn" + SMT), ("sel", "selch_s10sch")):
        y = dict(np.load(os.path.join(V5, f"{key}_{v}.npz"))); S = [s for s in range(20) if f"conf_te_s{s}" in y]
        per = np.stack([np.stack([r1k(y[f"corr_te_s{s}"], E, K, W, kk) for kk in ([ks, kin, kex] if kex else [ks])]) for s in S])   # S x groups x B
        out[m] = per[rng.integers(0, len(S), size=(B, len(S))), :, np.arange(B)[:, None]].mean(1)                                  # B x groups
    cos = r1k(z["Vte"], E, K, np.ones((1, len(E)), np.float32), ks)[0]
    dd = out["sm"] - out["sel"]; r = dict(r1_sm=float(np.median(out["sm"][:, 0])), r1_sel=float(np.median(out["sel"][:, 0])), r1_cos=float(cos),
                                         d=float(np.median(dd[:, 0])), lo=float(np.quantile(dd[:, 0], .025)), hi=float(np.quantile(dd[:, 0], .975)))
    if kex:
        ds = dd[:, 2] - dd[:, 1]; r.update(d_in=float(np.median(dd[:, 1])), d_ex=float(np.median(dd[:, 2])), d_shift=float(np.median(ds)),
                                           s_lo=float(np.quantile(ds, .025)), s_hi=float(np.quantile(ds, .975)))
    res[key] = r; print(key.ljust(30), json.dumps({k: round(v, 4) for k, v in r.items()}), flush=True)
X = [k for k in CELLS if "d_shift" in res[k]]; md = lambda f, ks: float(np.median([res[k][f] for k in ks]))
V = dict(med_d=md("d", CELLS), sm_higher=sum(res[k]["d"] > 0 for k in CELLS), ci_above=sum(res[k]["lo"] > 0 for k in CELLS), ci_below=sum(res[k]["hi"] < 0 for k in CELLS),
         med_r1=dict(sm=md("r1_sm", CELLS), sel=md("r1_sel", CELLS), cos=md("r1_cos", CELLS)),
         n_ext=len(X), med_d_in=md("d_in", X), med_d_ex=md("d_ex", X), med_d_shift=md("d_shift", X), shift_pos=sum(res[k]["d_shift"] > 0 for k in X),
         shift_ci_above=sum(res[k]["s_lo"] > 0 for k in X))
a_ = V["med_d"] > 0 and V["sm_higher"] >= 19; b_ = V["med_d_shift"] > 0 and V["shift_pos"] >= 15
V["CHO_a"], V["CHO_b"] = a_, b_; V["verdict"] = "improves choice, more as set grows" if a_ and b_ else ("improves choice" if a_ else "not shown")
print("CHO", json.dumps(V)); json.dump(dict(cells=res, verdict=V), open(os.path.join(O, "cho.json"), "w"), indent=1)
