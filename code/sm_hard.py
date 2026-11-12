"""11-05 HARD (PREREG 'HARD'; saved predictions only): the Figure 1 situation. Present test episodes whose cosine top-1
is wrong although the target is in the top-64 shortlist; success = accepted and correct at each rule's frozen threshold
(CTRL rule), mean over test sizes and at the largest size. SetWiseMatch vs SCH vs set pooling. Writes results/setmatch/hard.json."""
import json
import os
import sys

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "sm_swap.py")).read(); exec(src[src.index("\nSMT = "):src.index("\nres = {}")])   # noqa: S102 (own files: definitions only)
sys.path.insert(0, H)
import sm_v5 as V5M
src = open(os.path.join(H, "sm_idb.py")).read(); exec(src[src.index("\ndef test_pids"):src.index("\nres = {}")])   # noqa: S102

res = {}
for key in CELLS:
    z = dict(np.load(os.path.join(V5, f"{key}_meta.npz"))); E, K = z["E"], z["K"]; _, c = test_pids(key); assert (c["E"] == E).all(), key
    hard = (E == 1) & (z["Vte"] <= 0) & c["tm"].any(1); r = dict(n_hard=int(hard.sum()))
    for m, v in (("sm", "attn" + SMT), ("sch", "selch_s10sch"), ("ctx", "ctx" + SMT)):
        y = dict(np.load(os.path.join(V5, f"{key}_{v}.npz"))); S = [s for s in range(20) if f"conf_te_s{s}" in y]; mean_, big = [], []
        for s in S:
            ok = mask_rule(y[f"conf_tr_s{s}"], y[f"corr_tr_s{s}"], y["Etr"], y[f"conf_te_s{s}"], "prob") & (y[f"corr_te_s{s}"] > 0)
            per = [ok[hard & (K == k)].mean() for k in np.unique(K) if (hard & (K == k)).any()]; mean_.append(np.mean(per)); big.append(ok[hard & (K == K.max())].mean())
        r[m] = float(np.mean(mean_)); r[m + "_big"] = float(np.mean(big))
    res[key] = r; print(key.ljust(30), json.dumps({k: round(v, 4) for k, v in r.items()}), flush=True)
d = np.array([res[k]["sm"] - res[k]["sch"] for k in CELLS]); db = np.array([res[k]["sm_big"] - res[k]["sch_big"] for k in CELLS])
V = dict(med=dict(sm=float(np.median([res[k]["sm"] for k in CELLS])), sch=float(np.median([res[k]["sch"] for k in CELLS])), ctx=float(np.median([res[k]["ctx"] for k in CELLS]))),
         med_d=float(np.median(d)), sm_higher=int((d > 0).sum()), med_d_big=float(np.median(db)), sm_higher_big=int((db > 0).sum()),
         sm_gt_ctx=sum(res[k]["sm"] > res[k]["ctx"] for k in CELLS))
a_ = V["med_d"] > 0 and V["sm_higher"] >= 19; b_ = V["sm_higher_big"] >= 19
V["verdict"] = ("Figure-1 situation solved better than SCH" + (", holds at the largest size" if b_ else "")) if a_ else "supplement"
print("HARD", json.dumps(V)); json.dump(dict(cells=res, verdict=V), open(os.path.join(O, "hard.json"), "w"), indent=1)
