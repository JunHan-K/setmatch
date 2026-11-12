"""11-05 RANK (PREREG 'RANK'; saved predictions only): SetWiseMatch's advantage over SCH by the cosine rank of the best true
match among present episodes with the target in the shortlist (r = 1, 2, 3-5, 6-64); success = accepted and correct at
each rule's frozen threshold, mean over test sizes. Also the share of r >= 2 per test size. Writes results/setmatch/rank.json."""
import json
import os
import sys

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "sm_swap.py")).read(); exec(src[src.index("\nSMT = "):src.index("\nres = {}")])   # noqa: S102 (own files: definitions only)
sys.path.insert(0, H)
import sm_v5 as V5M
src = open(os.path.join(H, "sm_idb.py")).read(); exec(src[src.index("\ndef test_pids"):src.index("\nres = {}")])   # noqa: S102

BINS = (("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5), ("6-64", 6, 64))
res = {}
for key in CELLS:
    z = dict(np.load(os.path.join(V5, f"{key}_meta.npz"))); E, K = z["E"], z["K"]; _, c = test_pids(key); assert (c["E"] == E).all(), key
    tm = c["tm"]; ins = (E == 1) & tm.any(1); rk = np.where(tm.any(1), tm.argmax(1) + 1, 0)
    r = dict(share_r2={int(k): float((rk[ins & (K == k)] >= 2).mean()) for k in np.unique(K)})
    for m, v in (("sm", "attn" + SMT), ("sch", "selch_s10sch"), ("ctx", "ctx" + SMT)):
        y = dict(np.load(os.path.join(V5, f"{key}_{v}.npz"))); S = [s for s in range(20) if f"conf_te_s{s}" in y]; acc = {b: [] for b, _, _ in BINS}
        for s in S:
            ok = mask_rule(y[f"conf_tr_s{s}"], y[f"corr_tr_s{s}"], y["Etr"], y[f"conf_te_s{s}"], "prob") & (y[f"corr_te_s{s}"] > 0)
            for b, lo, hi in BINS:
                g = ins & (rk >= lo) & (rk <= hi); acc[b].append(np.mean([ok[g & (K == k)].mean() for k in np.unique(K) if (g & (K == k)).any()]) if g.any() else np.nan)
        r[m] = {b: float(np.mean(acc[b])) for b in acc}
    r["d"] = {b: r["sm"][b] - r["sch"][b] for b, _, _ in BINS}; res[key] = r
    print(key.ljust(30), json.dumps({b: round(x, 4) for b, x in r["d"].items()}), json.dumps({k: round(x, 3) for k, x in r["share_r2"].items()}), flush=True)
nn = lambda a: [x for x in a if np.isfinite(x)]
slope = [res[k]["d"]["3-5"] - res[k]["d"]["1"] for k in CELLS]
grow = [res[k]["share_r2"][max(res[k]["share_r2"])] > res[k]["share_r2"][min(res[k]["share_r2"])] for k in CELLS]
V = dict(med_d={b: float(np.median(nn([res[k]["d"][b] for k in CELLS]))) for b, _, _ in BINS},
         med={m: {b: float(np.median(nn([res[k][m][b] for k in CELLS]))) for b, _, _ in BINS} for m in ("sm", "sch", "ctx")},
         slope_pos=int(sum(x > 0 for x in slope if np.isfinite(x))), med_slope=float(np.median(nn(slope))), share_grows=int(sum(grow)),
         med_share_small=float(np.median([res[k]["share_r2"][min(res[k]["share_r2"])] for k in CELLS])), med_share_large=float(np.median([res[k]["share_r2"][max(res[k]["share_r2"])] for k in CELLS])))
a_ = V["slope_pos"] >= 19 and V["med_slope"] > 0; b_ = V["share_grows"] >= 24
V["RANK_a"], V["RANK_b"] = a_, b_; V["verdict"] = "gain concentrates on shift-induced competition" if a_ and b_ else ("gain grows with competition" if a_ else "supplement")
print("RANK", json.dumps(V)); json.dump(dict(cells=res, verdict=V), open(os.path.join(O, "rank.json"), "w"), indent=1)
