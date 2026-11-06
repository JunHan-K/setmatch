"""11-05 IDB (reviewer: resampling unit). Identity-cluster bootstrap of the main comparison: per setting, the frozen-
threshold strict accuracy difference SetMatch - GS (and SetMatch - SCH when its predictions exist). Each of 1000
replicates draws test identities with replacement (all queries of an identity and all gallery sizes of a query move
together) and SetMatch seeds with replacement; thresholds stay fixed from training (no test information). Reports the
paired effect, its 95% interval and the share of settings whose interval excludes 0. Writes results/setmatch/idb.json."""
import json
import os
import sys

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "sm_swap.py")).read(); exec(src[src.index("\nSMT = "):src.index("\nres = {}")])   # noqa: S102 (own file: definitions only)
sys.path.insert(0, H)
import sm_v5 as V5M

B = 1000; rng = np.random.default_rng(5)


def wacc(acc, corr, E, K, W):
    """weighted strict accuracy (mean over test sizes) for every replicate row of W"""
    out = []
    for k in np.unique(K):
        m = K == k; p, n = m & (E == 1), m & (E == 0)
        out.append(0.5 * (W[:, p] @ (acc[p] & (corr[p] > 0)).astype(np.float32)) / W[:, p].sum(1) + 0.5 * (W[:, n] @ (~acc[n]).astype(np.float32)) / W[:, n].sum(1))
    return np.mean(out, 0)


def test_pids(key):
    d, e = key.split("_", 1); D = V5M.market(e, "test", d) if d == "market" else V5M.scene(e, "test", d, 1 if d == "cuhk" else 20)
    c = np.load(os.path.join(O, "cache", f"{key}{'' if d in ('cuhk', 'market') else '_q20'}_k{KT[d]}_te.npz")); return D.qpid[c["qi"]], c


res = {}
for key in CELLS:
    z = dict(np.load(os.path.join(V5, f"{key}_meta.npz"))); y = dict(np.load(os.path.join(V5, f"{key}_attn{SMT}.npz"))); E, K = z["E"], z["K"]
    pid, c = test_pids(key); assert (c["E"] == E).all(), key
    _, g = np.unique(pid, return_inverse=True); W = rng.multinomial(g.max() + 1, np.full(g.max() + 1, 1 / (g.max() + 1)), size=B).astype(np.float32)[:, g]
    S = [s for s in range(20) if f"conf_te_s{s}" in y]; mg = mask_rule(z["sdse_tr"], z["Vtr"], z["Etr"], z["sdse_te"], "logit"); gs = wacc(mg, z["Vte"], E, K, W)
    per = np.stack([wacc(mask_rule(y[f"conf_tr_s{s}"], y[f"corr_tr_s{s}"], y["Etr"], y[f"conf_te_s{s}"], "prob"), y[f"corr_te_s{s}"], E, K, W) for s in S])
    pick = rng.integers(0, len(S), size=(B, len(S))); sm = per[pick, np.arange(B)[:, None]].mean(1); r = {"n_ids": int(g.max() + 1)}
    for nm, base in (("gs", gs),):
        dd = sm - base; r[nm] = dict(delta=float(np.median(dd)), lo=float(np.quantile(dd, .025)), hi=float(np.quantile(dd, .975)))
    fs = os.path.join(V5, f"{key}_selch_s10sch.npz")
    if os.path.exists(fs):
        ys = dict(np.load(fs)); Ss = [s for s in range(20) if f"conf_te_s{s}" in ys]
        ps = np.stack([wacc(mask_rule(ys[f"conf_tr_s{s}"], ys[f"corr_tr_s{s}"], ys["Etr"], ys[f"conf_te_s{s}"], "prob"), ys[f"corr_te_s{s}"], E, K, W) for s in Ss])
        sc = ps[rng.integers(0, len(Ss), size=(B, len(Ss))), np.arange(B)[:, None]].mean(1); dd = sm - sc
        r["sch"] = dict(delta=float(np.median(dd)), lo=float(np.quantile(dd, .025)), hi=float(np.quantile(dd, .975)))
    res[key] = r; print(key.ljust(30), json.dumps(r), flush=True)
V = {nm: dict(median_delta=float(np.median([res[k][nm]["delta"] for k in CELLS if nm in res[k]])), ci_above_0=sum(res[k][nm]["lo"] > 0 for k in CELLS if nm in res[k]),
              ci_below_0=sum(res[k][nm]["hi"] < 0 for k in CELLS if nm in res[k]), n=sum(nm in res[k] for k in CELLS)) for nm in ("gs", "sch")}
print("IDB", json.dumps(V)); json.dump(dict(cells=res, verdict=V), open(os.path.join(O, "idb.json"), "w"), indent=1)
