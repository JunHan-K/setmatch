"""11-05 SRC (PREREG 'SRC'; saved predictions only): source-fixed false-accept targets. Threshold = (1 - alpha) quantile of
the training absent-episode scores, alpha in {1%, 5%}; per test size FAR and DIR, worst-size FAR, max deviation from the
target, mean DIR. SetWiseMatch p_max and margin (attn_s10dra, same models), SCH, GS-MLP, GS. Writes results/setmatch/src.json."""
import json
import os
import sys

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "sm_swap.py")).read(); exec(src[src.index("\nSMT = "):src.index("\nres = {}")])   # noqa: S102 (own file: definitions only)
lgt = lambda x: np.log(np.clip(x, 1e-9, 1.0))


SA = os.environ.get("SA") == "1"                                   # 11-05 SRC2: size-aware FAR target (threshold linear in log N)


def thr(xtr, Etr, a, Ktr, Ntr, N):
    if not SA:
        return np.quantile(xtr[Etr == 0], 1 - a)
    ts, ln = zip(*[(np.quantile(xtr[(Ktr == k) & (Etr == 0)], 1 - a), np.log(Ntr[Ktr == k]).mean()) for k in np.unique(Ktr) if ((Ktr == k) & (Etr == 0)).any()])
    b1, b0 = np.polyfit(ln, ts, 1) if len(ts) > 1 else (0.0, ts[0]); return b0 + b1 * np.log(N)


def op(xtr, Etr, xte, cte, E, K, a):
    t = thr(xtr, Etr, a, G_["Ktr"], G_["Ntr"], G_["N"]); t = np.broadcast_to(t, xte.shape); F, D = [], []
    for k in np.unique(K):
        m = K == k; F.append((xte[m & (E == 0)] > t[m & (E == 0)]).mean()); D.append(((xte[m & (E == 1)] > t[m & (E == 1)]) & (cte[m & (E == 1)] > 0)).mean())
    return dict(far_w=float(max(F)), dev=float(max(abs(f - a) for f in F)), dir=float(np.mean(D)))


res = {}; G_ = {}
for key in CELLS:
    d_ = key.split("_")[0]; q_ = "" if d_ in ("cuhk", "market") else "_q20"
    ctr_ = np.load(os.path.join(O, "cache", f"{key}{q_}_k{KT[d_]}_tr.npz")); cte_ = np.load(os.path.join(O, "cache", f"{key}{q_}_k{KT[d_]}_te.npz"))
    G_.update(Ktr=ctr_["K"], Ntr=ctr_["N"].astype(float), N=cte_["N"].astype(float))
    z = dict(np.load(os.path.join(V5, f"{key}_meta.npz"))); E, K = z["E"], z["K"]; assert (ctr_["E"] == z["Etr"]).all() and (cte_["E"] == E).all(), key; L = lambda v: dict(np.load(os.path.join(V5, f"{key}_{v}.npz")))
    y, ys, yg = L("attn_s10dra"), L("selch_s10sch"), L("gsmlp_s10ng"); r = {}
    S = lambda d: [s for s in range(20) if f"conf_te_s{s}" in d]
    runs = dict(pmax=[(y[f"conf_tr_s{s}"], y["Etr"], y[f"conf_te_s{s}"], y[f"corr_te_s{s}"]) for s in S(y)],
                margin=[(lgt(y[f"conf_tr_s{s}"]) - lgt(y[f"null_tr_s{s}"]), y["Etr"], lgt(y[f"conf_te_s{s}"]) - lgt(y[f"null_te_s{s}"]), y[f"corr_te_s{s}"]) for s in S(y)],
                sch=[(ys[f"conf_tr_s{s}"], ys["Etr"], ys[f"conf_te_s{s}"], ys[f"corr_te_s{s}"]) for s in S(ys)],
                gs_mlp=[(yg[f"conf_tr_s{s}"], yg["Etr"], yg[f"conf_te_s{s}"], z["Vte"]) for s in S(yg)],
                gs=[(z["sdse_tr"], z["Etr"], z["sdse_te"], z["Vte"])])
    for m, R_ in runs.items():
        for a in (0.01, 0.05):
            o = [op(xtr, Etr, xte, cte, E, K, a) for xtr, Etr, xte, cte in R_]; r[f"{m}@{a}"] = {f: float(np.mean([x[f] for x in o])) for f in o[0]}
    res[key] = r; print(key.ljust(30), json.dumps({k: {f: round(v, 3) for f, v in x.items()} for k, x in r.items() if k.startswith(("pmax", "margin", "sch"))}), flush=True)
med = lambda k, f: float(np.median([res[c][k][f] for c in CELLS]))
V = dict(med={k: {f: med(k, f) for f in ("far_w", "dev", "dir")} for k in res[CELLS[0]]})
V["a_dev_lt"] = sum(res[c]["margin@0.05"]["dev"] < res[c]["pmax@0.05"]["dev"] for c in CELLS)
V["b_dir_gt_sch_1"] = sum(res[c]["margin@0.01"]["dir"] > res[c]["sch@0.01"]["dir"] for c in CELLS); V["b_dir_gt_sch_5"] = sum(res[c]["margin@0.05"]["dir"] > res[c]["sch@0.05"]["dir"] for c in CELLS)
a_ = V["a_dev_lt"] >= 19 and V["med"]["margin@0.05"]["dev"] < V["med"]["pmax@0.05"]["dev"]; b_ = V["b_dir_gt_sch_1"] >= 19 and V["b_dir_gt_sch_5"] >= 19
if SA:
    a_ = V["med"]["margin@0.05"]["dev"] <= .032; b_ = V["b_dir_gt_sch_1"] >= 19 and V["b_dir_gt_sch_5"] >= 19
V["SRC_a"], V["SRC_b"] = a_, b_; V["verdict"] = "both" if a_ and b_ else ("one" if a_ or b_ else "supplement")
print("SRC2" if SA else "SRC", json.dumps(V)); json.dump(dict(cells=res, verdict=V), open(os.path.join(O, "src2.json" if SA else "src.json"), "w"), indent=1)
