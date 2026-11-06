"""11-06 DG verdict (PREREG 'DECLAYER' (a)): Deep Gamblers reservation loss (attn_dg) vs SetMatch, 27 settings. The payoff
o in {1.5,2,4} is chosen per cell by in-sample train-eval strict accuracy (test-free), and that o's test metrics are
reported. Same CTRL rule (source-fixed threshold), strict accuracy and DIR at 1/5/10% FAR. Writes results/setmatch/dg.json."""
import json
import os
import sys

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "sm_ctrl.py")).read(); exec(src[:src.index("\nres, dropped = {}, {}")])   # noqa: S102 (own file: definitions only)
OS = ("1p5", "2", "4")
ctrl = json.load(open(os.path.join(O, "ctrl.json")))["cells"]; res = {}


def train_strict(x, c, E, grid):
    g1, g0 = strict_curves(x, c, E, grid); return float((0.5 * g1 + 0.5 * g0).max())


def dir_at_quant(x, c, E, K, a):
    return float(np.mean([((x[(K == k) & (E == 1)] > np.quantile(x[(K == k) & (E == 0)], 1 - a)) & (c[(K == k) & (E == 1)] > 0)).mean() for k in np.unique(K)]))


for key in CELLS:
    z = dict(np.load(os.path.join(V5, f"{key}_meta.npz"))); E, K = z["E"], z["K"]; best = None
    for ot in OS:
        f = os.path.join(V5, f"{key}_attn_dg_s10dg_o{ot}.npz")
        if not os.path.exists(f):
            continue
        y = dict(np.load(f)); S = [s for s in range(20) if f"conf_te_s{s}" in y]
        lg = lambda a: np.log(np.clip(a, 1e-7, 1 - 1e-7)) - np.log1p(-np.clip(a, 1e-7, 1 - 1e-7))
        tr = float(np.mean([train_strict(lg(y[f"conf_tr_s{s}"]), y[f"corr_tr_s{s}"], y["Etr"], np.unique(np.quantile(lg(y[f"conf_tr_s{s}"]), np.linspace(0, 1, 401)))) for s in S]))
        o = [evaluate(y[f"conf_tr_s{s}"], y[f"corr_tr_s{s}"], y["Etr"], y[f"conf_te_s{s}"], y[f"corr_te_s{s}"], E, K, y[f"ap_s{s}"], y[f"r1_s{s}"], "prob") for s in S]
        m = {fld: float(np.mean([x[fld] for x in o])) for fld in o[0]}
        for a in (0.01, 0.05, 0.1):
            m[f"dir{a}"] = float(np.mean([dir_at_quant(lg(y[f"conf_te_s{s}"]), y[f"corr_te_s{s}"], E, K, a) for s in S]))
        if best is None or tr > best[0]:
            best = (tr, ot, m)
    if best is None:
        print(key, "no DG runs"); continue
    res[key] = dict(o=best[1], dg=best[2], sm_acc=ctrl[key]["sm"]["acc_mean"], sch_acc=ctrl[key]["sch"]["acc_mean"])
    print(key.ljust(30), "o", best[1], json.dumps({k: round(best[2][k], 3) for k in ("acc_mean", "sel_worst", "dir0.01", "dir0.05", "dir0.1")}), flush=True)
K_ = list(res); d = np.array([res[k]["sm_acc"] - res[k]["dg"]["acc_mean"] for k in K_])
V = dict(n=len(K_), med_d=float(np.median(d)), sm_better=int((d >= .005).sum()), dg_better=int((d <= -.005).sum()),
         med_acc_dg=float(np.median([res[k]["dg"]["acc_mean"] for k in K_])), med_acc_sm=float(np.median([res[k]["sm_acc"] for k in K_])),
         med_dir1_dg=float(np.median([res[k]["dg"]["dir0.01"] for k in K_])))
V["verdict"] = "reservation loss below SetMatch" if V["med_d"] >= .003 and V["sm_better"] >= 15 else ("reservation loss matches" if V["med_d"] <= -.003 else "partial")
print("DG", json.dumps(V)); json.dump(dict(cells=res, verdict=V), open(os.path.join(O, "dg.json"), "w"), indent=1)
