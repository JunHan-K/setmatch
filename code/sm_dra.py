"""11-05 DRA (PREREG 'DRA', follow-up): on the same SetWiseMatch models and seeds (attn_s10dra, p(no-match) saved), the
accept score p_max vs the candidate/no-match margin log p_max - log p0, each with its own CTRL threshold (strict accuracy
on training episodes). Strict accuracy, DIR at 1/5/10% FAR, selective worst regret; and margin rule vs SCH accuracy.
Reuses sm_gv.py (rule). Writes results/setmatch/dra.json."""
import json
import os
import sys

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "sm_gv.py")).read(); exec(src[:src.index("\ndegen = ")])   # noqa: S102 (own file: definitions only)
lgt = lambda x: np.log(np.clip(x, 1e-9, 1.0))
ctrl = json.load(open(os.path.join(O, "ctrl.json")))["cells"]; res = {}
for key in CELLS:
    z = dict(np.load(os.path.join(V5, f"{key}_meta.npz"))); E, K = z["E"], z["K"]; y = dict(np.load(os.path.join(V5, f"{key}_attn{os.environ.get('DRA_TAG', '_s10dra')}.npz"))); S = seeds(y)
    ym = dict(y)
    for s in S:
        for nm in ("tr", "te"):
            ym[f"conf_{nm}_s{s}"] = lgt(y[f"conf_{nm}_s{s}"]) - lgt(y[f"null_{nm}_s{s}"])
    r = dict(pmax=rule(y, E, K, "prob"), margin=rule(ym, E, K, "logit"))
    for nm, yy, sp in (("pmax", y, "prob"), ("margin", ym, "logit")):
        r[nm]["dir0.01"] = float(np.mean([np.mean([dir_at(yy[f"conf_te_s{s}"][K == k], yy[f"corr_te_s{s}"][K == k], E[K == k], np.quantile(yy[f"conf_te_s{s}"][(K == k) & (E == 0)], .99))[0] for k in np.unique(K)]) for s in S]))
    r["sch_acc"] = ctrl[key]["sch"]["acc_mean"]; res[key] = r
    print(key.ljust(30), json.dumps({m: {f: round(r[m][f], 4) for f in ("acc_mean", "dir0.01", "dir0.05", "dir0.1", "sel_worst")} for m in ("pmax", "margin")}), flush=True)
d = np.array([res[k]["margin"]["acc_mean"] - res[k]["pmax"]["acc_mean"] for k in CELLS]); d1 = np.array([res[k]["margin"]["dir0.01"] - res[k]["pmax"]["dir0.01"] for k in CELLS])
md = lambda m, f: float(np.median([res[k][m][f] for k in CELLS]))
V = dict(med_d_acc=float(np.median(d)), margin_better=int((d > 0).sum()), med_d_dir1=float(np.median(d1)), dir1_better=int((d1 > 0).sum()),
         med={m: {f: md(m, f) for f in ("acc_mean", "dir0.01", "dir0.05", "dir0.1", "sel_worst", "far_max")} for m in ("pmax", "margin")},
         margin_vs_sch=float(np.median([res[k]["margin"]["acc_mean"] - res[k]["sch_acc"] for k in CELLS])), margin_gt_sch=sum(res[k]["margin"]["acc_mean"] > res[k]["sch_acc"] for k in CELLS),
         pmax_vs_sch=float(np.median([res[k]["pmax"]["acc_mean"] - res[k]["sch_acc"] for k in CELLS])))
V["DRA_a"] = V["med_d_acc"] >= .003 and V["margin_better"] >= 19; V["DRA_b"] = V["med_d_dir1"] > 0 and V["dir1_better"] >= 19
V["verdict"] = "no-match output improves the decision" if V["DRA_a"] else ("p_max already uses it" if V["med_d_acc"] <= 0 else "small/inconclusive")
print("DRA", json.dumps(V)); json.dump(dict(cells=res, verdict=V), open(os.path.join(O, "dra" + os.environ.get("DRA_OUT", "") + ".json"), "w"), indent=1)
