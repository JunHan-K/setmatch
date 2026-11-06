"""11-05 SCH verdict (PREREG 'SCH'): SetMatch (attn_s20) vs frozen selector + correctness head (selch_s10sch), 27 settings,
metrics of sm_gv.rule. Writes results/setmatch/sch.json."""
import json
import os

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "sm_gv.py")).read(); exec(src[src.index("\nH = "):src.index("\ndegen = ")])   # noqa: S102 (own file: definitions only)
fam = lambda k: "reid" if k.split("_", 1)[1] in ENC[:3] else ("generic" if k.split("_", 1)[1] in ENC[3:] else "seqnet")
L = lambda f: dict(np.load(os.path.join(V5, f))); med = lambda x: float(np.median(x)); res = {}
for key in CELLS:
    z = L(f"{key}_meta.npz"); E, K = z["E"], z["K"]
    res[key] = dict(gs=meta_rule(z, "sdse", "logit"), sm=rule(L(f"{key}_attn_s20.npz"), E, K, "prob"), sch=rule(L(f"{key}_selch_s10sch.npz"), E, K, "prob"))
    print(key.ljust(30), json.dumps({m: {f: round(r[f], 3) for f in ("acc_mean", "sel_worst", "dir0.05", "dir0.1")} for m, r in res[key].items()}), flush=True)
d = np.array([res[k]["sm"]["acc_mean"] - res[k]["sch"]["acc_mean"] for k in CELLS])
V = dict(med_delta_acc=med(d), sm_better=int((d >= .005).sum()), sch_better=int((d <= -.005).sum()),
         sch_gt_gs=sum(res[k]["sch"]["acc_mean"] > res[k]["gs"]["acc_mean"] for k in CELLS),
         med_acc={m: med([res[k][m]["acc_mean"] for k in CELLS]) for m in ("gs", "sch", "sm")}, med_sel={m: med([res[k][m]["sel_worst"] for k in CELLS]) for m in ("gs", "sch", "sm")})
for a in ("0.05", "0.1"):
    for f in ("reid", "generic"):
        ks = [k for k in CELLS if fam(k) == f]; V[f"dir{a}_{f}"] = {m: med([res[k][m][f"dir{a}"] for k in ks]) for m in ("gs", "sch", "sm")}
V["verdict"] = "joint distribution adds beyond selection-aware correctness learning" if V["med_delta_acc"] >= .005 and V["sm_better"] >= 19 else (
    "explained by selection-aware correctness learning" if V["med_delta_acc"] <= 0 else "partial")
print("SCH", json.dumps(V)); json.dump(dict(cells=res, verdict=V), open(os.path.join(O, "sch.json"), "w"), indent=1)
