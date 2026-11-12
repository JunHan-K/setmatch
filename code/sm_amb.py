"""11-05 AMB (PREREG 'AMB'): does acceptance reflect ambiguity among the candidates? Present test sets are split into
tertiles of candidate ambiguity, measured by an independent model: the maximum candidate posterior of the separately
trained selector (attn_sel_s5ds, mean over its seeds; no-match blocked). In each tertile, false identification rate
FIR = P(accept and the chosen candidate is wrong | present) at the frozen thresholds of Tab. ctrl (sm_swap.mask_rule),
for GS (cosine top-1) and SetWiseMatch (its own choice, mean over seeds); also DIR = P(accept and correct | present).
Writes results/setmatch/amb.json."""
import json
import os

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "sm_swap.py")).read(); exec(src[src.index("\nSMT = "):src.index("\nres = {}")])   # noqa: S102 (own file: definitions only)
res = {}
for key in CELLS:
    z = dict(np.load(os.path.join(V5, f"{key}_meta.npz"))); y = dict(np.load(os.path.join(V5, f"{key}_attn{SMT}.npz"))); ds = dict(np.load(os.path.join(V5, f"{key}_attn_sel_s5ds.npz")))
    E = z["E"]; P = E == 1; amb = np.mean([ds[f"conf_te_s{s}"] for s in range(20) if f"conf_te_s{s}" in ds], 0)
    cut = np.quantile(amb[P], [1 / 3, 2 / 3]); T = np.digitize(amb, cut)          # 0 = most ambiguous (lowest max posterior)
    mg = mask_rule(z["sdse_tr"], z["Vtr"], z["Etr"], z["sdse_te"], "logit"); S = [s for s in range(20) if f"conf_te_s{s}" in y]
    ms = [mask_rule(y[f"conf_tr_s{s}"], y[f"corr_tr_s{s}"], y["Etr"], y[f"conf_te_s{s}"], "prob") for s in S]
    r = {}
    for t in range(3):
        m = P & (T == t); cg = z["Vte"][m] > 0
        r[t] = dict(fir_gs=float((mg[m] & ~cg).mean()), dir_gs=float((mg[m] & cg).mean()),
                    fir_sm=float(np.mean([(a[m] & ~(y[f"corr_te_s{s}"][m] > 0)).mean() for a, s in zip(ms, S)])),
                    dir_sm=float(np.mean([(a[m] & (y[f"corr_te_s{s}"][m] > 0)).mean() for a, s in zip(ms, S)])), n=int(m.sum()))
    res[key] = r; print(key.ljust(30), json.dumps({t: {f: round(v, 3) for f, v in x.items()} for t, x in r.items()}), flush=True)
red = lambda k, t: res[k][t]["fir_gs"] - res[k][t]["fir_sm"]          # FIR reduction by SetWiseMatch in tertile t
did = [red(k, 0) - red(k, 2) for k in CELLS]
V = dict(med_fir={t: {m: float(np.median([res[k][t][f"fir_{m}"] for k in CELLS])) for m in ("gs", "sm")} for t in range(3)},
         med_dir={t: {m: float(np.median([res[k][t][f"dir_{m}"] for k in CELLS])) for m in ("gs", "sm")} for t in range(3)},
         med_reduction_ambiguous=float(np.median([red(k, 0) for k in CELLS])), med_reduction_clear=float(np.median([red(k, 2) for k in CELLS])),
         did_pos=int(sum(d > 0 for d in did)), med_did=float(np.median(did)),
         dir_ambiguous_sm_ge_gs=int(sum(res[k][0]["dir_sm"] >= res[k][0]["dir_gs"] for k in CELLS)))
V["verdict"] = "acceptance reflects candidate ambiguity" if V["med_reduction_ambiguous"] > 0 and V["did_pos"] >= 19 else "not shown"
print("AMB", json.dumps(V)); json.dump(dict(cells=res, verdict=V), open(os.path.join(O, "amb.json"), "w"), indent=1)
