"""11-05 SWAP audit (reviewer: 'with the accept mask fixed, FAR must not change'). Tab. ctrl re-fits the threshold of
every hybrid on the training episodes (its strict accuracy depends on the choice), so its FAR differs between rows.
Here the pure mask swap: the threshold of rule A is fitted with A's own choice, the resulting accept mask is kept and
only the choice is exchanged. FAR must then be identical across the two choices (asserted). Writes swap.json."""
import json
import os

import numpy as np

SMT = os.environ.get("SM_TAG", "_s20")
H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "sm_ctrl.py")).read(); exec(src[:src.index("\nres, dropped = {}, {}")])   # noqa: S102 (own file: definitions only)
SMT = os.environ.get("SM_TAG", "_s20")   # 11-05 fix: sm_ctrl's definitions reset SMT to its 10-seed default


def mask_rule(xtr, ctr, Etr, xte, space):
    """accept mask on the test sets with the threshold fitted exactly as in evaluate (strict accuracy, own choice)"""
    if space == "prob":
        xtr, xte = lg(xtr), lg(xte)
    grid = np.unique(np.quantile(xtr, np.linspace(0, 1, 401))); g1, g0 = strict_curves(xtr, ctr, Etr, grid)
    return xte > grid[int(np.argmax(0.5 * g1 + 0.5 * g0))]


def score(acc, corr, E, K):
    a, far = [], []
    for k in np.unique(K):
        m = K == k; p, n = m & (E == 1), m & (E == 0)
        a.append(0.5 * (acc[p] & (corr[p] > 0)).mean() + 0.5 * (~acc[n]).mean()); far.append(acc[n].mean())
    return float(np.mean(a)), float(np.max(far)), far


res = {}
for key in CELLS:
    z = dict(np.load(os.path.join(V5, f"{key}_meta.npz"))); y = dict(np.load(os.path.join(V5, f"{key}_attn{SMT}.npz"))); E, K = z["E"], z["K"]
    S = [s for s in range(20) if f"conf_te_s{s}" in y]; o = {m: [] for m in ("gs", "gsmask_smchoice", "sm", "smmask_coschoice")}; farx = {m: [] for m in o}
    mg = mask_rule(z["sdse_tr"], z["Vtr"], z["Etr"], z["sdse_te"], "logit")
    for s in S:
        ms = mask_rule(y[f"conf_tr_s{s}"], y[f"corr_tr_s{s}"], y["Etr"], y[f"conf_te_s{s}"], "prob")
        for m, (mask, corr) in dict(gs=(mg, z["Vte"]), gsmask_smchoice=(mg, y[f"corr_te_s{s}"]), sm=(ms, y[f"corr_te_s{s}"]), smmask_coschoice=(ms, z["Vte"])).items():
            a, fm, fk = score(mask, corr, E, K); o[m].append(a); farx[m].append(fm)
        assert np.allclose(score(mg, z["Vte"], E, K)[2], score(mg, y[f"corr_te_s{s}"], E, K)[2]) and np.allclose(score(ms, y[f"corr_te_s{s}"], E, K)[2], score(ms, z["Vte"], E, K)[2]), key
    res[key] = {m: dict(acc=float(np.mean(o[m])), far_max=float(np.mean(farx[m]))) for m in o}
    print(key.ljust(30), json.dumps({m: {f: round(v, 3) for f, v in r.items()} for m, r in res[key].items()}), flush=True)
med = lambda m, f: float(np.median([res[k][m][f] for k in CELLS]))
d1 = [res[k]["gsmask_smchoice"]["acc"] - res[k]["gs"]["acc"] for k in CELLS]; d2 = [res[k]["sm"]["acc"] - res[k]["smmask_coschoice"]["acc"] for k in CELLS]
d3 = [res[k]["sm"]["acc"] - res[k]["gsmask_smchoice"]["acc"] for k in CELLS]
V = dict(summary={m: dict(acc=med(m, "acc"), far_max=med(m, "far_max")) for m in res[CELLS[0]]}, far_identical_asserted=True,
         choice_gain_under_gs_mask=float(np.median(d1)), choice_gain_under_sm_mask=float(np.median(d2)), sm_minus_gsmask_smchoice=float(np.median(d3)),
         sm_better_than_gsmask_smchoice=int(sum(x >= .005 for x in d3)), worse=int(sum(x <= -.005 for x in d3)))
print("SWAP", json.dumps(V)); json.dump(dict(cells=res, verdict=V), open(os.path.join(O, "swap.json"), "w"), indent=1)
