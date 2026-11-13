"""11-04 SCA (PREREG): size-conditional accuracy threshold as a baseline for every rule. On the train-eval episodes, per
training K the threshold maximising (a) strict accuracy at pi = .5 and (b) selective accuracy at pi = .5 is found on a
401-point grid and regressed on the mean log N of that K; each test episode gets a + b log N. Strict: frozen accuracy
(mean / worst K), DIR, worst-K FAR, gap to the per-K oracle. Selective: worst regret over K x pi in {.2,.5,.8}.
Rules: cosine, GS (logit), SetWiseMatch (logit, 10 seeds); SetWiseMatch with its constant threshold from ctrl.json for reference.
Writes results/setmatch/sca.json."""
import json
import os

import numpy as np

SMT = os.environ.get("SM_TAG", "_s10main")   # 11-04: "_s20" = all 20 seeds (main numbers), default = 10 seeds

H = os.path.dirname(os.path.abspath(__file__)); src = open(os.path.join(H, "sm_ctrl.py")).read(); exec(src[:src.index("\nres, dropped = {}, {}")])   # noqa: S102


def fit(x, c, E, K, N, grid, which):
    ts, ln = [], []
    for k in np.unique(K):
        m = K == k
        g1, g0 = strict_curves(x[m], c[m], E[m], grid) if which == "strict" else EV.acc_grid(x[m], c[m], E[m], grid)
        ts.append(grid[int(np.argmax(0.5 * g1 + 0.5 * g0))]); ln.append(np.log(N[m]).mean())
    return np.polyfit(ln, ts, 1) if len(ts) > 1 else (0.0, ts[0])


def run(xtr, ctr, Etr, Ktr, Ntr, xte, cte, E, K, N):
    grid = np.unique(np.quantile(xtr, np.linspace(0, 1, 401))); out = {}
    b1, b0 = fit(xtr, ctr, Etr, Ktr, Ntr, grid, "strict"); t = b0 + b1 * np.log(N); acc, orc, dr, far = [], [], [], []
    for k in np.unique(K):
        m = K == k; a = xte[m] > t[m]; e = E[m] == 1; ok = cte[m] > 0
        d_, f_ = (a & ok)[e].mean(), a[~e].mean(); acc.append(0.5 * d_ + 0.5 * (1 - f_)); dr.append(d_); far.append(f_)
        og = np.unique(np.r_[xte[m], np.nextafter(xte[m].min(), -np.inf)]); ho1, ho0 = strict_curves(xte[m], cte[m], E[m], og); orc.append((0.5 * ho1 + 0.5 * ho0).max())   # oracle on this cell's test boundaries (+ accept-all)
    out.update(acc_mean=float(np.mean(acc)), acc_worst=float(np.min(acc)), dir=float(np.mean(dr)), far_max=float(np.max(far)), gap=float(np.max(np.array(orc) - np.array(acc))))
    b1, b0 = fit(xtr, ctr, Etr, Ktr, Ntr, grid, "selective"); t = b0 + b1 * np.log(N); w = 0.0
    for k in np.unique(K):
        m = K == k; g1, g0 = EV.acc_grid(xte[m], cte[m], E[m], np.unique(np.r_[xte[m], np.nextafter(xte[m].min(), -np.inf)])); a = xte[m] > t[m]; e = E[m] == 1; ok = cte[m] > 0   # selective oracle on test boundaries (+ accept-all)
        a1 = (a == ok)[e].mean(); a0 = (~a)[~e].mean()
        for pi in (0.2, 0.5, 0.8):
            w = max(w, float(np.max(pi * g1 + (1 - pi) * g0) - (pi * a1 + (1 - pi) * a0)))
    out["sel_worst"] = w; return out


ctrl = json.load(open(os.path.join(O, "ctrl.json")))["cells"]; res = {}
for key in CELLS:
    d = key.split("_")[0]; q = "" if d in ("cuhk", "market") else "_q20"; z = np.load(os.path.join(V5, f"{key}_meta.npz")); E, K = z["E"], z["K"]
    ctr_ = np.load(os.path.join(O, "cache", f"{key}{q}_k{KT[d]}_tr.npz")); cte_ = np.load(os.path.join(O, "cache", f"{key}{q}_k{KT[d]}_te.npz"))
    assert (ctr_["E"] == z["Etr"]).all() and (cte_["E"] == E).all() and ((cte_["tm"][:, 0] > 0) == (z["Vte"] > 0)).mean() > 0.999, key
    Ktr, Ntr, N = ctr_["K"], ctr_["N"].astype(float), cte_["N"].astype(float); y = np.load(os.path.join(V5, f"{key}_attn{SMT}.npz")); S = [s for s in range(20) if f"conf_te_s{s}" in y.files]
    r = dict(cos=run(z["raw_tr"], z["Vtr"], z["Etr"], Ktr, Ntr, z["raw_te"], z["Vte"], E, K, N), gs=run(z["sdse_tr"], z["Vtr"], z["Etr"], Ktr, Ntr, z["sdse_te"], z["Vte"], E, K, N))
    sm = [run(lg(y[f"conf_tr_s{s}"]), y[f"corr_tr_s{s}"], z["Etr"], Ktr, Ntr, lg(y[f"conf_te_s{s}"]), y[f"corr_te_s{s}"], E, K, N) for s in S]
    r["sm"] = {f: float(np.mean([o[f] for o in sm])) for f in sm[0]}; r["sm_const"] = dict(acc_mean=ctrl[key]["sm"]["acc_mean"], acc_worst=ctrl[key]["sm"]["acc_worst"], sel_worst=ctrl[key]["sm"]["sel_worst"])
    res[key] = r; print(key.ljust(30), "  ".join(f"{m} acc {v['acc_mean']:.3f} sel {v['sel_worst']:.3f}" for m, v in r.items()), flush=True)
md = lambda m, f: float(np.median([res[k][m][f] for k in CELLS]))
Sm = {m: {f: md(m, f) for f in res[CELLS[0]][m]} for m in ("cos", "gs", "sm", "sm_const")}
Sm["smconst_gt_cossize"] = sum(res[k]["sm_const"]["acc_mean"] > res[k]["cos"]["acc_mean"] for k in CELLS)
Sm["smconst_gt_gssize"] = sum(res[k]["sm_const"]["acc_mean"] > res[k]["gs"]["acc_mean"] for k in CELLS)
Sm["smsize_gt_gssize"] = sum(res[k]["sm"]["acc_mean"] > res[k]["gs"]["acc_mean"] for k in CELLS)
Sm["sel_smconst_lt_cossize"] = sum(res[k]["sm_const"]["sel_worst"] < res[k]["cos"]["sel_worst"] for k in CELLS)
Sm["sel_smconst_lt_gssize"] = sum(res[k]["sm_const"]["sel_worst"] < res[k]["gs"]["sel_worst"] for k in CELLS)
Sm["verdict"] = "size-conditional thresholds do not replace SetWiseMatch" if Sm["smconst_gt_cossize"] >= 19 else "size-conditional cosine competes with SetWiseMatch"
print(json.dumps(Sm, indent=0)); json.dump(dict(cells=res, summary=Sm), open(os.path.join(O, "sca.json"), "w"), indent=1)
