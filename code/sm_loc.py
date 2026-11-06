"""11-05 LOC + RISK (PREREG 'LOC', 'RISK'; saved predictions only, one run).
LOC  (reviewer: "is the extra gain over the strongest two-stage system specific to candidate-set shift?"): per setting,
     frozen-threshold strict accuracy (CTRL rule, pi = .5) averaged over the test sizes inside the training size range
     (K <= largest training gallery) and over the extrapolated sizes (K > it), for SetMatch, SCH (frozen selector +
     correctness head), GS-MLP and set pooling. D_shift = (SM - SCH)_extrapolated - (SM - SCH)_in-range, with an
     identity-cluster bootstrap (1000; all queries of an identity and all sizes of a query move together, seeds resampled).
     Market-1501 is excluded (its training sizes include the full gallery: no extrapolated test size).
RISK (reviewer: "is SetMatch's accuracy just a riskier operating point?"): for GS, GS-MLP, SCH, set pooling, SetMatch:
     DIR at matched test FAR 1/5/10% (per-size test quantile), present-query misidentification FIR = P(accept and wrong |
     present) and FAR at the frozen accuracy threshold, a source-fixed 5% FAR target (threshold from training absent
     episodes: worst-size test FAR, mean DIR), and the utility U = pi (DIR - c_mis FIR) - (1 - pi) c_fa FAR with its
     threshold fitted on training episodes, for pi in {.2,.5,.8} x c_mis in {1,3} x c_fa in {1,3} (12 fixed configurations).
Writes results/setmatch/loc.json and risk.json."""
import json
import os
import sys

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "sm_swap.py")).read(); exec(src[src.index("\nSMT = "):src.index("\nres = {}")])   # noqa: S102 (own files: definitions only)
sys.path.insert(0, H)
import sm_v5 as V5M
from sm_v5 import TRK
src = open(os.path.join(H, "sm_idb.py")).read(); exec(src[src.index("\nB = 1000"):src.index("\nres = {}")])   # noqa: S102

mode = sys.argv[1]
lg = lambda x: np.log(np.clip(x, 1e-7, 1 - 1e-7)) - np.log1p(-np.clip(x, 1e-7, 1 - 1e-7))


def runs(key, z):
    """method -> list over seeds of (x_tr, c_tr, E_tr, x_te, c_te) in logit space"""
    L = lambda v: dict(np.load(os.path.join(V5, f"{key}_{v}.npz"))); out = {"gs": [(z["sdse_tr"], z["Vtr"], z["Etr"], z["sdse_te"], z["Vte"])]}
    for m, v, own in (("gs_mlp", "gsmlp_s10ng", False), ("sch", "selch_s10sch", True), ("ctx", "ctx" + SMT, True), ("sm", "attn" + SMT, True)):
        y = L(v); S = [s for s in range(20) if f"conf_te_s{s}" in y]
        out[m] = [(lg(y[f"conf_tr_s{s}"]), y[f"corr_tr_s{s}"] if own else z["Vtr"], y["Etr"], lg(y[f"conf_te_s{s}"]), y[f"corr_te_s{s}"] if own else z["Vte"]) for s in S]
    return out


def wacc_k(acc, corr, E, K, W, ks):
    out = []
    for k in ks:
        m = K == k; p, n = m & (E == 1), m & (E == 0)
        out.append(0.5 * (W[:, p] @ (acc[p] & (corr[p] > 0)).astype(np.float32)) / W[:, p].sum(1) + 0.5 * (W[:, n] @ (~acc[n]).astype(np.float32)) / W[:, n].sum(1))
    return np.mean(out, 0)


def fit_u(x, c, E, pi, cm, cf):
    grid = np.unique(np.quantile(x, np.linspace(0, 1, 401))); p, n = E == 1, E == 0
    a = x[None, :] > grid[:, None]; d = (a[:, p] & (c[p] > 0)[None]).mean(1); f = (a[:, p] & (c[p] <= 0)[None]).mean(1); fa = a[:, n].mean(1)
    return grid[int(np.argmax(pi * (d - cm * f) - (1 - pi) * cf * fa))]


def op(x, c, E, K, t):
    """per-size DIR, FIR, FAR at threshold t, averaged over sizes (FAR also worst)"""
    D, F, A = [], [], []
    for k in np.unique(K):
        m = K == k; p, n = m & (E == 1), m & (E == 0); a = x > t
        D.append((a[p] & (c[p] > 0)).mean()); F.append((a[p] & (c[p] <= 0)).mean()); A.append(a[n].mean())
    return float(np.mean(D)), float(np.mean(F)), float(np.mean(A)), float(np.max(A))


res = {}
CF = [(pi, cm, cf) for pi in ((.8, .9) if mode == "search" else (.2, .5, .8)) for cm in (1, 3) for cf in (1, 3)]
if mode == "fm":                                                   # 11-05 FM: general-purpose encoders, aggregation of ctrl.json / risk.json
    ct, rk = json.load(open(os.path.join(O, "ctrl.json")))["cells"], json.load(open(os.path.join(O, "risk.json")))["cells"]
    GK = [k for k in CELLS if k.split("_", 1)[1] in ("dinov2", "clip")]
    res = {k: dict(acc={m: ct[k][c]["acc_mean"] for m, c in (("gs", "gs"), ("gs_mlp", "gs-mlp"), ("sch", "sch"), ("sm", "sm"))},
                   **{f"dir{a}": {m: rk[k][m][f"dir{a}"] for m in ("gs", "gs_mlp", "sch", "sm")} for a in (0.01, 0.05, 0.1)}) for k in GK}
    med = lambda f, m: float(np.median([res[k][f][m] for k in GK]))
    V = {f: {m: med(f, m) for m in ("gs", "gs_mlp", "sch", "sm")} for f in ("acc", "dir0.01", "dir0.05", "dir0.1")}
    V.update(acc_gt_sch=sum(res[k]["acc"]["sm"] > res[k]["acc"]["sch"] for k in GK), acc_gt_gsmlp=sum(res[k]["acc"]["sm"] > res[k]["acc"]["gs_mlp"] for k in GK),
             dir5_gt_sch=sum(res[k]["dir0.05"]["sm"] > res[k]["dir0.05"]["sch"] for k in GK), dir10_gt_sch=sum(res[k]["dir0.1"]["sm"] > res[k]["dir0.1"]["sch"] for k in GK), n=len(GK))
    V["verdict"] = "SetMatch strongest for general-purpose encoders" if V["acc_gt_sch"] >= 8 and V["dir5_gt_sch"] >= 7 else ("accuracy only" if V["acc_gt_sch"] >= 8 else "not shown")
    print("FM", json.dumps(V)); json.dump(dict(cells=res, verdict=V), open(os.path.join(O, "fm.json"), "w"), indent=1); sys.exit()
for key in CELLS:
    d = key.split("_")[0]; z = dict(np.load(os.path.join(V5, f"{key}_meta.npz"))); E, K = z["E"], z["K"]; R_ = runs(key, z)
    if mode in ("loc", "loc2"):
        if 0 in TRK[d]:
            continue
        kin = [k for k in np.unique(K) if k <= max(TRK[d])]; kex = [k for k in np.unique(K) if k > max(TRK[d])]
        pid, c = test_pids(key); assert (c["E"] == E).all(), key
        _, g = np.unique(pid, return_inverse=True); W = rng.multinomial(g.max() + 1, np.full(g.max() + 1, 1 / (g.max() + 1)), size=B).astype(np.float32)[:, g]
        r = dict(k_in=[int(k) for k in kin], k_ex=[int(k) for k in kex])
        LM = ("sch", "gs_mlp", "ctx", "sm") + (("gs",) if mode == "loc2" else ())
        for m in LM:
            per = []
            for xtr, ctr, Etr, xte, cte in R_[m]:
                acc = mask_rule(xtr, ctr, Etr, xte, "logit"); per.append((wacc_k(acc, cte, E, K, W, kin), wacc_k(acc, cte, E, K, W, kex)))
            per = np.array(per); pick = rng.integers(0, len(per), size=(B, len(per))); r[m] = per[pick, :, np.arange(B)[:, None]].mean(1)   # B x 2
        out = {}
        for b_ in [x for x in LM if x != "sm"]:
            dd = r["sm"] - r[b_]; ds = dd[:, 1] - dd[:, 0]
            out[b_] = dict(d_in=float(np.median(dd[:, 0])), d_ex=float(np.median(dd[:, 1])), d_shift=float(np.median(ds)), lo=float(np.quantile(ds, .025)), hi=float(np.quantile(ds, .975)))
        res[key] = dict(k_in=r["k_in"], k_ex=r["k_ex"], acc_in={m: float(np.median(r[m][:, 0])) for m in LM},
                        acc_ex={m: float(np.median(r[m][:, 1])) for m in LM}, **out)
    else:
        r = {}
        if mode == "search":                                       # 11-05 SEARCH: largest test gallery only
            kx = K == K.max(); E, K = E[kx], K[kx]
            R_ = {m: [(a_, b_, c_, d_[kx], e_[kx]) for a_, b_, c_, d_, e_ in L_] for m, L_ in R_.items()}
        for m, L_ in R_.items():
            o = []
            for xtr, ctr, Etr, xte, cte in L_:
                acc = mask_rule(xtr, ctr, Etr, xte, "logit")   # frozen strict-accuracy threshold (CTRL rule)
                Dm, Fm, Am, Aw = [], [], [], []
                for k in np.unique(K):
                    mk = K == k; p, n = mk & (E == 1), mk & (E == 0); Dm.append((acc[p] & (cte[p] > 0)).mean()); Fm.append((acc[p] & (cte[p] <= 0)).mean()); Am.append(acc[n].mean())
                q = dict(dir_acc=float(np.mean(Dm)), fir_acc=float(np.mean(Fm)), far_acc=float(np.mean(Am)), farw_acc=float(np.max(Am)))
                for a in (.01, .05, .1):
                    q[f"dir{a}"] = float(np.mean([((xte[(K == k) & (E == 1)] > np.quantile(xte[(K == k) & (E == 0)], 1 - a)) & (cte[(K == k) & (E == 1)] > 0)).mean() for k in np.unique(K)]))
                t5 = np.quantile(xtr[Etr == 0], .95); dd_, ff_, aa_, aw_ = op(xte, cte, E, K, t5); q.update(src_dir=dd_, src_farw=aw_)
                for pi, cm, cf in CF:
                    t = fit_u(xtr, ctr, Etr, pi, cm, cf); dd_, ff_, aa_, _ = op(xte, cte, E, K, t); q[f"U{pi}_{cm}_{cf}"] = float(pi * (dd_ - cm * ff_) - (1 - pi) * cf * aa_)
                o.append(q)
            r[m] = {f: float(np.mean([x[f] for x in o])) for f in o[0]}
        res[key] = r
    print(key.ljust(30), json.dumps(res[key], default=lambda v: round(v, 4)), flush=True)
K_ = list(res)
med = lambda f: float(np.median([f(res[k]) for k in K_]))
if mode == "loc2":
    v = {f: med(lambda r: r["gs"][f]) for f in ("d_in", "d_ex", "d_shift")}; V = dict(gs=dict(med_d_in=v["d_in"], med_d_ex=v["d_ex"], med_d_shift=v["d_shift"],
             pos=sum(res[k]["gs"]["d_shift"] > 0 for k in K_), ci_above=sum(res[k]["gs"]["lo"] > 0 for k in K_), ci_below=sum(res[k]["gs"]["hi"] < 0 for k in K_), n=len(K_)))
    V["verdict"] = "gain over calibration grows under extrapolation" if V["gs"]["med_d_shift"] > 0 and V["gs"]["pos"] >= 15 else ("supplement" if V["gs"]["med_d_shift"] <= 0 else "partial")
elif mode == "loc":
    V = {b_: dict(med_d_in=med(lambda r: r[b_]["d_in"]), med_d_ex=med(lambda r: r[b_]["d_ex"]), med_d_shift=med(lambda r: r[b_]["d_shift"]),
                  pos=sum(res[k][b_]["d_shift"] > 0 for k in K_), ci_above=sum(res[k][b_]["lo"] > 0 for k in K_), ci_below=sum(res[k][b_]["hi"] < 0 for k in K_), n=len(K_)) for b_ in ("sch", "gs_mlp", "ctx")}
    v = V["sch"]; V["verdict"] = "shift-concentrated" if v["med_d_shift"] > 0 and v["pos"] >= 15 else ("not shift-specific" if v["med_d_shift"] <= 0 else "partial")
else:
    M_ = ("gs", "gs_mlp", "sch", "ctx", "sm"); F_ = list(res[K_[0]]["sm"])
    V = dict(medians={m: {f: med(lambda r: r[m][f]) for f in F_} for m in M_})
    Uk = [f for f in F_ if f.startswith("U")]
    V["sm_best_U"] = sum(max(M_, key=lambda m: V["medians"][m][f]) == "sm" for f in Uk); V["n_U"] = len(Uk)
    V["sm_vs_sch_U_cells"] = {f: sum(res[k]["sm"][f] > res[k]["sch"][f] for k in K_) for f in Uk}
    if mode == "search":
        V["verdict"] = "SetMatch is the rule for large-archive search" if V["sm_best_U"] >= 6 else ("supplement" if V["sm_best_U"] <= 3 else "one sentence")
    else:
        V["verdict"] = "benefit beyond pi=.5 accuracy" if V["sm_best_U"] >= 8 else ("limited to the accuracy-optimal point" if V["sm_best_U"] <= 4 else "mixed")
print(mode.upper(), json.dumps(V)); json.dump(dict(cells=res, verdict=V), open(os.path.join(O, f"{mode}.json"), "w"), indent=1)
