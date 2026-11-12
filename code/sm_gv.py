"""11-04 verdicts of PREREG 'GV' and 'XM' (saved predictions only).
  gv : SetWiseMatch + set statistics (attn_gsc_s10gv) confirmed once on MSMT17 (CLIP, DINOv2), against GS and SetWiseMatch (attn_s20).
  xm : rules trained on MSMT17 and applied unchanged to Market-1501, CUHK-SYSU and PRW (CLIP, DINOv2): SetWiseMatch
       (attn_s10xm) vs GS and the cosine, both fitted on the same MSMT17 training episodes.
A cell is degenerate when the cosine, GS and SetWiseMatch frozen strict accuracies all lie within .500 +- .005 (every rule
rejects everything); it is reported but not counted. Reuses sm_ctrl.py (evaluate) and sm_mf.py (dir_at).
Writes results/setmatch/gv.json or xm.json."""
import json
import os
import sys

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
for f, stop in (("sm_ctrl.py", "\nres, dropped = {}, {}"), ("sm_mf.py", "\nres, curves = {}, {}")):
    src = open(os.path.join(H, f)).read(); exec(src[:src.index(stop)])   # noqa: S102 (own files: definitions only)
GEN2 = ("clip", "dinov2"); avg = lambda L: {f: float(np.mean([o[f] for o in L])) for f in L[0]}
seeds = lambda y: [s for s in range(20) if f"conf_te_s{s}" in y]


def rule(y, E, K, space):
    """strict metrics and DIR at 5% / 10% FAR averaged over the seeds of one saved prediction file"""
    S = seeds(y); m = avg([evaluate(y[f"conf_tr_s{s}"], y[f"corr_tr_s{s}"], y["Etr"], y[f"conf_te_s{s}"], y[f"corr_te_s{s}"], E, K, y[f"ap_s{s}"], y[f"r1_s{s}"], space) for s in S])
    for a in (0.05, 0.1):
        m[f"dir{a}"] = float(np.mean([np.mean([dir_at(y[f"conf_te_s{s}"][K == k], y[f"corr_te_s{s}"][K == k], E[K == k], np.quantile(y[f"conf_te_s{s}"][(K == k) & (E == 0)], 1 - a))[0] for k in np.unique(K)]) for s in S]))
    m["n_seeds"] = len(S); return m


def meta_rule(z, base, space):
    return rule(dict(conf_tr_s0=z[f"{base}_tr"], corr_tr_s0=z["Vtr"], Etr=z["Etr"], conf_te_s0=z[f"{base}_te"], corr_te_s0=z["Vte"], ap_s0=z["cos_ap"], r1_s0=z["cos_r1"]), z["E"], z["K"], space)


degen = lambda c: all(abs(c[m]["acc_mean"] - .5) <= .005 for m in ("cos", "gs", "sm"))
mode = sys.argv[1]; res = {}
if mode == "gv":
    for e in GEN2:
        z = dict(np.load(os.path.join(V5, f"msmt_{e}_meta.npz"))); E, K = z["E"], z["K"]
        res[e] = dict(cos=meta_rule(z, "raw", None), gs=meta_rule(z, "sdse", "logit"), sm=rule(dict(np.load(os.path.join(V5, f"msmt_{e}_attn_s20.npz"))), E, K, "prob"),
                      gsc=rule(dict(np.load(os.path.join(V5, f"msmt_{e}_attn_gsc_s10gv.npz"))), E, K, "prob"))
        print(e, json.dumps({m: {f: round(v, 3) for f, v in r.items()} for m, r in res[e].items()}), flush=True)
    live = [e for e in GEN2 if not degen(res[e])]; c = lambda e: res[e]
    crit = {e: dict(a_acc_gt_gs=c(e)["gsc"]["acc_mean"] > c(e)["gs"]["acc_mean"], b_dir5_gt_gs=c(e)["gsc"]["dir0.05"] > c(e)["gs"]["dir0.05"],
                    c_dir10_gt_gs=c(e)["gsc"]["dir0.1"] > c(e)["gs"]["dir0.1"], d_sel_le_sm=c(e)["gsc"]["sel_worst"] <= c(e)["sm"]["sel_worst"],
                    e_acc_ge_sm=c(e)["gsc"]["acc_mean"] >= c(e)["sm"]["acc_mean"] - .005) for e in live}
    V = dict(live=live, degenerate=[e for e in GEN2 if e not in live], criteria=crit)
    V["verdict"] = "confirmed on unseen data" if live and all(all(v.values()) for v in crit.values()) else "not confirmed (stays exploratory)"
elif mode == "extm":                                               # 11-05 PREREG 'EXT-M': margin rule on external data
    lgt = lambda x: np.log(np.clip(x, 1e-9, 1.0))
    def marg(y):
        y = dict(y)
        for s in seeds(y):
            for nm in ("tr", "te"):
                y[f"conf_{nm}_s{s}"] = lgt(y[f"conf_{nm}_s{s}"]) - lgt(y[f"null_{nm}_s{s}"])
        return y
    def d1(y, E, K):
        return float(np.mean([np.mean([dir_at(y[f"conf_te_s{s}"][K == k], y[f"corr_te_s{s}"][K == k], E[K == k], np.quantile(y[f"conf_te_s{s}"][(K == k) & (E == 0)], .99))[0] for k in np.unique(K)]) for s in seeds(y)]))
    ext = json.load(open(os.path.join(O, "ext.json"))); live = ext["verdict"]["live"]
    for k in live:
        if k.startswith("msmt_"):
            e = k.split("_", 1)[1]; z = dict(np.load(os.path.join(V5, f"msmt_{e}_meta.npz"))); E, K = z["E"], z["K"]
            y = dict(np.load(os.path.join(V5, f"msmt_{e}_attn_s10dra.npz"))); ys = dict(np.load(os.path.join(V5, f"msmt_{e}_selch_s10sch.npz")))
        else:
            t, e = k.split("_")[0], k.split("_")[1]; ld = lambda v: dict(np.load(os.path.join(V5, f"{t}_{e}_{v}_xmsmt_src.npz")))
            y = ld("attn_s10xmdra"); ys = ld("selch_s10ext"); E, K = y["E"], y["K"]
        ym = marg(y); r = dict(pmax=rule(y, E, K, "prob"), margin=rule(ym, E, K, "logit"), sch=rule(ys, E, K, "prob"))
        for m, yy in (("pmax", y), ("margin", ym), ("sch", ys)):
            r[m]["dir0.01"] = d1(yy, E, K)
        res[k] = r; print(k, json.dumps({m: {f: round(r[m][f], 3) for f in ("acc_mean", "dir0.01", "dir0.05")} for m in r}), flush=True)
    n = len(live); need = int(np.ceil(4 * n / 5))
    a_ = sum(res[k]["margin"]["dir0.01"] > res[k]["pmax"]["dir0.01"] for k in live); b_ = sum(res[k]["margin"]["dir0.01"] > res[k]["sch"]["dir0.01"] for k in live)
    c_ = sum(res[k]["margin"]["acc_mean"] > res[k]["sch"]["acc_mean"] for k in live)
    V = dict(live=live, n=n, need=need, EXTMa_dir1_gt_pmax=a_, EXTMb_dir1_gt_sch=b_, acc_gt_sch=c_,
             med={m: {f: float(np.median([res[k][m][f] for k in live])) for f in ("acc_mean", "dir0.01", "dir0.05")} for m in ("pmax", "margin", "sch")})
    V["verdict"] = "margin rule holds on external data" if a_ >= need and b_ >= need else ("partial" if a_ >= need else "not held")
elif mode == "ext":                                                # 11-05 PREREG 'EXT': strongest baselines on external data
    def fill(y, ref):                                              # GS-MLP keeps the cosine top-1: borrow its AP / R1
        for s in seeds(y):
            y.setdefault(f"ap_s{s}", ref[0]); y.setdefault(f"r1_s{s}", ref[1])
        return y
    for e in GEN2:
        z = dict(np.load(os.path.join(V5, f"msmt_{e}_meta.npz"))); E, K = z["E"], z["K"]; L = lambda v: dict(np.load(os.path.join(V5, f"msmt_{e}_{v}.npz")))
        res[f"msmt_{e}"] = dict(cos=meta_rule(z, "raw", None), gs=meta_rule(z, "sdse", "logit"), sm=rule(L("attn_s20"), E, K, "prob"),
                                gs_mlp=rule(fill(L("gsmlp_s10ng"), (z["cos_ap"], z["cos_r1"])), E, K, "prob"), sch=rule(L("selch_s10sch"), E, K, "prob"))
    for t in ("market", "cuhk", "prw"):
        for e in GEN2:
            ld = lambda v: dict(np.load(os.path.join(V5, f"{t}_{e}_{v}_xmsmt_src.npz"))); sm = ld("attn_s10xm"); E, K = sm["E"], sm["K"]; c0 = ld("cos")
            res[f"{t}_{e}_xmsmt"] = dict(cos=rule(c0, E, K, None), gs=rule(ld("gs"), E, K, "prob"), sm=rule(sm, E, K, "prob"),
                                         gs_mlp=rule(fill(ld("gsmlp_s10ext"), (c0["ap_s0"], c0["r1_s0"])), E, K, "prob"), sch=rule(ld("selch_s10ext"), E, K, "prob"))
    for k, r in res.items():
        print(k, json.dumps({m: {f: round(v, 3) for f, v in x.items()} for m, x in r.items()}), flush=True)
    live = [k for k in res if not degen(res[k])]; n = len(live); need = int(np.ceil(2 * n / 3))
    a_ = sum(res[k]["sm"]["acc_mean"] > res[k]["sch"]["acc_mean"] for k in live); b_ = sum(res[k]["sm"]["acc_mean"] > res[k]["gs_mlp"]["acc_mean"] for k in live)
    md = lambda m: float(np.median([res[k]["sm"]["acc_mean"] - res[k][m]["acc_mean"] for k in live])) if live else None
    V = dict(live=live, n=n, need=need, EXTa_sm_gt_sch=a_, EXTb_sm_gt_gsmlp=b_, med_d_sch=md("sch"), med_d_gsmlp=md("gs_mlp"), med_d_gs=md("gs"))
    V["verdict"] = "holds externally" if n and min(a_, b_) >= need else ("not held externally" if a_ <= n // 3 else "partial")
else:
    for t in ("market", "cuhk", "prw"):
        for e in GEN2:
            ld = lambda v: dict(np.load(os.path.join(V5, f"{t}_{e}_{v}_xmsmt_src.npz"))); sm = ld("attn_s10xm"); E, K = sm["E"], sm["K"]
            r = dict(cos=rule(ld("cos"), E, K, None), gs=rule(ld("gs"), E, K, "prob"), sm=rule(sm, E, K, "prob"))
            ind = {"market": f"market_{e}", "cuhk": f"cuhk_{e}", "prw": f"prw_{e}"}[t]; ctrl = json.load(open(os.path.join(O, "ctrl.json")))["cells"].get(ind, {})
            r["ref_in_domain"] = {m: ctrl[m]["acc_mean"] for m in ("gs", "sm") if m in ctrl}; res[f"{t}_{e}"] = r
            print(f"{t}_{e}", json.dumps({m: ({f: round(v, 3) for f, v in x.items()}) for m, x in r.items()}), flush=True)
    live = [k for k in res if not degen(res[k])]
    a_ = sum(res[k]["sm"]["acc_mean"] > res[k]["gs"]["acc_mean"] for k in live); b_ = sum(res[k]["sm"]["acc_mean"] > res[k]["cos"]["acc_mean"] for k in live)
    c_ = sum(res[k]["sm"]["dir0.05"] > res[k]["gs"]["dir0.05"] for k in live); n = len(live); need = int(np.ceil(2 * n / 3))
    V = dict(live=live, n=n, XMa_acc_gt_gs=a_, XMb_acc_gt_cos=b_, XMc_dir5_gt_gs=c_, need=need)
    V["verdict"] = "transfers" if n and min(a_, b_, c_) >= need else ("does not transfer" if a_ <= n // 3 else "partially transfers")
print(mode, json.dumps(V)); json.dump(dict(cells=res, verdict=V), open(os.path.join(O, f"{mode}.json"), "w"), indent=1)
