"""11-04 SEL / GSF (PREREG '사용자 결정 + CPU 분석 사전등록'; no training).
SEL: forced-choice candidate re-selection of SetWiseMatch (S10, 10 seeds) vs the cosine top-1 that GS also returns, on
target-present test episodes: Rescue / Harm / both correct / both wrong, split by whether the target is in the encoded
top-64; net correction rate NCR = P(Rescue) - P(Harm) (target in top-64) with a query-level cluster bootstrap 95% CI;
top-1 and mAP change. GSF: across the 27 cells, x = GS worst regret, y = GS - SetWiseMatch worst regret (S10), Spearman rho
with a one-sided permutation p; figure new_paper/figs/gs_gap.pdf. Writes results/setmatch/sel.json."""
import json
import os

import numpy as np

SMT = os.environ.get("SM_TAG", "_s10main")   # 11-04: "_s20" = all 20 seeds (main numbers), default = 10 seeds
from scipy.stats import spearmanr

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")); O = os.path.join(R, "results/setmatch"); V5 = os.path.join(O, "v5")
ENC = ("osnet_ain_x1_0_msmt17", "transreidssl_msmt", "solider_msmt", "dinov2", "clip")
CELLS = [f"{d}_{e}" for d in ("cuhk", "prw", "cuhkdet", "prwdet", "market") for e in ENC] + ["cuhkseq_seqnet", "prwseq_seqnet"]
KT = {"market": "10-20-50-100-200-0", "cuhk": "50-100-200-500", "prw": "50-100-200-500-1000", "cuhkdet": "50-100-200-500", "prwdet": "50-100-200-500-1000", "cuhkseq": "50-100-200-500", "prwseq": "50-100-200-500-1000"}
B = 1000; rng = np.random.default_rng(0); out = {}
for key in CELLS:
    d = key.split("_")[0]; z = np.load(os.path.join(V5, f"{key}_meta.npz")); E = z["E"]
    c = np.load(os.path.join(O, "cache", f"{key}{'' if d in ('cuhk', 'market') else '_q20'}_k{KT[d]}_te.npz"))
    assert (c["E"] == E).all() and ((c["tm"][:, 0] > 0) == (z["Vte"] > 0)).mean() > 0.999, key
    y = np.load(os.path.join(V5, f"{key}_attn{SMT}.npz")); S = [s for s in range(20) if f"corr_te_s{s}" in y.files]
    p = E == 1; inM = c["tm"].any(1); cos1 = c["tm"][:, 0] > 0
    sm = np.mean([y[f"corr_te_s{s}"] > 0 for s in S], 0)                       # per-episode share of seeds whose pick is correct
    rescue, harm = (~cos1) * sm, cos1 * (1 - sm); v = (rescue - harm)
    ap_sm = np.mean([y[f"ap_s{s}"] for s in S], 0); m = p & inM
    _, qi = np.unique(c["qi"], return_inverse=True); W = rng.multinomial(qi.max() + 1, np.full(qi.max() + 1, 1 / (qi.max() + 1)), size=B).astype(np.float32)[:, qi[m]]
    ncr_b = (W * v[m]).sum(1) / W.sum(1)
    r = dict(n_present=int(p.sum()), share_in_top64=float(inM[p].mean()),
             in64=dict(rescue=float(rescue[m].mean()), harm=float(harm[m].mean()), both_correct=float((cos1 * sm)[m].mean()), both_wrong=float(((~cos1) * (1 - sm))[m].mean())),
             out64=dict(n=int((p & ~inM).sum()), sm_correct=float(sm[p & ~inM].mean()) if (p & ~inM).any() else None),
             ncr=float(v[m].mean()), ncr_ci=[float(np.quantile(ncr_b, .025)), float(np.quantile(ncr_b, .975))],
             top1_cos=float(cos1[p].mean()), top1_sm=float(sm[p].mean()), map_cos=float(np.nanmean(z["cos_ap"][p])), map_sm=float(np.nanmean(ap_sm[p])))
    r["d_top1"] = r["top1_sm"] - r["top1_cos"]; r["d_map"] = r["map_sm"] - r["map_cos"]; out[key] = r
    print(key.ljust(30), f"in64 {r['share_in_top64']:.3f} rescue {r['in64']['rescue']:.4f} harm {r['in64']['harm']:.4f} NCR {r['ncr']:+.4f} [{r['ncr_ci'][0]:+.4f},{r['ncr_ci'][1]:+.4f}] dtop1 {r['d_top1']:+.4f} dmAP {r['d_map']:+.4f}", flush=True)
V = list(out.values()); med = lambda f: float(np.median([f(x) for x in V]))
sel = dict(med_ncr=med(lambda x: x["ncr"]), n_ci_pos=sum(x["ncr_ci"][0] > 0 for x in V), n_ci_neg=sum(x["ncr_ci"][1] < 0 for x in V),
           med_rescue=med(lambda x: x["in64"]["rescue"]), med_harm=med(lambda x: x["in64"]["harm"]), med_dtop1=med(lambda x: x["d_top1"]), med_dmap=med(lambda x: x["d_map"]),
           n_dmap_pos=sum(x["d_map"] > 0 for x in V))
sel["caseA"] = bool(sel["med_ncr"] >= .01 and sel["n_ci_pos"] >= 19)
f2 = json.load(open(os.path.join(O, "judge_f2.json")))["S10"]; x = np.array([f2[k]["gs"] for k in CELLS]); yv = np.array([f2[k]["gs"] - f2[k]["s10"] for k in CELLS])
rho = spearmanr(x, yv)[0]; perm = np.array([spearmanr(x, rng.permutation(yv))[0] for _ in range(10000)]); pval = float((perm >= rho).mean())
gsf = dict(rho=float(rho), p=pval, caseB=bool(rho > 0 and pval < .05), points={k: [float(a), float(b)] for k, a, b in zip(CELLS, x, yv)})
case = "A" if sel["caseA"] else ("B" if gsf["caseB"] else "C"); print("SEL", json.dumps(sel)); print("GSF", json.dumps({k: v for k, v in gsf.items() if k != "points"}), "CASE", case)
json.dump(dict(cells=out, sel=sel, gsf=gsf, case=case), open(os.path.join(O, "sel.json"), "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(3.4, 2.6)); mk = {"cuhk": "o", "prw": "s", "cuhkdet": "^", "prwdet": "v", "market": "D", "cuhkseq": "*", "prwseq": "P"}
for k, a, b in zip(CELLS, x, yv):
    e = k.split("_", 1)[1]; ax.scatter(a, b, marker=mk[k.split("_")[0]], s=22, c="#1f77b4" if e in ENC[:3] or e == "seqnet" else "#ff7f0e", edgecolors="k", linewidths=0.3)
ax.axhline(0, c="gray", lw=0.6); ax.set_xlabel("GS worst regret"); ax.set_ylabel("GS $-$ SetWiseMatch"); ax.set_title(f"Spearman $\\rho$ = {rho:.2f} (" + ("p < 0.001" if pval < 1e-3 else f"p = {pval:.3f}") + ")", fontsize=8)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker=v, ls="", c="w", markeredgecolor="k", label=n) for n, v in mk.items()] + [Line2D([], [], marker="o", ls="", c=cl, label=n) for n, cl in (("ReID/SeqNet", "#1f77b4"), ("general", "#ff7f0e"))]
ax.legend(handles=h, fontsize=5, ncol=2, loc="upper left"); fig.tight_layout(); fig.savefig(os.path.join(R, "new_paper/figs/gs_gap.pdf")); fig.savefig(os.path.join(R, "new_paper/figs/gs_gap.png"), dpi=200)
