"""11-04 MF + DF (PREREG): DIR at matched FAR under the strict convention. Rules: cosine, GS, SetWiseMatch (S10, 10 seeds,
per-seed then averaged), set pooling (ctx_s10main). (a) target-oracle: per test K the threshold giving FAR = alpha on
absent episodes; (b) source-fixed: threshold giving FAR = alpha on the train-eval absent episodes. alpha in {.05, .10}.
DF: DIR-FAR curves at the largest test K for four settings -> new_paper/figs/dirfar.pdf. Writes results/setmatch/mf.json."""
import json
import os

import numpy as np

SMT = os.environ.get("SM_TAG", "_s10main")   # 11-04: "_s20" = all 20 seeds (main numbers), default = 10 seeds

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); O = os.path.join(R, "results/setmatch"); V5 = os.path.join(O, "v5")
os.makedirs(os.path.join(R, "new_paper/figs"), exist_ok=True)
ENC = ("osnet_ain_x1_0_msmt17", "transreidssl_msmt", "solider_msmt", "dinov2", "clip")
CELLS = [f"{d}_{e}" for d in ("cuhk", "cuhkdet", "prw", "prwdet", "market") for e in ENC] + ["cuhkseq_seqnet", "prwseq_seqnet"]
AL = (0.05, 0.10); FG = np.linspace(0.0, 0.3, 61)


def rules(key):
    z = np.load(os.path.join(V5, f"{key}_meta.npz")); out = {"cos": [(z["raw_tr"], z["raw_te"], z["Vte"])], "gs": [(z["sdse_tr"], z["sdse_te"], z["Vte"])]}
    for m, v in (("sm", "attn" + SMT), ("ctx", "ctx" + SMT)):
        y = np.load(os.path.join(V5, f"{key}_{v}.npz")); out[m] = [(y[f"conf_tr_s{s}"], y[f"conf_te_s{s}"], y[f"corr_te_s{s}"]) for s in range(20) if f"conf_te_s{s}" in y.files]
    return z, out


def dir_at(x, c, E, t):
    return float(((x > t) & (c > 0))[E == 1].mean()), float((x > t)[E == 0].mean())


res, curves = {}, {}
for key in CELLS:
    z, rr = rules(key); E, K, Etr = z["E"], z["K"], z["Etr"]; Ks = np.unique(K); r = {}
    for m, runs in rr.items():
        acc = {f"orc{a}": [] for a in AL} | {f"src{a}": [] for a in AL} | {f"srcfar{a}": [] for a in AL} | {f"srcdev{a}": [] for a in AL}; cur = []
        for xtr, xte, c in runs:
            for a in AL:
                o, sd, sf = [], [], []; ts = np.quantile(xtr[Etr == 0], 1 - a)
                for k in Ks:
                    m_ = K == k; t = np.quantile(xte[m_ & (E == 0)], 1 - a); o.append(dir_at(xte[m_], c[m_], E[m_], t)[0])
                    d_, f_ = dir_at(xte[m_], c[m_], E[m_], ts); sd.append(d_); sf.append(f_)
                acc[f"orc{a}"].append(np.mean(o)); acc[f"src{a}"].append(np.mean(sd)); acc[f"srcfar{a}"].append(np.max(sf)); acc[f"srcdev{a}"].append(np.max(np.abs(np.array(sf) - a)))
            m_ = K == Ks.max(); x0 = xte[m_ & (E == 0)]; cur.append([dir_at(xte[m_], c[m_], E[m_], np.quantile(x0, 1 - f) if f > 0 else np.inf)[0] for f in FG])
        r[m] = {k: float(np.mean(v)) for k, v in acc.items()}; curves.setdefault(key, {})[m] = np.mean(cur, 0).tolist()
    res[key] = r; print(key.ljust(30), "  ".join(f"{m} DIR@5 {v['orc0.05']:.3f} DIR@10 {v['orc0.1']:.3f} src5 {v['src0.05']:.3f}/far {v['srcfar0.05']:.3f}" for m, v in r.items()), flush=True)
d = np.array([res[k]["sm"]["orc0.05"] - res[k]["gs"]["orc0.05"] for k in CELLS]); d10 = np.array([res[k]["sm"]["orc0.1"] - res[k]["gs"]["orc0.1"] for k in CELLS])
dc = np.array([res[k]["sm"]["orc0.05"] - res[k]["cos"]["orc0.05"] for k in CELLS])
S = dict(med_d_gs5=float(np.median(d)), sm_gt_gs5=int((d > 0).sum()), med_d_gs10=float(np.median(d10)), sm_gt_gs10=int((d10 > 0).sum()), med_d_cos5=float(np.median(dc)), sm_gt_cos5=int((dc > 0).sum()),
         med={m: {f: float(np.median([res[k][m][f] for k in CELLS])) for f in res[CELLS[0]][m]} for m in ("cos", "gs", "ctx", "sm")})
S["verdict"] = "more true accepts at matched risk" if S["med_d_gs5"] >= .01 and S["sm_gt_gs5"] >= 19 else ("limit to accuracy-optimal operating point" if S["med_d_gs5"] <= 0 else "mixed")
print(json.dumps(S)); json.dump(dict(cells=res, curves=curves, far_grid=FG.tolist(), summary=S), open(os.path.join(O, "mf.json"), "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
show = (("cuhk_osnet_ain_x1_0_msmt17", "CUHK-SYSU, OSNet"), ("cuhk_clip", "CUHK-SYSU, CLIP"), ("prwdet_transreidssl_msmt", "PRW prop., TReID"), ("cuhkseq_seqnet", "CUHK-SYSU, SeqNet"))
fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.9), sharey=False); col = {"cos": "#718096", "gs": "#dd6b20", "sm": "#2b6cb0"}; lab = {"cos": "cosine", "gs": "GS", "sm": "SetWiseMatch"}
for ax, (k, t) in zip(axs, show):
    for m in ("cos", "gs", "sm"):
        ax.plot(FG, curves[k][m], c=col[m], lw=1.2, label=lab[m])
    ax.axvline(0.05, c="gray", lw=0.5, ls=":"); ax.set_title(t, fontsize=7); ax.set_xlabel("FAR", fontsize=7); ax.tick_params(labelsize=6)
axs[0].set_ylabel("DIR", fontsize=7); axs[0].legend(fontsize=6, frameon=False); fig.tight_layout(); fig.savefig(os.path.join(R, "new_paper/figs/dirfar.pdf")); fig.savefig(os.path.join(R, "new_paper/figs/dirfar.png"), dpi=200)
