"""11-04 paper tables / figures from the final runs (mean over seeds of the per-seed worst; AUROC and mAP averaged over
test K, then seeds). Methods: cosine, GS (SDSE-inspired gallery statistics), k-reciprocal re-ranking, pairwise BCE,
pairwise + NULL, set pooling + NULL (ctx), SetMatch (attn). SetMatch and ctx: 10 seeds (_s10main); pairwise: the 5-seed
ladder runs (_cmpraw, SeqNet _s5). Writes results/setmatch/paper.json, new_paper/tab_main.tex, new_paper/tab_full.tex,
new_paper/tab_sel.tex, new_paper/figs/agnostic.pdf."""
import json
import os
import warnings

import numpy as np

SMT = os.environ.get("SM_TAG", "_s10main")   # 11-04: "_s20" = all 20 seeds (main numbers), default = 10 seeds
NSEED = "20" if SMT == "_s20" else "10"

import sm_v5_eval as EV

warnings.filterwarnings("ignore")
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")); O = os.path.join(R, "results/setmatch"); V5 = os.path.join(O, "v5"); P = os.path.join(R, "new_paper")
ENC = ("osnet_ain_x1_0_msmt17", "transreidssl_msmt", "solider_msmt", "dinov2", "clip"); EN = dict(zip(ENC + ("seqnet",), ("OSNet", "TReID", "SOLIDER", "DINOv2", "CLIP", "SeqNet")))
DS = (("cuhk", "CUHK-SYSU"), ("cuhkdet", "CUHK prop."), ("prw", "PRW"), ("prwdet", "PRW prop."), ("market", "Market"))
CELLS = [f"{d}_{e}" for d, _ in DS for e in ENC] + ["cuhkseq_seqnet", "prwseq_seqnet"]
lg = lambda x: np.log(np.clip(x, 1e-7, 1 - 1e-7)) - np.log1p(-np.clip(x, 1e-7, 1 - 1e-7))


def summ(c):
    return max(max(x["regret"].values()) for x in c.values()), float(np.nanmean([x["auroc"] for x in c.values()])), float(np.nanmean([x["mAP"] for x in c.values()]))


def learned(key, v):
    y = np.load(os.path.join(V5, f"{key}_{v}.npz")); z = np.load(os.path.join(V5, f"{key}_meta.npz")); o = []
    for s in range(20):
        if f"conf_te_s{s}" not in y.files:
            break
        o.append(summ(EV.metrics(y[f"conf_tr_s{s}"], y[f"corr_tr_s{s}"], y["Etr"], y[f"conf_te_s{s}"], y[f"corr_te_s{s}"], z["E"], z["K"], y[f"ap_s{s}"], y[f"r1_s{s}"], space="prob")))
    o = np.array(o); return dict(regret=float(o[:, 0].mean()), auroc=float(o[:, 1].mean()), mAP=float(o[:, 2].mean()), seeds=len(o))


def base(key, b):
    z = np.load(os.path.join(V5, f"{key}_meta.npz"))
    r = summ(EV.metrics(z[b + "_tr"], z["Vtr"], z["Etr"], z[b + "_te"], z["Vte"], z["E"], z["K"], z["cos_ap"], z["cos_r1"], space="logit" if b == "sdse" else None))
    return dict(regret=r[0], auroc=r[1], mAP=r[2], seeds=0)


def rerank(key):
    y = np.load(os.path.join(V5, f"{key}_rerank.npz")); z = np.load(os.path.join(V5, f"{key}_meta.npz"))
    r = summ(EV.metrics(y["conf_tr_s0"], y["corr_tr_s0"], y["Etr"], y["conf_te_s0"], y["corr_te_s0"], z["E"], z["K"], y["ap_s0"], y["r1_s0"]))
    return dict(regret=r[0], auroc=r[1], mAP=r[2], seeds=0)


M = ("cos", "gs", "rr", "pair_bce", "pair_null", "ctx", "setmatch"); res = {}
for key in CELLS:
    t5 = "_s5" if key.endswith("seq_seqnet") else "_cmpraw"
    res[key] = dict(cos=base(key, "raw"), gs=base(key, "sdse"), rr=rerank(key), pair_bce=learned(key, "pair_bce" + t5), pair_null=learned(key, "pair_null" + t5),
                    ctx=learned(key, "ctx" + SMT), setmatch=learned(key, "attn" + SMT))
    print(key.ljust(30), "  ".join(f"{m} {r['regret']:.3f}/{r['auroc']:.3f}/{r['mAP']:.3f}" for m, r in res[key].items()), flush=True)
coh = json.load(open(os.path.join(O, "judge_cohort.json")))["cells"]; cohk = [k for k in coh if k.startswith("in_")]
st = json.load(open(os.path.join(O, "st_query.json")))["summary"]; sel = json.load(open(os.path.join(O, "sel.json")))
med = lambda m, f: float(np.median([res[k][m][f] for k in CELLS]))
S = {m: dict(regret=med(m, "regret"), auroc=med(m, "auroc"), mAP=med(m, "mAP"), half_cos=sum(res[k][m]["regret"] <= 0.5 * res[k]["cos"]["regret"] for k in CELLS),
             d_auroc=float(np.median([res[k][m]["auroc"] - res[k]["cos"]["auroc"] for k in CELLS])), d_map=float(np.median([res[k][m]["mAP"] - res[k]["cos"]["mAP"] for k in CELLS])),
             best=sum(res[k][m]["regret"] == min(res[k][x]["regret"] for x in M) for k in CELLS)) for m in M}
S["cohort"] = dict(n=len(cohk), regret=float(np.median([min(coh[k][x] for x in ("tnorm", "znorm", "snorm")) for k in cohk])), cos_same=float(np.median([coh[k]["cos"] for k in cohk])),
                   worse_than_cos=sum(min(coh[k][x] for x in ("tnorm", "znorm", "snorm")) > coh[k]["cos"] for k in cohk),
                   half_cos=sum(min(coh[k][x] for x in ("tnorm", "znorm", "snorm")) <= 0.5 * coh[k]["cos"] for k in cohk))
sca = json.load(open(os.path.join(O, "sca.json")))["cells"]   # size-aware thresholds (a + b log N fitted on training galleries)
for m, b_ in (("cos_sz", "cos"), ("gs_sz", "gs")):
    S[m] = dict(regret=float(np.median([sca[k][b_]["sel_worst"] for k in CELLS])), auroc=S[b_]["auroc"], mAP=S[b_]["mAP"],
                half_cos=sum(sca[k][b_]["sel_worst"] <= 0.5 * res[k]["cos"]["regret"] for k in CELLS))
for m, v in (("gs_mlp", "gsmlp_s10ng"), ("sch", "selch_s10sch")):   # 11-05 strongest baselines (NG, SCH), Tab. 1 rows only
    ex = {}
    for key in CELLS:
        y = np.load(os.path.join(V5, f"{key}_{v}.npz")); z = np.load(os.path.join(V5, f"{key}_meta.npz")); o = []
        for s_ in range(20):
            if f"conf_te_s{s_}" not in y.files:
                break
            ap, r1 = (y[f"ap_s{s_}"], y[f"r1_s{s_}"]) if f"ap_s{s_}" in y.files else (z["cos_ap"], z["cos_r1"])
            o.append(summ(EV.metrics(y[f"conf_tr_s{s_}"], y[f"corr_tr_s{s_}"], y["Etr"], y[f"conf_te_s{s_}"], y[f"corr_te_s{s_}"], z["E"], z["K"], ap, r1, space="prob")))
        o = np.array(o); ex[key] = dict(regret=float(o[:, 0].mean()), auroc=float(o[:, 1].mean()), mAP=float(o[:, 2].mean()))
    S[m] = dict(regret=float(np.median([ex[k]["regret"] for k in CELLS])), auroc=float(np.median([ex[k]["auroc"] for k in CELLS])),
                mAP=float(np.median([ex[k]["mAP"] for k in CELLS])), half_cos=sum(ex[k]["regret"] <= 0.5 * res[k]["cos"]["regret"] for k in CELLS))
print(json.dumps(S, indent=0)); json.dump(dict(cells=res, summary=S), open(os.path.join(O, "paper.json"), "w"), indent=1)
f3 = lambda x: ("%.3f" % x).replace("0.", ".", 1) if x >= 0 else ("$-$%.3f" % -x).replace("0.", ".", 1)
pm = lambda x: ("$+$" if x >= 0 else "$-$") + ("%.3f" % abs(x)).replace("0.", ".", 1)
# ---- main summary table
rows = [("Cosine threshold", "\\checkmark", "threshold", "--", "cos"), ("Cosine, size-aware threshold", "\\checkmark", "thr.\\ in $\\log N$", "--", "cos_sz"),
        ("k-reciprocal re-ranking", "\\checkmark", "threshold", "gallery graph", "rr"),
        ("GS (SDSE-inspired)", "top-1 only", "\\checkmark", "statistics", "gs"), ("GS, size-aware threshold", "top-1 only", "\\checkmark", "statistics", "gs_sz"), ("GS-MLP", "top-1 only", "\\checkmark", "statistics", "gs_mlp"),
        ("Pairwise (BCE)", "\\checkmark", "threshold", "--", "pair_bce"),
        ("Pairwise + $\\nul$", "\\checkmark", "\\checkmark", "--", "pair_null"), ("Selector + correctness head", "\\checkmark", "head", "attention", "sch"), ("Set pooling + $\\nul$", "\\checkmark", "\\checkmark", "pooling", "ctx"),
        ("\\textbf{SetMatch}", "\\checkmark", "\\checkmark", "attention", "setmatch")]
L = []
for name, cc, rj, sr, m in rows:
    s = S[m]; hc = "--" if m == "cos" else f"{s['half_cos']}/27"; L.append(f"{name} & {cc} & {rj} & {sr} & {f3(s['regret'])} & {hc} & {f3(s['auroc'])} & {f3(s['mAP'])}\\\\")
    if m == "rr":
        c = S["cohort"]; L.append(f"Cohort Z/T/S-norm$^\\ddagger$ & \\checkmark & threshold & cohort & {f3(c['regret'])} (cos {f3(c['cos_same'])}) & {c['half_cos']}/{c['n']} & -- & =cos\\\\")
tab = ("\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\caption{Decision layers on identical frozen features, 27 dataset--encoder cells "
       "(5 encoders $\\times$ CUHK-SYSU, CUHK proposals, PRW, PRW proposals, Market-1501, plus SeqNet on CUHK-SYSU and PRW). Medians over cells; worst set-expansion "
       "regret (selective convention, Sec.~\\ref{sec:shift}) is the mean over seeds of each seed's worst case over test sizes and priors (SetMatch and set pooling: "+NSEED+" seeds; pairwise: 5). "
       "``$\\le\\frac12$ cos'': cells where the worst regret is at most half of the cosine's. AUROC ranks correct choices against all other episodes (open-world), mAP is closed-set on present queries. GS-MLP and the selector with a correctness head are the strongest baselines of Sec.~\\ref{sec:ctrl}. "
       "Size-aware: the accuracy-optimal threshold of each training gallery size regressed on $\\log N$ (two parameters). $^\\ddagger$Best of Z/T/S-norm per cell on the in-domain cells of the cohort study (CUHK-SYSU, PRW and SeqNet, "+str(S['cohort']['n'])+" cells) (2000 unlabelled training boxes as cohort); its own cosine median in brackets.}\\label{tab:main}\n"
       "\\begin{tabular}{lccccccc}\\toprule\n & candidate & explicit & set & worst & & & \\\\\n method & choice & reject & reasoning & regret$\\downarrow$ & $\\le\\frac12$ cos & AUROC$\\uparrow$ & mAP$\\uparrow$\\\\\\midrule\n"
       + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\\end{table*}\n")
open(os.path.join(P, "tab_main.tex"), "w").write(tab)
# ---- full per-cell table (supplement)
L = []
for d, dn in DS + (("seq", "SeqNet"),):
    ks = [f"{d}_{e}" for e in ENC] if d != "seq" else ["cuhkseq_seqnet", "prwseq_seqnet"]
    for i, k in enumerate(ks):
        en = EN[k.split("_", 1)[1]] if d != "seq" else ("CUHK-SYSU" if k.startswith("cuhk") else "PRW")
        b = min(res[k][m]["regret"] for m in ("cos", "gs", "rr", "ctx", "setmatch"))
        cell = lambda m: ("\\textbf{%s}" % f3(res[k][m]["regret"]) if res[k][m]["regret"] == b else f3(res[k][m]["regret"])) + "/" + f3(res[k][m]["mAP"])
        L.append((f"\\multirow{{{len(ks)}}}{{*}}{{{dn}}}" if i == 0 else "") + f" & {en} & " + " & ".join(cell(m) for m in ("cos", "gs", "rr", "ctx", "setmatch")) + "\\\\")
    L.append("\\midrule")
L[-1] = "\\bottomrule"
open(os.path.join(P, "tab_full.tex"), "w").write("\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{3pt}\n\\caption{Per-cell worst set-expansion regret (best in bold) / mAP. "
    "SetMatch and set pooling: mean of "+NSEED+" seeds. CUHK-SYSU and proposals: training galleries $\\le500$, test up to 4000; PRW: training $\\le1000$, test 6111.}\\label{tab:full}\n"
    "\\begin{tabular}{llccccc}\\toprule\n & & cosine & GS & re-rank & set pooling & SetMatch\\\\\\midrule\n" + "\n".join(L) + "\n\\end{tabular}\\end{table*}\n")
# ---- re-selection table by encoder family
fam = {"ReID (15)": [f"{d}_{e}" for d, _ in DS for e in ENC[:3]], "general (10)": [f"{d}_{e}" for d, _ in DS for e in ENC[3:]], "SeqNet (2)": ["cuhkseq_seqnet", "prwseq_seqnet"]}
C = sel["cells"]; L = []
for n, ks in fam.items():
    g = lambda f: float(np.median([f(C[k]) for k in ks]))
    L.append(f"{n} & {f3(g(lambda c: c['share_in_top64']))} & {f3(g(lambda c: c['in64']['rescue']))} & {f3(g(lambda c: c['in64']['harm']))} & {pm(g(lambda c: c['ncr']))} & "
             f"{sum(C[k]['ncr_ci'][0] > 0 for k in ks)}/{len(ks)} & {sum(C[k]['ncr_ci'][1] < 0 for k in ks)}/{len(ks)} & {pm(g(lambda c: c['d_map']))}\\\\")
open(os.path.join(P, "tab_sel.tex"), "w").write("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{2.5pt}\n\\caption{Candidate re-selection on present queries (forced choice). "
    "GS returns the cosine top-1, so these are also the differences to GS. Medians per encoder family; NCR = Rescue $-$ Harm when the target is among the encoded top-64; "
    "significance by a query-level bootstrap (95\\%).}\\label{tab:sel}\n\\resizebox{\\linewidth}{!}{\\begin{tabular}{lccccccc}\\toprule\n & in top-64 & Rescue & Harm & NCR & $>0$ & $<0$ & $\\Delta$mAP\\\\\\midrule\n"
    + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}}\\end{table}\n")
# ---- encoder-agnostic figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.8, 2.3)); mk = {"cuhk": "o", "cuhkdet": "^", "prw": "s", "prwdet": "v", "market": "D", "cuhkseq": "o", "prwseq": "s"}
for xi, e in enumerate(ENC + ("seqnet",)):
    ks = [k for k in CELLS if k.split("_", 1)[1] == e]
    for j, k in enumerate(ks):
        x = xi + (j - (len(ks) - 1) / 2) * 0.14; a, b = res[k]["cos"]["regret"], res[k]["setmatch"]["regret"]
        ax.annotate("", xy=(x, b), xytext=(x, a), arrowprops=dict(arrowstyle="->", lw=0.8, color="#2b6cb0" if b < a else "#c53030"))
        ax.scatter([x], [a], marker=mk[k.split("_")[0]], s=14, facecolors="white", edgecolors="gray", linewidths=0.6, zorder=3)
        ax.scatter([x], [b], marker=mk[k.split("_")[0]], s=14, c="#2b6cb0" if b < a else "#c53030", zorder=3)
ax.set_xticks(range(6)); ax.set_xticklabels([EN[e] for e in ENC + ("seqnet",)]); ax.set_ylabel("worst regret"); ax.set_ylim(bottom=0)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker=v, ls="", c="w", markeredgecolor="k", label=n) for n, v in (("CUHK-SYSU", "o"), ("CUHK prop.", "^"), ("PRW", "s"), ("PRW prop.", "v"), ("Market", "D"))],
          fontsize=6, ncol=5, loc="upper center", frameon=False)
ax.set_title("open marker: cosine threshold, filled: SetMatch on the same frozen features", fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(P, "figs/agnostic_single.pdf"))   # 11-05: was overwriting the two-panel Fig. (sm_figs.py)
