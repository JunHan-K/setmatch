"""11-04 CTRL (PREREG 'CTRL'; saved predictions only, no training).
J: is the joint candidate/no-match output needed?  Rules on identical frozen features and the same 10 SetWiseMatch seeds:
   cos (cosine accept, cosine top-1), gs (GS accept, cosine top-1), sm (SetWiseMatch accept and choice), ctx (set pooling),
   gs+sel (GS accept, SetWiseMatch choice: the same learned selector with a statistics presence detector),
   sm+cos (SetWiseMatch accept, cosine top-1).
U: utility under the strict gallery-level convention at pi = .5 -- a present query is handled correctly only if the
   true match is accepted; FAR = P(accept | absent), DIR = P(accept and correct | present); threshold fixed on the
   train-eval episodes by maximising the same strict accuracy; per test K: frozen accuracy, oracle accuracy, their gap.
   Also the selective worst regret of the paper (rejecting a wrong choice counts as correct) and R_K (pi = .5).
SA (env DROP_MISSED=1, PREREG 'SA'): present test episodes whose target is outside the encoded top-64 are removed
   (shortlist-level evaluation); training and thresholds unchanged. Writes ctrl_drop.json / tab_ctrl_drop.tex.
Writes results/setmatch/ctrl.json and new_paper/tab_ctrl.tex."""
import json
import os
import warnings

import numpy as np

SMT = os.environ.get("SM_TAG", "_s10main")   # 11-04: "_s20" = all 20 seeds (main numbers), default = 10 seeds

import sm_v5_eval as EV

warnings.filterwarnings("ignore")
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")); O = os.path.join(R, "results/setmatch"); V5 = os.path.join(O, "v5")
ENC = ("osnet_ain_x1_0_msmt17", "transreidssl_msmt", "solider_msmt", "dinov2", "clip")
CELLS = [f"{d}_{e}" for d in ("cuhk", "cuhkdet", "prw", "prwdet", "market") for e in ENC] + ["cuhkseq_seqnet", "prwseq_seqnet"]
KT = {"market": "10-20-50-100-200-0", "cuhk": "50-100-200-500", "prw": "50-100-200-500-1000", "cuhkdet": "50-100-200-500", "prwdet": "50-100-200-500-1000", "cuhkseq": "50-100-200-500", "prwseq": "50-100-200-500-1000"}
DROP = os.environ.get("DROP_MISSED") == "1"
TE = ("conf_te", "corr_te", "ap_s", "r1_s")
lg = lambda x: np.log(np.clip(x, 1e-7, 1 - 1e-7)) - np.log1p(-np.clip(x, 1e-7, 1 - 1e-7))


def strict_curves(x, c, E, grid):
    """g1(t) = P(x > t and correct | E=1), g0(t) = P(x <= t | E=0) for every t in grid"""
    a = np.sort(x[(E == 1) & (c > 0)]); z = np.sort(x[E == 0]); n1 = max((E == 1).sum(), 1); n0 = max(len(z), 1)
    return (len(a) - np.searchsorted(a, grid, side="right")) / n1, np.searchsorted(z, grid, side="right") / n0


def evaluate(xtr, ctr, Etr, xte, cte, E, K, ap, r1, space):
    sel = EV.metrics(xtr, ctr, Etr, xte, cte, E, K, ap, r1, space=space)
    out = dict(sel_worst=max(max(v["regret"].values()) for v in sel.values()), R_K=max(v["regret"][0.5] for v in sel.values()))
    if space == "prob":
        xtr, xte = lg(xtr), lg(xte)
    grid = np.unique(np.quantile(xtr, np.linspace(0, 1, 401))); g1, g0 = strict_curves(xtr, ctr, Etr, grid); j = int(np.argmax(0.5 * g1 + 0.5 * g0))
    acc, orc, dir_, far = [], [], [], []
    for k in np.unique(K):
        m = K == k; h1, h0 = strict_curves(xte[m], cte[m], E[m], grid); a = 0.5 * h1 + 0.5 * h0
        acc.append(a[j]); orc.append(a.max()); dir_.append(h1[j]); far.append(1 - h0[j])
    acc, orc = np.array(acc), np.array(orc)
    out.update(acc_worst=float(acc.min()), acc_mean=float(acc.mean()), oracle_mean=float(orc.mean()), strict_regret=float((orc - acc).max()),
               dir_mean=float(np.mean(dir_)), far_max=float(np.max(far)), far_mean=float(np.mean(far)))
    return out


res, dropped = {}, {}
for key in CELLS:
    z = dict(np.load(os.path.join(V5, f"{key}_meta.npz"))); y = dict(np.load(os.path.join(V5, f"{key}_attn{SMT}.npz"))); yc = dict(np.load(os.path.join(V5, f"{key}_ctx{SMT}.npz")))
    assert (y["Etr"] == z["Etr"]).all(), key
    d_ = key.split("_")[0]; _cc = np.load(os.path.join(O, "cache", f"{key}{'' if d_ in ('cuhk', 'market') else '_q20'}_k{KT[d_]}_tr.npz"))   # 11-06 audit: catch a stale/mismatched meta vs v6 cache for the mixed rows (gs+sel, sm+cos, gs-mlp)
    assert (z["raw_tr"] == _cc["s"][:, 0]).all() and (z["Vtr"] == ((_cc["E"] == 1) & _cc["tm"][:, 0])).all(), key + ": meta tr != v6 tr cache"
    if DROP:
        d = key.split("_")[0]; cc = np.load(os.path.join(O, "cache", f"{key}{'' if d in ('cuhk', 'market') else '_q20'}_k{KT[d]}_te.npz"))
        assert (cc["E"] == z["E"]).all() and ((cc["tm"][:, 0] > 0) == (z["Vte"] > 0)).mean() > 0.999, key
        keep = ~((z["E"] == 1) & ~cc["tm"].any(1)); dropped[key] = float((~keep).sum() / max((z["E"] == 1).sum(), 1))
        for f in ("raw_te", "sdse_te", "Vte", "cos_ap", "cos_r1", "E", "K"):
            z[f] = z[f][keep]
        for D in (y, yc):
            for f in list(D):
                if f.startswith(TE):
                    D[f] = D[f][keep]
    E, K = z["E"], z["K"]; S = [s for s in range(20) if f"conf_te_s{s}" in y]; ap0, r10 = z["cos_ap"], z["cos_r1"]; r = {}
    r["cos"] = [evaluate(z["raw_tr"], z["Vtr"], z["Etr"], z["raw_te"], z["Vte"], E, K, ap0, r10, None)]
    r["gs"] = [evaluate(z["sdse_tr"], z["Vtr"], z["Etr"], z["sdse_te"], z["Vte"], E, K, ap0, r10, "logit")]
    r["sm"] = [evaluate(y[f"conf_tr_s{s}"], y[f"corr_tr_s{s}"], y["Etr"], y[f"conf_te_s{s}"], y[f"corr_te_s{s}"], E, K, y[f"ap_s{s}"], y[f"r1_s{s}"], "prob") for s in S]
    r["ctx"] = [evaluate(yc[f"conf_tr_s{s}"], yc[f"corr_tr_s{s}"], yc["Etr"], yc[f"conf_te_s{s}"], yc[f"corr_te_s{s}"], E, K, yc[f"ap_s{s}"], yc[f"r1_s{s}"], "prob") for s in S if f"conf_te_s{s}" in yc]
    r["gs+sel"] = [evaluate(z["sdse_tr"], y[f"corr_tr_s{s}"], z["Etr"], z["sdse_te"], y[f"corr_te_s{s}"], E, K, y[f"ap_s{s}"], y[f"r1_s{s}"], "logit") for s in S]
    fg = os.path.join(V5, f"{key}_gsmlp_s10ng.npz")                 # 11-04 NG: nonlinear GS (MLP on the statistics), cosine top-1
    if os.path.exists(fg) and not DROP:
        yg = dict(np.load(fg)); r["gs-mlp"] = [evaluate(yg[f"conf_tr_s{s}"], z["Vtr"], z["Etr"], yg[f"conf_te_s{s}"], z["Vte"], E, K, ap0, r10, "prob") for s in range(20) if f"conf_te_s{s}" in yg]
    fh = os.path.join(V5, f"{key}_selch_s10sch.npz")                 # 11-05 SCH: frozen selector + correctness head
    if os.path.exists(fh) and not DROP:
        yh = dict(np.load(fh)); r["sch"] = [evaluate(yh[f"conf_tr_s{s}"], yh[f"corr_tr_s{s}"], yh["Etr"], yh[f"conf_te_s{s}"], yh[f"corr_te_s{s}"], E, K, yh[f"ap_s{s}"], yh[f"r1_s{s}"], "prob") for s in range(20) if f"conf_te_s{s}" in yh]
    r["sm+cos"] = [evaluate(y[f"conf_tr_s{s}"], z["Vtr"], z["Etr"], y[f"conf_te_s{s}"], z["Vte"], E, K, ap0, r10, "prob") for s in S]
    res[key] = {m: {f: float(np.mean([o[f] for o in v])) for f in v[0]} for m, v in r.items()}
    print(key.ljust(30), "  ".join(f"{m} sel {v['sel_worst']:.3f} acc {v['acc_worst']:.3f}/{v['acc_mean']:.3f} gap {v['strict_regret']:.3f} DIR {v['dir_mean']:.3f} FAR {v['far_max']:.3f}" for m, v in res[key].items()), flush=True)
M = tuple(m for m in ("cos", "gs", "gs-mlp", "gs+sel", "sch", "sm+cos", "ctx", "sm") if m in res[CELLS[0]]); med = lambda m, f: float(np.median([res[k][m][f] for k in CELLS]))
summ = {m: {f: med(m, f) for f in res[CELLS[0]][m]} for m in M}
d_j = np.array([res[k]["sm"]["acc_mean"] - res[k]["gs+sel"]["acc_mean"] for k in CELLS])
J = dict(med_delta_acc=float(np.median(d_j)), sm_better=int((d_j >= .005).sum()), gssel_better=int((d_j <= -.005).sum()),
         sel_regret_sm_lt_gssel=sum(res[k]["sm"]["sel_worst"] < res[k]["gs+sel"]["sel_worst"] for k in CELLS))
J["verdict"] = "joint supported" if J["med_delta_acc"] >= .005 and J["sm_better"] >= 19 else ("joint replaceable by GS + selector" if J["med_delta_acc"] <= 0 else "mixed")
U = dict(sm_acc_gt_cos=sum(res[k]["sm"]["acc_mean"] > res[k]["cos"]["acc_mean"] for k in CELLS), sm_accworst_gt_cos=sum(res[k]["sm"]["acc_worst"] > res[k]["cos"]["acc_worst"] for k in CELLS),
         sm_acc_gt_gs=sum(res[k]["sm"]["acc_mean"] > res[k]["gs"]["acc_mean"] for k in CELLS))
U["verdict"] = "low regret carries over to frozen-threshold accuracy" if U["sm_acc_gt_cos"] >= 19 else "does not carry over"
if DROP:
    U["dropped_share_median"] = float(np.median(list(dropped.values()))); U["dropped_share_max"] = float(max(dropped.values()))
print("SUMMARY", json.dumps(summ, indent=0)); print("J", json.dumps(J)); print("U", json.dumps(U))
json.dump(dict(cells=res, summary=summ, J=J, U=U, dropped=dropped), open(os.path.join(O, "ctrl_drop.json" if DROP else "ctrl.json"), "w"), indent=1)
f3 = lambda x: ("%.3f" % x).replace("0.", ".", 1)
names = {"cos": "Cosine threshold", "gs": "GS (cosine top-1)", "gs-mlp": "GS-MLP (cosine top-1)", "gs+sel": "GS accept + SetWiseMatch choice", "sch": "Selector + correctness head", "sm+cos": "SetWiseMatch accept + cosine top-1", "ctx": "Set pooling + $\\nul$", "sm": "\\textbf{SetWiseMatch}"}
L = [f"{names[m]} & {f3(summ[m]['acc_mean'])} & {f3(summ[m]['acc_worst'])} & {f3(summ[m]['oracle_mean'])} & {f3(summ[m]['strict_regret'])} & {f3(summ[m]['dir_mean'])} & {f3(summ[m]['far_max'])} & {f3(summ[m]['R_K'])} & {f3(summ[m]['sel_worst'])}\\\\" for m in M]
fd = os.path.join(O, "dra20.json")                                 # 11-05 DRA follow-up: margin accept rule on the same models (10 seeds)
if os.path.exists(fd) and not DROP:
    dc = json.load(open(fd))["cells"]; dm = lambda f: float(np.median([dc[k]["margin"][f] for k in CELLS]))
    L.append("SetWiseMatch, margin rule (follow-up) & " + " & ".join(f3(dm(f)) for f in ("acc_mean", "acc_worst", "oracle_mean", "strict_regret", "dir_mean", "far_max", "R_K", "sel_worst")) + "\\\\")
open(os.path.join(R, "new_paper/tab_ctrl_drop.tex" if DROP else "new_paper/tab_ctrl.tex"), "w").write(
    "\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{3.5pt}\n\\caption{Decision utility at $\\pi=0.5$ under the strict gallery-level convention (a present query counts as correct only if its true match is accepted), "
    "medians over the 27 settings. Frozen: threshold fixed on training episodes; oracle: best threshold per test size. DIR: accepted and correct among present queries; FAR: accepts among absent queries (worst test size). "
    "$R_K$: cardinality-only regret ($\\pi=0.5$); selective: worst regret of Tab.~\\ref{tab:main}. GS-MLP: the same statistics with a two-layer MLP trained on SetWiseMatch's pool. Selector + correctness head: a frozen selector of SetWiseMatch's architecture without the no-match output and an MLP that predicts whether its choice is correct. The swap rows exchange the accept and the choice between GS and SetWiseMatch on the same seeds."
    + (" Present queries whose target is outside the encoded top-64 are removed." if DROP else "") + "}\\label{tab:ctrl" + ("drop" if DROP else "") + "}\n"
    "\\begin{tabular}{lcccccccc}\\toprule\n & \\multicolumn{2}{c}{frozen accuracy$\\uparrow$} & oracle & & & & & \\\\\n method & mean $K$ & worst $K$ & accuracy & gap$\\downarrow$ & DIR$\\uparrow$ & FAR$\\downarrow$ & $R_K\\downarrow$ & selective$\\downarrow$\\\\\\midrule\n"
    + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\\end{table*}\n")
