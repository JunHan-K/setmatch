"""11-04 all paper figures in one style (figstyle.py), from saved results only:
  teaser_curves.pdf  Fig.1a  P(true match ranked first) vs gallery size, CUHK-SYSU | PRW, OSNet, cosine bins
  teaser_acc.pdf     Fig.1c  frozen-threshold strict accuracy (median, IQR over 27 settings) of the threshold rules and SetMatch
  agnostic.pdf       Fig.    (a) worst regret: cosine / size-aware cosine / SetMatch per setting, grouped by encoder;
                             (b) strict-accuracy gain of SetMatch over the best threshold baseline per setting
  gs_gap.pdf         Fig.    GS vs SetMatch worst regret, 27 settings, diagonal
  dirfar.pdf         Fig.    DIR vs FAR at the largest test gallery, four settings
  riskpower.pdf      Supp.   cross-dataset risk-power, 10 seeds (SetMatch with shortlist-aware training)"""
import json
import os

import numpy as np
from matplotlib.lines import Line2D

import figstyle as F
from figstyle import plt

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")); O = os.path.join(R, "results/setmatch"); V5 = os.path.join(O, "v5"); P = os.path.join(R, "new_paper/figs")
ENC = ("osnet_ain_x1_0_msmt17", "transreidssl_msmt", "solider_msmt", "dinov2", "clip"); EN = dict(zip(ENC + ("seqnet",), ("OSNet", "TReID-SSL", "SOLIDER", "DINOv2", "CLIP", "SeqNet"))); ENS = dict(EN, transreidssl_msmt="TReID")
CELLS = [f"{d}_{e}" for d in ("cuhk", "cuhkdet", "prw", "prwdet", "market") for e in ENC] + ["cuhkseq_seqnet", "prwseq_seqnet"]
J = lambda f: json.load(open(os.path.join(O, f)))
ctrl, sca, paper, sel, mf = J("ctrl.json")["cells"], J("sca.json")["cells"], J("paper.json")["cells"], J("sel.json"), J("mf.json")
save = lambda fig, n: (fig.savefig(os.path.join(P, n + ".pdf")), fig.savefig(os.path.join(P, n + ".png"), dpi=200), plt.close(fig))

# ---- Fig.1a
fig, axs = plt.subplots(1, 2, figsize=(0.33 * F.TXT, 1.5), sharey=True)
for ax, d, t in zip(axs, ("cuhk", "prw"), ("CUHK-SYSU", "PRW")):
    z = J(f"fig1_{d}_osnet_ain_x1_0_msmt17.json")
    for b, c in zip(("0.6-0.7", "0.7-0.8", "0.8-0.9"), F.RAMP):
        k = sorted(z[b], key=int); ax.plot([int(x) for x in k], [z[b][x]["p_correct"] for x in k], c=c, marker="o", ms=2.2, lw=1.2, label=b.replace("-", "–"))
    ax.set_xscale("log"); ax.set_title(t, pad=2); ax.set_xlabel("gallery size $K$"); ax.set_ylim(0, 1.04); ax.set_xticks([10, 100, 1000])
axs[0].set_ylabel("P(match ranked 1st)"); axs[0].legend(title="cosine", title_fontsize=6, loc="lower left", borderaxespad=0.2, handlelength=1.2)
fig.tight_layout(w_pad=0.6); save(fig, "teaser_curves")

# ---- Fig.1c (11-05: stability and accuracy side by side, C2)
ng = J("ng.json")["cells"]
rules = (("cos", "Cosine", lambda k: ctrl[k]["cos"]), ("cos_sz", "Cosine (size)", lambda k: sca[k]["cos"]), ("gs", "GS", lambda k: ctrl[k]["gs"]),
         ("gs_sz", "GS (size)", lambda k: sca[k]["gs"]), ("gsmlp", "GS-MLP", lambda k: ng[k]["gsmlp"]), ("sm", "SetMatch", lambda k: ctrl[k]["sm"]))
col = dict(F.C, gsmlp=F.C["gs"]); mk = dict(F.MK, gsmlp="s")
fig, axs = plt.subplots(1, 2, figsize=(0.30 * F.TXT, 1.62), sharey=True)
for ax, f, xl, fmt in ((axs[0], "sel_worst", "regret $\\downarrow$", "{:.3f}"), (axs[1], "acc_mean", "accuracy $\\uparrow$", "{:.3f}")):
    for i, (m, _, g) in enumerate(rules):
        v = np.array([g(k)[f] for k in CELLS]); q1, q2, q3 = np.quantile(v, [.25, .5, .75]); y = len(rules) - 1 - i
        ax.plot([q1, q3], [y, y], c=col[m], lw=2.0, alpha=0.35, solid_capstyle="round")
        ax.plot([q2], [y], marker=mk[m], c=col[m], ms=4.2, mfc=col[m] if m not in ("cos_sz", "gsmlp") else "none", mew=1.0)
        ax.text(q2, y + 0.32, fmt.format(q2).replace("0.", ".", 1), ha="center", va="bottom", fontsize=5.6, color=F.INK if m == "sm" else F.INK2, fontweight="bold" if m == "sm" else "normal")
    ax.set_xlabel(xl, labelpad=1); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0); ax.set_ylim(-0.6, len(rules) - 0.2)
axs[0].set_xticks([0.1, 0.2, 0.3]); axs[0].set_xlim(0.0, 0.36); axs[1].set_xticks([0.7, 0.8]); axs[1].set_xlim(0.66, 0.9)
axs[0].set_yticks(range(len(rules))); axs[0].set_yticklabels([n for _, n, _ in rules][::-1])
fig.tight_layout(w_pad=0.4); save(fig, "teaser_acc")

# ---- agnostic (two panels)
groups = [(e, [k for k in CELLS if k.split("_", 1)[1] == e]) for e in ENC + ("seqnet",)]
fig, axs = plt.subplots(1, 2, figsize=(F.TXT, 1.9), gridspec_kw=dict(width_ratios=[1.25, 1]))
ax = axs[0]
for gi, (e, ks) in enumerate(groups):
    for j, k in enumerate(ks):
        x = gi + (j - (len(ks) - 1) / 2) * 0.15; mk = F.DSM[k.split("_")[0]]; a, b, c = paper[k]["cos"]["regret"], sca[k]["cos"]["sel_worst"], paper[k]["setmatch"]["regret"]
        ax.plot([x, x], [min(a, b, c), max(a, b, c)], c=F.GRID, lw=1.0, zorder=1)
        ax.scatter(x, a, marker=mk, s=12, facecolors="white", edgecolors=F.C["cos"], linewidths=0.7, zorder=3)
        ax.scatter(x, b, marker="x", s=10, c=F.C["cos_sz"], linewidths=0.7, zorder=3)
        ax.scatter(x, c, marker=mk, s=13, c=F.C["sm"], edgecolors="white", linewidths=0.3, zorder=4)
ax.set_xticks(range(len(groups))); ax.set_xticklabels([EN[e] for e, _ in groups]); ax.set_ylabel("worst set-expansion regret"); ax.set_ylim(bottom=0); ax.grid(axis="x", visible=False)
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=F.C["cos"], label="cosine"), Line2D([], [], marker="x", ls="", c=F.C["cos_sz"], label="cosine, size-aware"),
                   Line2D([], [], marker="o", ls="", c=F.C["sm"], label="SetMatch")], loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, borderaxespad=0.1, columnspacing=0.8, handletextpad=0.2)
F.panel(ax, "a", x=-0.09, y=1.10)
ax = axs[1]
for gi, (e, ks) in enumerate(groups):
    for j, k in enumerate(ks):
        x = gi + (j - (len(ks) - 1) / 2) * 0.15; best = max(sca[k]["cos"]["acc_mean"], sca[k]["gs"]["acc_mean"], ctrl[k]["gs"]["acc_mean"]); d = ctrl[k]["sm"]["acc_mean"] - best
        ax.bar(x, d, width=0.12, color=F.C["sm"] if d >= 0 else F.C["neg"], linewidth=0)
ax.axhline(0, c=F.INK2, lw=0.6); ax.set_xticks(range(len(groups))); ax.set_xticklabels([ENS[e] for e, _ in groups]); ax.grid(axis="x", visible=False)
ax.set_ylabel("accuracy gain over best\nthreshold baseline"); F.panel(ax, "b", x=-0.13, y=1.10)
fig.tight_layout(w_pad=1.2); save(fig, "agnostic")

# ---- gs_gap
pts = sel["gsf"]["points"]; fig, ax = plt.subplots(figsize=(F.COL * 0.78, F.COL * 0.68)); hi = 0
for k, (g, dd) in pts.items():
    s = g - dd; hi = max(hi, g, s); fam = k.split("_", 1)[1]
    ax.scatter(g, s, marker=F.DSM[k.split("_")[0]], s=16, c=F.C["sm"] if fam in ENC[:3] or fam == "seqnet" else F.C["ctx"], edgecolors="white", linewidths=0.3, zorder=3)
lim = hi * 1.06; ax.plot([0, lim], [0, lim], c=F.MUTED, lw=0.7, ls="--", zorder=1); ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
ax.set_xlabel("GS worst regret"); ax.set_ylabel("SetMatch worst regret")
ax.text(lim * 0.97, lim * 0.05, "SetMatch lower", ha="right", fontsize=6, color=F.INK2); ax.text(lim * 0.04, lim * 0.93, "GS lower", fontsize=6, color=F.INK2, va="top")
ax.legend(handles=[Line2D([], [], marker="o", ls="", c=F.C["sm"], label="ReID / SeqNet"), Line2D([], [], marker="o", ls="", c=F.C["ctx"], label="general-purpose")]
          + [Line2D([], [], marker=v, ls="", mfc="white", mec=F.INK2, label=n) for n, v in (("CUHK-SYSU", "o"), ("CUHK prop.", "^"), ("PRW", "s"), ("PRW prop.", "v"), ("Market", "D"))],
          loc="center left", bbox_to_anchor=(1.0, 0.5), handletextpad=0.2, labelspacing=0.35)
fig.tight_layout(); save(fig, "gs_gap")

# ---- dirfar
show = (("cuhk_osnet_ain_x1_0_msmt17", "CUHK-SYSU · OSNet"), ("prwdet_transreidssl_msmt", "PRW prop. · TReID-SSL"), ("cuhk_clip", "CUHK-SYSU · CLIP"), ("cuhkseq_seqnet", "CUHK-SYSU · SeqNet"))
FG = np.array(mf["far_grid"]); fig, axs = plt.subplots(1, 4, figsize=(F.TXT, 1.55))
for ax, (k, t) in zip(axs, show):
    for m in ("cos", "gs", "sm"):
        ax.plot(FG, mf["curves"][k][m], c=F.C[m], ls=F.LS[m], lw=1.3, label=F.NAME[m], marker=F.MK[m], markevery=10, ms=2.5)
    for a in (0.05, 0.10):
        ax.axvline(a, c=F.MUTED, lw=0.5, ls=":")
    ax.set_title(t, pad=2); ax.set_xlabel("FAR"); ax.set_xlim(0, 0.3)
axs[0].set_ylabel("DIR"); axs[0].legend(loc="lower right", handletextpad=0.3)
fig.tight_layout(w_pad=0.8); save(fig, "dirfar")

# ---- riskpower (supplement)
def curve(src, tgt, enc, rule):
    y = np.load(os.path.join(V5, f"{tgt}_{enc}_{rule}_x{src}_src.npz")); E = np.load(os.path.join(V5, f"{tgt}_{enc}_meta.npz"))["E"]; grid = np.linspace(0, 0.6, 61); T, op = [], []
    for s in range(10):
        if f"conf_te_s{s}" not in y.files:
            break
        c, k = y[f"conf_te_s{s}"], y[f"corr_te_s{s}"].astype(bool); ts = np.quantile(y[f"conf_tr_s{s}"][y["Etr"] == 0], 0.95)
        T.append([((c > np.quantile(c[E == 0], 1 - f)) & k)[E == 1].mean() if f > 0 else 0.0 for f in grid]); op.append(((c[E == 0] > ts).mean(), ((c > ts) & k)[E == 1].mean()))
    return grid, np.mean(T, 0), np.mean(op, 0)


fig, axs = plt.subplots(2, 5, figsize=(F.TXT, 2.9), sharex=True)
for i, (s, t, name) in enumerate((("cuhk", "prw", "CUHK-SYSU → PRW"), ("prw", "cuhk", "PRW → CUHK-SYSU"))):
    for j, e in enumerate(ENC):
        ax = axs[i, j]
        for m, rule in (("cos", "cos"), ("gs", "gs"), ("sm", "attn_ht10")):
            g, tar, op = curve(s, t, e, rule); ax.plot(g, tar, c=F.C[m], ls=F.LS[m], lw=1.2, label=F.NAME[m]); ax.scatter(*op, marker=F.MK[m], s=14, c=F.C[m], edgecolors="white", linewidths=0.3, zorder=4)
        ax.axvline(0.05, c=F.MUTED, lw=0.5, ls=":"); ax.set_title(f"{EN[e]}" if i == 0 else "", pad=2)
        if j == 0:
            ax.set_ylabel(f"{name}\nTAR")
        if i == 1:
            ax.set_xlabel("target FAR")
axs[0, 0].legend(loc="lower right", handletextpad=0.3); fig.tight_layout(h_pad=0.4, w_pad=0.5); save(fig, "riskpower")
print("figures written")
