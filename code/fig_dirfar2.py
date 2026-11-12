"""11-05 Fig. 4 (after DRA20/EXT-M): DIR vs FAR at the largest test gallery for GS, the selector with a correctness head
(SCH, 10 seeds) and SetWiseMatch with the p_max and the margin accept score (same models, attn_s20dra, 20 seeds); four
settings. Threshold swept on the test absent queries (matched FAR). Writes new_paper/figs/dirfar.pdf/.png."""
import os

import numpy as np

import figstyle as F
from figstyle import plt

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")); V5 = os.path.join(R, "results/setmatch/v5"); P = os.path.join(R, "new_paper/figs")
lgt = lambda x: np.log(np.clip(x, 1e-9, 1.0)); FG = np.linspace(0.0, 0.3, 61)


def curve(runs, E, K):
    m = K == K.max(); out = []
    for x, c in runs:
        x, c, e = x[m], c[m], E[m]; neg = np.sort(x[e == 0])
        out.append([((x[e == 1] > (np.quantile(neg, 1 - f) if f > 0 else neg[-1] + 1)) & (c[e == 1] > 0)).mean() for f in FG])
    return np.mean(out, 0)


show = (("cuhk_osnet_ain_x1_0_msmt17", "CUHK-SYSU · OSNet"), ("prwdet_transreidssl_msmt", "PRW prop. · TReID-SSL"), ("cuhk_clip", "CUHK-SYSU · CLIP"), ("cuhkseq_seqnet", "CUHK-SYSU · SeqNet"))
ST = dict(gs=("GS", F.C["gs"], "-", "s"), sch=("selector + corr. head", "#7a5195", "-", "^"), pmax=("SetWiseMatch, $p_{\\max}$", F.C["sm"], "--", "o"),
          margin=("SetWiseMatch, margin", F.C["sm"], "-", "o"))
fig, axs = plt.subplots(1, 4, figsize=(F.TXT, 1.6))
for ax, (k, t) in zip(axs, show):
    z = dict(np.load(os.path.join(V5, f"{k}_meta.npz"))); E, K = z["E"], z["K"]
    y = dict(np.load(os.path.join(V5, f"{k}_attn_s20dra.npz"))); ys = dict(np.load(os.path.join(V5, f"{k}_selch_s10sch.npz")))
    S = lambda d: [s for s in range(20) if f"conf_te_s{s}" in d]
    runs = dict(gs=[(z["sdse_te"], z["Vte"])], sch=[(ys[f"conf_te_s{s}"], ys[f"corr_te_s{s}"]) for s in S(ys)],
                pmax=[(y[f"conf_te_s{s}"], y[f"corr_te_s{s}"]) for s in S(y)],
                margin=[(lgt(y[f"conf_te_s{s}"]) - lgt(y[f"null_te_s{s}"]), y[f"corr_te_s{s}"]) for s in S(y)])
    for m, (nm, c, ls, mk) in ST.items():
        ax.plot(FG, curve(runs[m], E, K), c=c, ls=ls, lw=1.3, label=nm, marker=mk, markevery=10, ms=2.5)
    for a in (0.01, 0.05):
        ax.axvline(a, c=F.MUTED, lw=0.5, ls=":")
    ax.set_title(t, pad=2); ax.set_xlabel("FAR"); ax.set_xlim(0, 0.3)
axs[0].set_ylabel("DIR"); axs[0].legend(loc="lower right", handletextpad=0.3, fontsize=5.5)
fig.tight_layout(w_pad=0.8); fig.savefig(os.path.join(P, "dirfar.pdf")); fig.savefig(os.path.join(P, "dirfar.png"), dpi=200)
print("saved dirfar")
