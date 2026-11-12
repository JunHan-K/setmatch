"""11-04 IDS figure from results/setmatch/ids.json (no recomputation): strict accuracy SetWiseMatch - GS vs the number of
training identities used as queries; dotted line = RPIfield's 56 training actors. Writes new_paper/figs/ids.pdf."""
import json
import os

import figstyle as F
from figstyle import plt

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")); res = json.load(open(os.path.join(R, "results/setmatch/ids.json")))["cells"]
NS = ("56", "150", "400", "all"); fig, axs = plt.subplots(1, 2, figsize=(F.COL, 1.5), sharey=True)
for ax, d, t in zip(axs, ("market", "cuhk"), ("Market-1501", "CUHK-SYSU")):
    full = {"market": 751, "cuhk": 5532}[d]
    for e, mk, n in (("osnet_ain_x1_0_msmt17", "o", "OSNet"), ("solider_msmt", "s", "SOLIDER"), ("clip", "^", "CLIP")):
        v = res[f"{d}_{e}"]; ax.plot([56, 150, 400, full], [v[k]["delta"] for k in NS], marker=mk, c=F.C["sm"], lw=1.1, ms=3, label=n)
    ax.axhline(0, c=F.INK2, lw=0.6); ax.axvline(56, c=F.MUTED, lw=0.6, ls=":"); ax.set_xscale("log"); ax.set_title(t, pad=2); ax.set_xlabel("training identities")
axs[0].text(60, 0.052, "RPIfield", fontsize=6, color=F.INK2)
axs[0].set_ylabel("SetWiseMatch $-$ GS acc."); axs[0].legend(loc="lower right", handletextpad=0.3, borderaxespad=0.1)
fig.tight_layout(w_pad=0.5); P = os.path.join(R, "new_paper/figs"); fig.savefig(os.path.join(P, "ids.pdf")); fig.savefig(os.path.join(P, "ids.png"), dpi=200)
