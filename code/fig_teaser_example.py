"""11-04 Fig.1(a): illustrative silhouettes with the real scores of one CUHK-SYSU test query (OSNet, query 1545, found by
find_teaser.py). Small gallery (50 images): the true match ranks first (0.811). Large gallery (4000 images): the same
true match with the same score ranks second behind a distractor (0.848). Below each row: the decision of a fixed cosine
threshold (0.763, 5% FAR on training galleries), a size-aware threshold (cannot change the top-1) and SetWiseMatch (picks and
accepts the true match in 18/20 seeds). Writes new_paper/figs/teaser_example.pdf."""
import os

from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

import figstyle as F
from figstyle import plt

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRUE, DIST, QRY = F.INK2, "#c9c8c3", F.INK2


def person(ax, x, y, h, col, lw=0.0, ec="none"):
    ax.add_patch(Circle((x, y + 0.78 * h), 0.13 * h, fc=col, ec=ec, lw=lw, zorder=3))
    ax.add_patch(FancyBboxPatch((x - 0.17 * h, y), 0.34 * h, 0.58 * h, boxstyle="round,pad=0,rounding_size=0.08", fc=col, ec=ec, lw=lw, zorder=3))


fig, ax = plt.subplots(figsize=(0.33 * F.TXT, 1.62)); ax.set_xlim(0, 10); ax.set_ylim(0, 7.2); ax.axis("off")
rows = ((4.55, "50 images", [(0.811, True), (0.675, False), (0.645, False), (0.602, False)], 0, 0,
         (("fixed $\\tau$", "$\\checkmark$"), ("$\\tau(N)$", "$\\checkmark$"), ("SetWiseMatch", "$\\checkmark$"))),
        (1.25, "4000 images", [(0.848, False), (0.811, True), (0.802, False), (0.777, False)], 0, 1,
         (("fixed $\\tau$", "$\\times$ distractor"), ("$\\tau(N)$", "$\\times$ top-1"), ("SetWiseMatch", "$\\checkmark$ true"))))
person(ax, 0.75, 3.1, 1.6, QRY); ax.text(0.75, 2.6, "query", ha="center", fontsize=6.5, color=F.INK2)
ax.plot([1.55, 1.55], [0.6, 6.9], c=F.GRID, lw=0.8); ax.plot([6.55, 6.55], [0.6, 6.9], c=F.GRID, lw=0.8)
for y, lab, cands, cos_pick, sm_pick, dec in rows:
    ax.text(1.85, y + 1.6, lab, fontsize=6.5, color=F.INK, fontweight="bold")
    for j, (sc, t) in enumerate(cands):
        x = 2.4 + j * 1.15; person(ax, x, y, 1.15, TRUE if t else DIST)
        ax.text(x, y - 0.42, f"{sc:.3f}", ha="center", fontsize=6, color=F.INK if t else F.MUTED, fontweight="bold" if t else "normal")
    xc = 2.4 + cos_pick * 1.15; ax.add_patch(Rectangle((xc - 0.42, y - 0.62), 0.84, 1.95, fill=False, ec=F.C["cos"], lw=0.8, ls="--", zorder=4))
    xs = 2.4 + sm_pick * 1.15; ax.add_patch(Rectangle((xs - 0.49, y - 0.69), 0.98, 2.09, fill=False, ec=F.C["sm"], lw=1.2, zorder=4))
    for i, (n, o) in enumerate(dec):
        c = F.C["sm"] if n == "SetWiseMatch" else F.INK2; yy = y + 1.2 - 0.55 * i
        ax.text(6.75, yy, n, fontsize=5.8, color=c, va="center"); ax.text(9.95, yy, o, fontsize=5.8, color=c, ha="right", va="center")
ax.text(5.5, 0.05, "illustrative silhouettes; real scores of one CUHK-SYSU query", ha="center", fontsize=5.2, color=F.MUTED, style="italic")
fig.tight_layout(pad=0.1); P = os.path.join(R, "new_paper/figs"); os.makedirs(P, exist_ok=True); fig.savefig(os.path.join(P, "teaser_example.pdf")); fig.savefig(os.path.join(P, "teaser_example.png"), dpi=250)
