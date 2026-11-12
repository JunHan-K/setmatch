"""11-04 Fig. 3 (replaces the coupled GS vs GS-SetWiseMatch plot): worst regret of GS (x) against SetWiseMatch (y), 27 settings,
diagonal = equal. From results/setmatch/sel.json (gsf points: [GS, GS - SetWiseMatch]). Writes new_paper/figs/gs_gap.pdf."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
P = json.load(open(os.path.join(R, "results/setmatch/sel.json")))["gsf"]["points"]
REID = ("osnet_ain_x1_0_msmt17", "transreidssl_msmt", "solider_msmt", "seqnet")
mk = {"cuhk": "o", "cuhkdet": "^", "prw": "s", "prwdet": "v", "market": "D", "cuhkseq": "*", "prwseq": "P"}
fig, ax = plt.subplots(figsize=(3.3, 2.7)); hi = 0
for k, (g, d) in P.items():
    s = g - d; hi = max(hi, g, s)
    ax.scatter(g, s, marker=mk[k.split("_")[0]], s=24, c="#1f77b4" if k.split("_", 1)[1] in REID else "#ff7f0e", edgecolors="k", linewidths=0.3, zorder=3)
ax.plot([0, hi * 1.05], [0, hi * 1.05], c="gray", lw=0.7, ls="--"); ax.set_xlim(0, hi * 1.05); ax.set_ylim(0, hi * 1.05)
ax.set_xlabel("GS worst regret"); ax.set_ylabel("SetWiseMatch worst regret")
h = [Line2D([], [], marker=v, ls="", c="w", markeredgecolor="k", label=n) for n, v in (("CUHK-SYSU", "o"), ("CUHK prop.", "^"), ("PRW", "s"), ("PRW prop.", "v"), ("Market", "D"), ("SeqNet CUHK", "*"), ("SeqNet PRW", "P"))]
h += [Line2D([], [], marker="o", ls="", c=c, label=n) for n, c in (("ReID / SeqNet", "#1f77b4"), ("generic", "#ff7f0e"))]
ax.legend(handles=h, fontsize=5, ncol=2, loc="upper left", frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(R, "new_paper/figs/gs_gap.pdf")); fig.savefig(os.path.join(R, "new_paper/figs/gs_gap.png"), dpi=200)
