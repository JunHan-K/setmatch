"""11-04 Figure 1(a): probability that a fixed true match is still ranked first as distractor images are added in one
fixed order, per true-match cosine bin (frozen OSNet; CUHK-SYSU solid, PRW dashed). From results/setmatch/fig1_*.json.
Writes new_paper/figs/fig1a.pdf."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.makedirs(os.path.join(R, "new_paper/figs"), exist_ok=True)
fig, ax = plt.subplots(figsize=(3.3, 2.2)); col = {"0.6-0.7": "#c53030", "0.7-0.8": "#dd6b20", "0.8-0.9": "#2b6cb0"}
for d, ls in (("cuhk", "-"), ("prw", "--")):
    z = json.load(open(os.path.join(R, f"results/setmatch/fig1_{d}_osnet_ain_x1_0_msmt17.json")))
    for b, c in col.items():
        k = sorted(z[b], key=int); ax.plot([int(x) for x in k], [z[b][x]["p_correct"] for x in k], ls, c=c, lw=1.2, marker="o", ms=2, label=f"cos {b.replace('-', '–')}" if d == "cuhk" else None)
ax.set_xscale("log"); ax.set_xlabel("gallery images $K$"); ax.set_ylabel("P(match ranked 1st)"); ax.set_ylim(0, 1.03)
ax.plot([], [], "k-", lw=1, label="CUHK-SYSU"); ax.plot([], [], "k--", lw=1, label="PRW"); ax.legend(fontsize=6, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(R, "new_paper/figs/fig1a.pdf")); fig.savefig(os.path.join(R, "new_paper/figs/fig1a.png"), dpi=200)
