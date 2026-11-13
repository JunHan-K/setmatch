"""11-04 shared figure style for the CVPR paper: Times-like font (TeX Gyre Termes), 7-8 pt text, CVPR column widths,
one fixed colour per method (dataviz reference palette, categorical slots 1-3 are validated all-pairs; cosine is the
neutral secondary ink), redundant line style / marker per method so identity never rests on colour alone."""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in glob.glob("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-*.otf"):
    font_manager.fontManager.addfont(f)
COL, TXT = 3.25, 6.875                      # CVPR single / double column width (inch)
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8c8b86", "#e6e5e1", "#ffffff"
C = dict(sm="#1457e6", gs="#eb6834", ctx="#1baf7a", cos=INK2, cos_sz=INK2, gs_sz="#eb6834", neg="#e34948")
LS = dict(sm="-", gs="-", ctx="-", cos="-", cos_sz="--", gs_sz="--")
MK = dict(sm="o", gs="s", ctx="^", cos="D", cos_sz="x", gs_sz="+")
NAME = dict(sm="SetWiseMatch", gs="GS", ctx="Set pooling", cos="Cosine", cos_sz="Cosine, size-aware", gs_sz="GS, size-aware")
RAMP = ("#86b6ef", "#2a78d6", "#104281")   # ordinal blue ramp (steps 250, 450, 650)
DSM = {"cuhk": "o", "cuhkdet": "^", "prw": "s", "prwdet": "v", "market": "D", "cuhkseq": "o", "prwseq": "s"}
DSN = {"cuhk": "CUHK-SYSU", "cuhkdet": "CUHK prop.", "prw": "PRW", "prwdet": "PRW prop.", "market": "Market"}

plt.rcParams.update({
    "font.family": "TeX Gyre Termes", "mathtext.fontset": "stix", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.4, "lines.markersize": 3.5,
    "legend.frameon": False, "legend.handlelength": 1.8, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
    "pdf.fonttype": 42, "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.titleweight": "bold"})


def panel(ax, letter, x=-0.16, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")
