"""11-02 SetWiseMatch v5 evaluation: merges results/setmatch/v5/<data>_<enc>_{meta,<variant>}.npz.
Per method and cell K: open-world AUROC (confidence vs correctness of the method's own choice, E = 0 and 1 mixed),
frozen-threshold regret for pi in {.2, .5, .8} (threshold chosen on train episodes at pi = .5), closed-set mAP / R1 on
E = 1 episodes. Learned variants: metrics per seed, then averaged. Writes results/setmatch/v5_eval.json."""
import glob
import json
import os

import numpy as np
from scipy.stats import rankdata

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V5 = os.path.join(R, "results/setmatch/v5")
BASE = ("raw", "margin", "sdse", "pnull")
LEARN = ("pair_bce", "pair_null", "ctx", "attn", "setmatch", "ctx_cal", "attn_cal", "ctx_w", "attn_w", "ctx_cal_w", "attn_cal_w")
EXTRAP = dict(market=(200, 0), cuhk=(1000, 2000, 4000), cuhkdet=(1000, 2000, 4000), prw=(2000, 6111), prwdet=(2000, 6111), cuhkseq=(1000, 2000, 4000), prwseq=(2000, 6111), msmt=(1000, 0), rpif=(1000, 0))


def auc(s, y):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return np.nan
    r = rankdata(s); n1 = y.sum(); return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def acc(x, c, E, pi, t):
    # SELECTIVE convention (NOT strict accuracy): a present query is credited when accept == correct,
    # so rejecting a wrong choice also counts. Used only for the paper's selective-regret / R_K columns;
    # strict accuracy is computed separately in sm_ctrl.strict_curves() with the (x > t) & (c > 0) rule.
    return pi * ((x[E == 1] > t) == (c[E == 1] > 0)).mean() + (1 - pi) * (x[E == 0] <= t).mean()


def acc_grid(x, c, E, grid):
    """(E=1 term, E=0 term) of acc() for every threshold in grid, vectorised via sorted counts"""
    a = np.sort(x[(E == 1) & (c > 0)]); b = np.sort(x[(E == 1) & ~(c > 0)]); z = np.sort(x[E == 0]); n1 = max(len(a) + len(b), 1); n0 = max(len(z), 1)
    t1 = ((len(a) - np.searchsorted(a, grid, side="right")) + np.searchsorted(b, grid, side="right")) / n1
    return t1, np.searchsorted(z, grid, side="right") / n0


def metrics(xtr, ctr, Etr, xte, cte, E, K, ap, r1, space=None):
    """space: 'prob' (posterior) or 'logit' enables the v11 prior-corrected threshold logit(t_pi) = logit(t_0.5) - logit(pi)"""
    if space == "prob":
        lg = lambda x: np.log(np.clip(x, 1e-7, 1 - 1e-7)) - np.log1p(-np.clip(x, 1e-7, 1 - 1e-7)); xtr, xte = lg(np.asarray(xtr)), lg(np.asarray(xte))
    grid = np.unique(np.quantile(xtr, np.linspace(0, 1, 401)))
    t1, t0 = acc_grid(xtr, ctr, Etr, grid); fz = grid[int(np.argmax(0.5 * t1 + 0.5 * t0))]; cells = {}
    for k in np.unique(K):
        m = K == k; x, c, e = xte[m], cte[m], E[m]
        og = np.unique(np.r_[x, fz, np.nextafter(x.min(), -np.inf)])   # oracle scans this cell's OWN test boundaries (+ frozen t + an accept-all point below min(x)) -> no oracle/regret understatement
        og1, og0 = acc_grid(x, c, e, og)
        mAP = float(np.nanmean(np.clip(ap[m & (E == 1)], 0.0, 1.0)))   # guard: AP in [0,1] (detector duplicate proposals could otherwise exceed 1)
        cells[int(k)] = dict(auroc=auc(x, c), regret={pi: float(np.max(pi * og1 + (1 - pi) * og0) - acc(x, c, e, pi, fz)) for pi in (0.2, 0.5, 0.8)},
                             mAP=mAP, R1=float(np.nanmean(np.clip(r1[m & (E == 1)], 0.0, 1.0))), acc_frozen=float(acc(x, c, e, 0.5, fz)))
        if space:
            cells[int(k)]["regret_pc"] = {pi: float(np.max(pi * og1 + (1 - pi) * og0) - acc(x, c, e, pi, fz - np.log(pi / (1 - pi)))) for pi in (0.2, 0.5, 0.8)}
    return cells


def main():
    out = {}
    for meta in sorted(glob.glob(os.path.join(V5, "*_meta.npz"))):
        key = os.path.basename(meta)[:-9]; data = key.split("_")[0]; z = np.load(meta); E, K = z["E"], z["K"]; per = {}
        for b in BASE:
            per[b] = metrics(z[b + "_tr"], z["Vtr"], z["Etr"], z[b + "_te"], z["Vte"], E, K, z["cos_ap"], z["cos_r1"], space="logit" if b == "sdse" else None)
        f = os.path.join(V5, f"{key}_evt.npz")                     # v6 extreme-value baselines (cosine top-1, same labels)
        if os.path.exists(f):
            y = np.load(f)
            for b in ("wscore", "qnull"):
                per[b] = metrics(y[b + "_tr"], z["Vtr"], z["Etr"], y[b + "_te"], z["Vte"], E, K, z["cos_ap"], z["cos_r1"])
        extra = sorted(os.path.basename(f)[len(key) + 1:-4] for f in glob.glob(os.path.join(V5, f"{key}_*.npz")))
        for v in list(LEARN) + [x for x in extra if x not in LEARN and x not in ("meta", "evt", "base") and not x.endswith("smoke")]:
            f = os.path.join(V5, f"{key}_{v}.npz")
            if not os.path.exists(f):
                continue
            y = np.load(f); seeds = sorted({k.split("_s")[-1] for k in y if k.startswith("conf_te")}); runs = []
            Etr = y["Etr"] if "Etr" in y else z["Etr"]
            if len(y[f"conf_tr_s{seeds[0]}"]) != len(Etr):              # other training-K grid without stored labels
                print("  skip", v, "(train-eval episodes differ from meta)"); continue
            if f"ap_s{seeds[0]}" not in y:                              # non-eval variant (e.g. attn_dg saves only conf/corr)
                print("  skip", v, "(no ap/r1 arrays)"); continue
            for s in seeds:
                runs.append(metrics(y[f"conf_tr_s{s}"], y[f"corr_tr_s{s}"], Etr, y[f"conf_te_s{s}"], y[f"corr_te_s{s}"], E, K, y[f"ap_s{s}"], y[f"r1_s{s}"], space="prob"))
            per[v] = {k: {m: (float(np.nanmean([r[k][m] for r in runs])) if not m.startswith("regret") else {pi: float(np.mean([r[k][m][pi] for r in runs])) for pi in (0.2, 0.5, 0.8)})
                          for m in runs[0][k]} for k in runs[0]}
        summ = {}
        for nm, cells in per.items():
            ks = list(cells); ex = [k for k in ks if k in EXTRAP[data]]
            summ[nm] = dict(auroc=round(np.nanmean([cells[k]["auroc"] for k in ks]), 4), auroc_extrap=round(np.nanmean([cells[k]["auroc"] for k in ex]), 4),
                            regret_worst=round(max(max(cells[k]["regret"].values()) for k in ks), 4), regret_mean=round(np.mean([np.mean(list(cells[k]["regret"].values())) for k in ks]), 4),
                            mAP=round(np.nanmean([cells[k]["mAP"] for k in ks]), 4), R1=round(np.nanmean([cells[k]["R1"] for k in ks]), 4),
                            regret_K=round(max(cells[k]["regret"][0.5] for k in ks), 4))
            if "regret_pc" in cells[ks[0]]:
                summ[nm]["regret_pc_worst"] = round(max(max(cells[k]["regret_pc"].values()) for k in ks), 4)
        out[key] = dict(summary=summ, cells={nm: {str(k): v for k, v in c.items()} for nm, c in per.items()})
        print(key); [print("  ", nm.ljust(9), s) for nm, s in summ.items()]
    p = os.path.join(R, "results/setmatch/v5_eval.json"); json.dump(out, open(p + ".tmp", "w"), indent=1, default=float); os.replace(p + ".tmp", p)


if __name__ == "__main__":
    main()
