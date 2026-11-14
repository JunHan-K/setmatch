"""Candidate-set-shift analysis: performance as a FUNCTION of gallery size K (the phenomenon the paper is about),
reported SEPARATELY PER DATASET. Market's K is an identity count and Market K=0 is the FULL gallery, whereas
CUHK-SYSU / PRW's K is an image count -- different units, so they must not be pooled on one K axis. For
cosine / GS / SetWiseMatch, per dataset and gallery size K (averaged over encoders): strict accuracy at the
frozen (train-chosen) threshold, and forced-choice selection accuracy among present queries (cosine top-1 vs
SetWiseMatch re-ranking). Size-ADAPTIVE thresholding (t = b0 + b1*log N) is evaluated separately in sm_sca.py.
Reads the v5 prediction dumps; writes results/setmatch/shift.json. CPU only (numpy/scipy)."""
import glob
import json
import os
from collections import defaultdict

import numpy as np

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V5 = os.path.join(R, "results/setmatch/v5")
SMT = os.environ.get("SM_TAG", "_s10main")
lg = lambda x: np.log(np.clip(x, 1e-7, 1 - 1e-7)) - np.log1p(-np.clip(x, 1e-7, 1 - 1e-7))


def sc(x, c, E, grid):                                                 # strict curves: P(x>t & correct | E=1), P(x<=t | E=0)
    a = np.sort(x[(E == 1) & (c > 0)]); z = np.sort(x[E == 0]); n1 = max((E == 1).sum(), 1); n0 = max(len(z), 1)
    return (len(a) - np.searchsorted(a, grid, side="right")) / n1, np.searchsorted(z, grid, side="right") / n0


def frozen_per_k(xtr, ctr, Etr, xte, cte, E, K, space=None):          # strict accuracy per K at a single train-chosen threshold
    if space == "prob":                                              # only posteriors are logit-transformed; GS 'sdse' is ALREADY a logit (do NOT transform it again)
        xtr, xte = lg(xtr), lg(xte)
    grid = np.unique(np.quantile(xtr, np.linspace(0, 1, 401))); g1, g0 = sc(xtr, ctr, Etr, grid); j = int(np.argmax(0.5 * g1 + 0.5 * g0))
    out = {}
    for k in np.unique(K):
        m = K == k; h1, h0 = sc(xte[m], cte[m], E[m], grid); out[int(k)] = float((0.5 * h1 + 0.5 * h0)[j])
    return out


FAMS = ("cuhkdet", "prwdet", "cuhkseq", "prwseq", "cuhk", "prw", "market", "msmt", "rpif")   # order: longer prefixes first


def main():
    D = {f: {"accc": defaultdict(list), "accg": defaultdict(list), "accs": defaultdict(list),
             "selc": defaultdict(list), "sels": defaultdict(list)} for f in FAMS}
    for meta in sorted(glob.glob(os.path.join(V5, "*_meta.npz"))):
        key = os.path.basename(meta)[:-9]; a = os.path.join(V5, f"{key}_attn{SMT}.npz")
        if not os.path.exists(a):
            continue
        fam = next((f for f in FAMS if key.startswith(f)), None)
        if fam is None:
            continue
        z = dict(np.load(meta)); E, K = z["E"], z["K"]; y = dict(np.load(a)); S = [s for s in range(20) if f"conf_te_s{s}" in y]
        ac = frozen_per_k(z["raw_tr"], z["Vtr"], z["Etr"], z["raw_te"], z["Vte"], E, K)
        ag = frozen_per_k(z["sdse_tr"], z["Vtr"], z["Etr"], z["sdse_te"], z["Vte"], E, K, "logit")
        asm = defaultdict(list)
        for s in S:
            for k, v in frozen_per_k(y[f"conf_tr_s{s}"], y[f"corr_tr_s{s}"], y["Etr"], y[f"conf_te_s{s}"], y[f"corr_te_s{s}"], E, K, "prob").items():
                asm[k].append(v)
        g = D[fam]
        for k in ac:
            g["accc"][k].append(ac[k]); g["accg"][k].append(ag[k]); g["accs"][k].append(float(np.mean(asm[k])))
        for k in np.unique(K):                                        # forced-choice selection among present: cosine top-1 (Vte) vs SM choice (corr)
            mp = (K == k) & (E == 1)
            if mp.sum() < 20:
                continue
            g["selc"][int(k)].append(float(z["Vte"][mp].mean())); g["sels"][int(k)].append(float(np.mean([y[f"corr_te_s{s}"][mp].mean() for s in S])))
    out = {}
    for fam in FAMS:
        g = D[fam]
        if not g["accc"]:
            continue
        print(f"\n=== {fam}  ({'identity-count K; K=0=full gallery' if fam in ('market','msmt') else 'image-count K'}) ===")
        print(f"{'K':>6} {'enc':>4} | {'cos':>6} {'GS':>6} {'SM':>6} {'SM-GS':>7} | {'sel_cos':>7} {'sel_SM':>7}")
        out[fam] = {}
        for k in sorted(g["accc"]):
            row = (float(np.mean(g["accc"][k])), float(np.mean(g["accg"][k])), float(np.mean(g["accs"][k])))
            sl = (float(np.mean(g["selc"][k])), float(np.mean(g["sels"][k]))) if k in g["selc"] else (float("nan"), float("nan"))
            out[fam][k] = {"cos": row[0], "gs": row[1], "sm": row[2], "encs": len(g["accc"][k]), "sel_cos": sl[0], "sel_sm": sl[1]}
            print(f"{k:>6} {len(g['accc'][k]):>4} | {row[0]:6.3f} {row[1]:6.3f} {row[2]:6.3f} {row[2]-row[1]:+7.3f} | {sl[0]:7.3f} {sl[1]:7.3f}")
    json.dump(out, open(os.path.join(R, "results/setmatch/shift.json"), "w"), indent=1)
    print("\nwrote results/setmatch/shift.json  (per dataset; size-adaptive thresholds: see sm_sca.py)")


if __name__ == "__main__":
    main()
