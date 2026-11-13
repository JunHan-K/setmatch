"""11-02 C2 / C3 feasibility (PREREG_alpha_mmp.md, 'C2·C3 사전등록'). One encoder per run.
C2 effective competition on the TEST splits (labels used for analysis only): target-absent episodes, N_eff = a N^gamma
fitted by maximum likelihood (-N_eff log F0(s_1) ~ Exp(1)), 95% interval from a query bootstrap; conditions
  orig   every gallery item
  dedup  Market: one image per identity; scene data: one box per image (removes same-scene correlation)
  synth  the maximum of N i.i.d. draws from F0 (estimator check, expect gamma ~ 1)
C3 null estimates for targets cuhk / prw: F0 from a labelled source (market train; cuhk train for prw), from
co-occurring boxes in the target test images (no identity labels, IoU <= 0.1), the target's own labelled train split,
and the oracle (target test query vs impostors). T1 compares each with the oracle; T2 predicts the target-absent
maximum per gallery size from F0^(a N^gamma) with (a, gamma) transferred from the source.
Writes results/setmatch/comp_<enc>.json."""
import argparse
import json
import os

import numpy as np

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
O = os.path.join(R, "results/setmatch")
G = np.linspace(0.2, 1.3, 111)


def feats(name, enc):
    return np.load(os.path.join(O, "feats", f"{name}_{enc}.npz"))["f"].astype(np.float32)


def fit(x, N):
    """max-likelihood (a, gamma) over the grid; x = -log F0(s_1), N = nominal size"""
    x = np.maximum(x, 1e-12); Ng = N[None] ** G[:, None]; a = len(x) / (Ng * x[None]).sum(1)
    ll = np.log(a) * len(x) + G * np.log(N).sum() - (a[:, None] * Ng * x[None]).sum(1); j = int(np.argmax(ll)); return float(a[j]), float(G[j])


def boot(x, N, nq, rng, B=200):
    """x, N: [Q, nK] per query; returns gamma point estimate and 95% interval"""
    m = np.isfinite(x); g0 = fit(x[m], N[m])[1]; gs = []
    for _ in range(B):
        q = rng.integers(0, nq, nq); xb, Nb = x[q], N[q]; mb = np.isfinite(xb); gs.append(fit(xb[mb], Nb[mb])[1])
    return dict(gamma=g0, lo=float(np.quantile(gs, 0.025)), hi=float(np.quantile(gs, 0.975)))


def xof(F0, s):
    return -np.log(np.clip(np.searchsorted(F0, s, side="right") / len(F0), 1e-9, 1 - 1e-9))


def sub(v, n, rng):
    v = np.asarray(v, np.float32).ravel(); return np.sort(v if len(v) <= n else rng.choice(v, n, replace=False))


# ---------------------------------------------------------------- episodes (target absent) per dataset
def qstd(S, cols):
    """per-query robust standardisation over the query's own gallery scores (median / MAD), no labels"""
    med = np.median(S[:, cols], 1, keepdims=True); mad = np.median(np.abs(S[:, cols] - med), 1, keepdims=True) * 1.4826 + 1e-6
    return ((S - med) / mad).astype(np.float32)


def market_eps(enc, rng, std=False):
    z = np.load(os.path.join(O, "feats", f"market_{enc}.npz")); f = z["f"].astype(np.float32); pid, sp = z["pid"], z["split"]
    g = np.flatnonzero((sp == 1) & (pid > 0)); q = np.flatnonzero(sp == 2); ids = np.unique(pid[g]); S = f[q] @ f[g].T
    if std:
        S = qstd(S, np.arange(S.shape[1]))
    col = np.searchsorted(ids, pid[g]); nimg = np.bincount(col, minlength=len(ids))
    Mx = np.full((len(q), len(ids)), -99, np.float32); np.maximum.at(Mx.T, col, S.T)       # per query x identity max
    one = np.array([rng.choice(np.flatnonzero(col == c)) for c in range(len(ids))]); S1 = S[:, one]
    imp = sub(S[pid[q][:, None] != pid[g][None]], 300000, rng); Ks = (10, 50, 100, 200, 0)
    so, sd, No, Nd = [np.full((len(q), len(Ks)), np.nan) for _ in range(4)]
    for i, p in enumerate(pid[q]):
        pool = np.flatnonzero(ids != p)
        for j, K in enumerate(Ks):
            sel = pool if K == 0 else rng.choice(pool, K, replace=False)
            so[i, j], No[i, j] = Mx[i, sel].max(), nimg[sel].sum(); sd[i, j], Nd[i, j] = S1[i, sel].max(), len(sel)
    return dict(Ks=Ks, so=so, No=No, sd=sd, Nd=Nd, imp=imp)


def scene_eps(name, enc, rng, std=False):
    z = np.load(os.path.join(O, f"{name}_index.npz")); f = feats(name, enc); img, pid = z["img"], z["pid"]
    Gs = sorted(int(k[1:-2]) for k in z.files if k.startswith("G") and k.endswith("_q")); q = z[f"G{Gs[0]}_q"]
    nI = len(z["imgs"]); gal = z["gal"].astype(bool) if "gal" in z.files else np.ones(len(img), bool)   # proposals only
    nbox = np.bincount(img[gal], minlength=nI); S = f[q] @ f.T
    if std:
        S = qstd(S, np.flatnonzero((z["split"] == 1) & gal))
    S[:, ~gal] = -99
    Mx = np.full((len(q), nI), -99, np.float32); np.maximum.at(Mx.T, img, S.T)
    perm = rng.permutation(np.flatnonzero(gal)); _, first = np.unique(img[perm], return_index=True); one = np.full(nI, -1); one[img[perm][first]] = perm[first]
    S1 = np.where(one[None] >= 0, S[:, np.maximum(one, 0)], -99)
    tb = np.flatnonzero((z["split"] == 1) & gal)
    imp = sub(S[:, tb][pid[q][:, None] != pid[tb][None]], 300000, rng)
    so, sd, No, Nd = [np.full((len(q), len(Gs)), np.nan) for _ in range(4)]
    for j, K in enumerate(Gs):
        gal, has = z[f"G{K}_gal"], z[f"G{K}_has"]
        for i in range(len(q)):
            g = gal[i][~has[i]]; g = g[nbox[g] > 0]
            if len(g) < 2:
                continue
            so[i, j], No[i, j] = Mx[i, g].max(), nbox[g].sum(); sd[i, j], Nd[i, j] = S1[i, g].max(), len(g)
    return dict(Ks=tuple(Gs), so=so, No=No, sd=sd, Nd=Nd, imp=imp, z=z, f=f)


def cooccur(z, f, test, rng, iou_max=0.1, nimg=None):
    """unlabelled null: pairs of distinct boxes in the same target test image with IoU <= iou_max (duplicate-detection
    guard); nimg: use only this many randomly chosen images (sample-efficiency study)"""
    img, bx = z["img"], z["xywh"].astype(float); out = []; ks = np.unique(img[test])
    if nimg:
        ks = rng.choice(ks, min(nimg, len(ks)), replace=False)
    for k in ks:
        b = np.flatnonzero((img == k) & test)
        if len(b) < 2:
            continue
        x1, y1, x2, y2 = bx[b, 0], bx[b, 1], bx[b, 0] + bx[b, 2], bx[b, 1] + bx[b, 3]
        iw = np.clip(np.minimum(x2[:, None], x2[None]) - np.maximum(x1[:, None], x1[None]), 0, None)
        ih = np.clip(np.minimum(y2[:, None], y2[None]) - np.maximum(y1[:, None], y1[None]), 0, None); inter = iw * ih
        iou = inter / ((bx[b, 2] * bx[b, 3])[:, None] + (bx[b, 2] * bx[b, 3])[None] - inter + 1e-9)
        i, j = np.triu_indices(len(b), 1); ok = iou[i, j] <= iou_max
        out.append(np.einsum("nd,nd->n", f[b[i[ok]]], f[b[j[ok]]]))
    return sub(np.concatenate(out) if out else np.zeros(1), 300000, rng)


def labelled_null(name, enc, rng):
    """impostor scores of a labelled TRAIN split (source / in-domain reference)"""
    if name == "market":
        z = np.load(os.path.join(O, "feats", f"market_{enc}.npz")); f = z["f"].astype(np.float32); pid, sp = z["pid"], z["split"]
        b = np.flatnonzero((sp == 0) & (pid > 0))
    else:
        z = np.load(os.path.join(O, f"{name}_index.npz")); f = feats(name, enc); pid = z["pid"]; b = np.flatnonzero((z["split"] == 0) & (pid > 0))
    i, j = rng.choice(b, 600000), rng.choice(b, 600000); ok = pid[i] != pid[j]
    return sub(np.einsum("nd,nd->n", f[i[ok]], f[j[ok]]), 300000, rng)


def t1(F, Fo):
    grid = np.unique(np.r_[F, Fo]); cdf = lambda A: np.searchsorted(A, grid, side="right") / len(A)
    qs = (0.9, 0.95, 0.99, 0.999); qo = np.quantile(Fo, qs); qe = np.quantile(F, qs)
    exc = {str(p): float(np.mean(F > np.quantile(Fo, p)) - (1 - p)) for p in (0.99, 0.999)}
    return dict(KS=float(np.abs(cdf(F) - cdf(Fo)).max()), W1=float(np.mean(np.abs(np.quantile(F, np.linspace(0.0005, 0.9995, 2000)) - np.quantile(Fo, np.linspace(0.0005, 0.9995, 2000))))),
                q_err={str(p): float(a - b) for p, a, b in zip(qs, qe, qo)}, exceed_err=exc)


def t2(F0, a, g, so, No, Ks):
    """KS between the empirical target-absent maximum and mean_i F0(s)^(a N_i^g), per gallery size"""
    out = {}
    for j, K in enumerate(Ks):
        m = np.isfinite(so[:, j]); s = np.sort(so[m, j]); N = No[m, j]
        if len(s) < 20:
            continue
        lF = -xof(F0, s); pred = np.exp(np.outer(lF, a * N ** g)).mean(1); emp = np.arange(1, len(s) + 1) / len(s)
        out[str(K)] = float(np.max(np.maximum(np.abs(pred - emp), np.abs(pred - (emp - 1 / len(s))))))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--enc", required=True); ap.add_argument("--det", action="store_true"); a = ap.parse_args()
    rng = np.random.default_rng(0); res = {"C2": {}, "C3": {}}
    if a.det:                                                      # C2 on frozen detector proposals only
        eps = {d: scene_eps(d, a.enc, rng) for d in ("cuhkdet", "prwdet")}
    else:
        eps = {"market": market_eps(a.enc, rng), "cuhk": scene_eps("cuhk", a.enc, rng), "prw": scene_eps("prw", a.enc, rng)}
    for d, e in eps.items():                                       # ---- C2
        F0 = e["imp"]; xo, xd = xof(F0, e["so"]), xof(F0, e["sd"]); nq = len(xo)
        u = rng.random(e["No"].shape); xs = xof(F0, np.quantile(F0, np.clip(u ** (1 / np.nan_to_num(e["No"], nan=1)), 0, 1)))
        xs[~np.isfinite(e["so"])] = np.nan
        res["C2"][d] = {c: boot(x, N, nq, rng) for c, x, N in (("orig", xo, e["No"]), ("dedup", xd, e["Nd"]), ("synth", xs, e["No"]))}
        print("C2", d, {c: (round(v["gamma"], 2), round(v["lo"], 2), round(v["hi"], 2)) for c, v in res["C2"][d].items()}, flush=True)
    if a.det:
        p = os.path.join(O, f"comp_det_{a.enc}.json"); json.dump(res, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p); return
    srcfit = {}
    for s in ("market", "cuhk"):                                   # source (a, gamma) from the labelled train episodes of the source
        c = np.load(os.path.join(O, "cache", f"{s}_{a.enc}_k{'10-20-50-100-200-0' if s == 'market' else '50-100-200-500'}_tr.npz"))
        e0 = c["E"] == 0; srcfit[s] = (c["F0"], *fit(xof(c["F0"], c["s"][e0, 0]), c["N"][e0]))
    for tgt, srcs in (("cuhk", ("market",)), ("prw", ("market", "cuhk"))):   # ---- C3
        e = eps[tgt]; z = e["z"]; test = z["split"] == 1; Fo = e["imp"]
        nulls = {"unlab": cooccur(z, e["f"], test, rng), "intrain": labelled_null(tgt, a.enc, rng)}
        for s in srcs:
            nulls[f"src_{s}"] = labelled_null(s, a.enc, rng)
        oa, og = fit(xof(Fo, e["so"][np.isfinite(e["so"])]), e["No"][np.isfinite(e["so"])])
        r = {"T1": {k: t1(v, Fo) for k, v in nulls.items()}, "T2": {}, "oracle_fit": (oa, og)}
        for s in srcs:
            _, sa, sg = srcfit[s]
            for k, v in list(nulls.items()) + [("oracle", Fo)]:
                r["T2"][f"{k}|law_{s}"] = t2(v, sa, sg, e["so"], e["No"], e["Ks"])
            r["T2"][f"oracle|indep"] = t2(Fo, 1.0, 1.0, e["so"], e["No"], e["Ks"])
        r["T2"]["oracle|law_oracle"] = t2(Fo, oa, og, e["so"], e["No"], e["Ks"])
        res["C3"][tgt] = r
        print("C3", tgt, "T1 KS", {k: round(v["KS"], 3) for k, v in r["T1"].items()}, "q.99 err", {k: round(v["q_err"]["0.99"], 3) for k, v in r["T1"].items()}, flush=True)
        print("   T2 mean KS", {k: round(float(np.mean(list(v.values()))), 3) for k, v in r["T2"].items()}, flush=True)
    p = os.path.join(O, f"comp_{a.enc}.json"); json.dump(res, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)


if __name__ == "__main__":
    main()
