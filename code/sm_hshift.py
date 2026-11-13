"""11-02 L5 hardness shift at fixed gallery size (PREREG_alpha_mmp.md, 'L5 H 이동').
Test episodes (CUHK-SYSU / PRW, ground-truth boxes): a random subset of m*K test images (target images included when
E = 1) is drawn and the K images with the highest max-score to the query are kept (top-K of a random subset: the
target survives only if it ranks there, so its position carries no construction cue). m = 1 is the random protocol;
m = 4, 16 give harder distractors at the same K. Models (ctx, ctx_cal, ctx_eff) are trained on the usual random
episodes; statistical rules use the source-free unlabelled null of the target. For the Sidak rules the nominal size is
either the shown candidate count N or the pre-shortlist count m*N ('shortlist-aware').
Per m: AUROC, frozen-threshold regret (sm_v5_eval.metrics, cells = m) and FAR / TPR at alpha = .05.
Writes results/setmatch/hshift_<data>_<enc>.json."""
import json
import os
import sys

import numpy as np
import torch

import sm_v5 as V
import sm_v5_eval as EV
import sm_v6 as S6

K0 = 200; MS = (1, 4, 8, 16, 32)                               # 8: unseen interpolation; 32: unseen extrapolation (PRW: capped by the test set, "full pool", m ~ 30.6)


def hard_eps(data, enc, rng):
    z = np.load(os.path.join(V.O, f"{data}_index.npz")); f = V.np.load(os.path.join(V.O, "feats", f"{data}_{enc}.npz"))["f"].astype(np.float32)
    img, pid = z["img"], z["pid"]; test = z["split"] == 1; tb = np.flatnonzero(test); timgs = np.unique(img[tb])
    q = z[sorted(k for k in z.files if k.startswith("G") and k.endswith("_q"))[0]]; Fq = f[q]
    box_of = {}
    for b in tb:
        box_of.setdefault(int(img[b]), []).append(b)
    nI = img.max() + 1; eps = []
    for i, qb in enumerate(q):
        s_all = f[tb] @ Fq[i]; Mx = np.full(nI, -9.0, np.float32); np.maximum.at(Mx, img[tb], s_all)
        tgt = np.unique(img[tb][pid[tb] == pid[qb]]); tgt = tgt[tgt != img[qb]]; others = timgs[~np.isin(timgs, np.r_[tgt, img[qb]])]
        for m in MS:
            for E in (0, 1):
                if E and not len(tgt):
                    continue
                pool = rng.choice(others, min(m * K0 - (len(tgt) if E else 0), len(others)), replace=False)
                if E:
                    pool = np.r_[pool, tgt]
                sel = pool[np.argsort(-Mx[pool])[:K0]]; it = np.concatenate([box_of[int(k)] for k in sel])
                s = f[it] @ Fq[i]; o = np.argsort(-s); tm = pid[it] == pid[qb]; Ee = int(tm.any())
                eps.append((i, it[o].astype(np.int32), tm[o], s[o].astype(np.float32), Ee, m, int(tm.sum())))
    return eps, f, Fq


def hard_pool(Dtr, data, n, trk, seed):
    """L5c training episodes: m*K random training images (m in {4, 16}), keep the K with the highest max-score"""
    img = np.load(os.path.join(V.O, f"{data}_index.npz"))["img"]; g = np.random.default_rng(seed); qs = np.flatnonzero(Dtr.ok); out, ms = [], []
    while len(out) < n:
        i = g.choice(qs); E = int(g.random() < 0.5); K = int(g.choice(trk)); m = int(g.choice([4, 16]))
        try:
            its, tm = Dtr.sample(i, E, m * K, g)
        except ValueError:
            continue
        sc = Dtr.Fg[its] @ Dtr.Fq[i]; im = img[its]; u, inv = np.unique(im, return_inverse=True); mx = np.full(len(u), -9.0); np.maximum.at(mx, inv, sc)
        sel = np.isin(im, u[np.argsort(-mx)[:K]]); its, tm, sc = its[sel], tm[sel], sc[sel]
        if len(its) < 2:
            continue
        o = np.argsort(-sc); out.append((i, its[o], tm[o], sc[o].astype(np.float32), int(tm.any()), m)); ms.append(m)
    return out


def shortlist_inputs(z, F0, mult):
    """competition inputs with the pre-shortlist size mult * N (mult = m)"""
    z = dict(z); NM = (z["N"] * mult).astype(np.float32); z["N"] = NM
    t = (S6.tail(F0, z["s"], NM[:, None]) * (z["idx"] >= 0)).astype(np.float32); z["t"] = t
    Cc = z["Cc"].copy(); Cc[:, 0], Cc[:, 1] = t[:, 0], t[:, 1]; z["Cc"] = Cc; Cr = z["Cr"].copy(); Cr[:, 0] = np.log(NM); z["Cr"] = Cr; return z


def main(data, enc, hardtrain=False, variants=("ctx", "ctx_cal", "ctx_eff"), tag="", seed0=0):
    rng = np.random.default_rng(0); trk = V.TRK[data]; kt = "-".join(map(str, trk)); qpp = 20 if data == "prw" else 1
    Dtr = V.scene(enc, "train", data, qpp); cpre = f"{data}_{enc}" + (f"_q{qpp}" if qpp != 1 else "")
    ztr = dict(np.load(os.path.join(V.O, "cache", f"{cpre}_k{kt}_tr.npz"))); zpl = dict(np.load(os.path.join(V.O, "cache", f"{cpre}_k{kt}_pool150000.npz")))
    eps, f, Fq = hard_eps(data, enc, rng); F0u = np.sort(S6.target_null("unlab", data, enc, ztr["F0"]).astype(np.float32))
    zte = S6.pack(eps, ztr["F0"]); E, M = zte["E"], zte["K"]; s1, N, top = zte["s"][:, 0], zte["N"], zte["tm"][:, 0]

    class Dte:
        pass
    Dte.Fg, Dte.Fq = f, Fq
    zsl = dict(zte); NM = (zte["N"] * zte["K"]).astype(np.float32); zsl["N"] = NM           # shortlist-aware: pre-shortlist size m N
    t = (S6.tail(ztr["F0"], zte["s"], NM[:, None]) * (zte["idx"] >= 0)).astype(np.float32); zsl["t"] = t
    Cc = zte["Cc"].copy(); Cc[:, 0], Cc[:, 1] = t[:, 0], t[:, 1]; zsl["Cc"] = Cc; Cr = zte["Cr"].copy(); Cr[:, 0] = np.log(NM); zsl["Cr"] = Cr
    if hardtrain:                                                  # half random (m = 1), half shortlisted (m = 4 / 16) training episodes
        zr = S6.pack(hard_pool(Dtr, data, 60000, trk, 11), ztr["F0"]); zh = shortlist_inputs(zr, ztr["F0"], zr["K"])   # K field holds m
        keys = ("idx", "s", "t", "tq", "N", "Cr", "Cc", "Cq", "qi", "P", "E", "tm"); half = len(zpl["qi"]) // 2
        zpl = {k: np.concatenate([zpl[k][:half], zh[k]]) for k in keys}
    Gtr, Gpl, Gte = S6.G(ztr, Dtr), S6.G(zpl, Dtr), S6.G(zte, Dte); Gsl = S6.G(zsl, Dte, share=Gte); S6.G.lmax = float(np.log(zpl["N"].max()))
    S6.G.lncut = float(np.log(np.quantile(zpl["N"], 0.7)))
    if any(v.endswith("_qn") for v in variants):
        S6.G.qt = S6.query_targets(Dtr, zpl)
    S6.G.a, S6.G.gam = S6.fit_neff(ztr); S6.G.F0 = torch.from_numpy(ztr["F0"].astype(np.float32)).to(V.dev); a, g = S6.G.a, S6.G.gam
    res = {"n_eps": len(eps), "E1_share": {str(m): float(E[M == m].mean()) for m in MS}}
    ap = np.full(len(eps), np.nan)
    for v in variants:
        runs, runs_sl = [], []
        for sd in range(seed0, seed0 + 3):
            net = S6.train(Gpl, v, sd, 2000); ctr, ktr, _, _ = S6.run(net, Gtr)
            for Gx, acc in ((Gte, runs), (Gsl, runs_sl)):
                cte, kte, apv, r1 = S6.run(net, Gx); acc.append(EV.metrics(ctr, ktr, ztr["E"], cte, kte, E, M, apv, r1, space="prob"))
        for name, rr in ((v, runs), (v + "|shortlist", runs_sl)):
            res[name] = {str(m): dict(auroc=float(np.mean([r[m]["auroc"] for r in rr])), regret=float(np.mean([max(r[m]["regret"].values()) for r in rr]))) for m in MS}
    lF = np.log(np.clip(np.searchsorted(F0u, s1, side="right") / len(F0u), 1e-300, 1)); al = 0.05
    e0 = ztr["E"] == 0; tau = np.quantile(ztr["s"][e0, 0], 1 - al)
    rules = {"cosine": s1 > tau, "sidak_Neff|unlab": -np.expm1(a * N ** g * lF) < al,
             "sidak_Neff|unlab|shortlist": -np.expm1(a * (M * N) ** g * lF) < al}
    for name, x in rules.items():
        res[name] = {str(m): dict(FAR=float(x[(M == m) & (E == 0)].mean()), TPR=float((x & top)[(M == m) & (E == 1)].mean())) for m in MS}
    print(data, enc, json.dumps({k: v for k, v in res.items() if k not in ("n_eps",)}), flush=True)
    p = os.path.join(V.O, f"hshift{'3' if hardtrain else '2'}{tag}_{data}_{enc}.json"); json.dump(res, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)


if __name__ == "__main__":
    ar = sys.argv + [""] * 4
    main(ar[1], ar[2], ar[3] == "hardtrain", tuple(ar[4].split(",")) if ar[4] else ("ctx", "ctx_cal", "ctx_eff"), ar[5], int(ar[6] or 0))
