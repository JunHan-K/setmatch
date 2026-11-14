"""11-02 SetWiseMatch v5: image / box-level candidates, open-world decision, Market-1501 and CUHK-SYSU (GT boxes).
Candidates are gallery IMAGES (Market) or person BOXES inside gallery images (CUHK-SYSU), so one identity can occur
several times in a candidate set -- the situation in which candidate-candidate evidence can matter.
Episodes
  market: K = number of gallery identities (all their images; target images from other cameras only); test K in
          {10, 50, 100, 200, full}, train K in {10, 20, 50, 100, 200, full} on train identities.
  cuhk:   K = number of gallery images (official TestG50..4000 lists at test; E = 0 removes the images that contain the
          target); train: Train-split persons that never appear in test images, G in {50, 100, 200, 500}.
Every method returns (choice, confidence); an open-world decision is correct iff it accepts a true target or rejects
when no target is present. Baselines choose the cosine top-1.
Variants (identical pair tokens [q*c, |q-c|, c, s]):
  pair_bce  independent per-pair verifier (sigmoid), no set-relative input
  pair_null softmax over candidates + a learned NULL vector, no set-relative input
  ctx       + set context (log N, score quantiles), rank / percentile features, set-pooled NULL token
  attn      + Transformer attention between candidates
  setmatch  + candidate-affinity bias cos(c_i, c_j) in every attention head (learned per-head scale)
  <kind>_cal (v6) cardinality-calibrated inputs: the set context drops log N and score quantiles and uses the
            order-statistic tail t(s) = -log10(1 - F0(s)^N) of the top scores (F0 = impostor-score CDF from train
            episodes) plus N-free shape statistics; the percentile feature is replaced by t(s_i). Nothing grows with N.
  evt       (v6 baselines, cosine top-1) wscore: Meta-Recognition (Scheirer et al., TPAMI 2011) -- Weibull fitted to
            the top-n non-match tail s_2..s_{n+1} of each list, confidence = its CDF at s_1; qnull: query-local null --
            Gaussian fitted to the ranks below the top-m, confidence = -log10(1 - Phi(z_1)^N). No training.
  --trk / --tag (v6) override the training K grid; outputs get the tag appended to the variant name.
Outputs results/setmatch/v5/<data>_<enc>_<variant>.npz (per test episode: conf, correct, ap, r1) and
<data>_<enc>_meta.npz (E, K, cosine baselines) written by the 'base' run."""
import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kurtosis, norm, skew, weibull_min

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
O = os.path.join(R, "results/setmatch")
M = 64
dev = "cuda"
SAMPLER_VER = 9   # bump on any change to sample()/episodes()/scene build so stale episode caches are rebuilt (see sm_v6.cached and gs_stats)


# ---------------------------------------------------------------- data
class Dom:
    """queries Fq/qpid, gallery items Fg/gpid; sample(i, E, K, rng) -> (items, target mask)"""


def market(enc, split, name="market"):                            # 11-04: name="rpif" (RPIfield, CF) uses the same layout
    z = np.load(os.path.join(O, "feats", f"{name}_{enc}.npz")); f = z["f"].astype(np.float32); f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-12; pid, cam, sp = z["pid"], z["cam"], z["split"]   # L2 so Fg @ Fq is cosine (features are already unit-norm; idempotent)
    D = Dom(); rng = np.random.default_rng(0)
    if split == "train":
        g = np.flatnonzero((sp == 0) & (pid > 0)); q = []
        for key in np.unique(pid[g] * 100 + cam[g]):            # 11-06: *100 (was *10) so pid*k+cam does not collide for MSMT17's 15 cameras
            q.append(rng.choice(g[pid[g] * 100 + cam[g] == key]))
        q = np.array(q)
    else:
        g = np.flatnonzero((sp == 1) & (pid > 0)); q = np.flatnonzero(sp == 2)
    D.Fq, D.qpid, D.qcam, D.qimg = f[q], pid[q], cam[q], q
    D.Fg, D.gpid, D.gcam = f[g], pid[g], cam[g]
    D.ids = np.unique(D.gpid); D.items = {p: np.flatnonzero(D.gpid == p) for p in D.ids}
    D.ok = np.array([(D.gcam[D.items.get(p, np.zeros(0, int))] != c).any() for p, c in zip(D.qpid, D.qcam)])
    gpos = {int(x): j for j, x in enumerate(g)}                    # original index -> gallery position
    D.qpos = np.array([gpos.get(int(x), -1) for x in q])          # the query's own gallery position (train: in the gallery; test: -1) -> drop the self-match
    byc = {}                                                       # gallery identities grouped by image count (for an exact-size K=0 decoy)
    for qq in D.ids:
        byc.setdefault(len(D.items[qq]), []).append(int(qq))
    idcnt = np.array([len(D.items[qq]) for qq in D.ids])

    def present_items(ii, K, rng):                                 # K-1 distractor identities + the "focus" identity's images (identical layout for every query)
        p = D.qpid[ii]; pool = D.ids[D.ids != p]
        sel = pool if K == 0 else rng.choice(pool, min(K - 1, len(pool)), replace=False)
        di = np.concatenate([D.items[s] for s in sel]) if len(sel) else np.zeros(0, int)
        t = D.items[p]; t = t[D.gcam[t] != D.qcam[ii]]
        return np.concatenate([di, t])

    def sample(i, E, K, rng):                                      # 11-13 LEAK FIX v3: absent = the SAME present-construction run for ANOTHER query whose identity is absent here -> present/absent share one distribution
        p = D.qpid[i]
        if K == 0:                                                 # full gallery: present keeps the target & drops a decoy identity, absent drops the target & keeps the decoy. One identity removed per side (candidate-IDENTITY count matches); the present person's SAME-camera boxes are excluded (Market cross-camera rule, now applied at EVERY K -- A1); the trim matches the candidate-BOX count.
            qc = D.qcam[i]; sp = len(D.items[p])
            xcam = lambda q: bool((D.gcam[D.items[q]] != qc).any())   # decoy must have >=1 box OUTSIDE the query camera, else its same-camera boxes are all removed when it is the absent focus -> it vanishes and the candidate-identity count mismatches (A1 case A)
            same = [q for q in byc.get(sp, ()) if q != p and xcam(q)]
            if same:
                d = int(same[int(rng.integers(len(same)))]); sd = sp
            else:                                                  # no exact-size cross-camera decoy -> nearest count among cross-camera identities (market ~1.3-2% |delta|~1; msmt ~3-4% heavier tail), then trim the larger side so N still matches
                order = np.argsort(np.abs(idcnt - sp)); d = next((int(D.ids[o]) for o in order if D.ids[o] != p and xcam(int(D.ids[o]))), None)
                if d is None:                                      # degenerate: no cross-camera identity at all -> any decoy (episode will still be dropped if the focus ends up empty)
                    d = next(int(D.ids[o]) for o in order if D.ids[o] != p)
                sd = len(D.items[d])
            foc = p if E else d                                    # the "present person" in this episode (target when present, decoy when absent)
            it = np.concatenate([D.items[s] for s in D.ids[D.ids != (d if E else p)]])
            it = it[~((D.gpid[it] == foc) & (D.gcam[it] == qc))]   # A1: the present person's same-camera boxes are never candidates (consistent with present_items at K>0); for the train query this also removes its own box, so no self-match
            a_p = int((D.gcam[D.items[p]] == qc).sum()); a_d = int((D.gcam[D.items[d]] == qc).sum())
            rem = max(sd + a_p, sp + a_d); r = rem - ((sd + a_p) if E else (sp + a_d))   # match both sides to the SAME removed-box count (present removed decoy sd + target same-cam a_p; absent removed target sp + decoy same-cam a_d) so present N == absent N
            if r > 0:
                pit = D.gpid[it]; _, inv, cn = np.unique(pit, return_inverse=True, return_counts=True)
                budget = cn - 1; drop = []                         # never remove the present person's boxes, never an identity's LAST box (keeps the candidate-identity count matched too)
                for bi in rng.permutation(len(it)):
                    if len(drop) >= r:
                        break
                    if pit[bi] != foc and budget[inv[bi]] > 0:
                        budget[inv[bi]] -= 1; drop.append(bi)
                if len(drop) == r:
                    it = np.delete(it, drop)
            return it, (D.gpid[it] == p)
        if E:
            it = present_items(i, K, rng)
        else:
            it = None
            for _ in range(64):
                j = int(rng.integers(len(D.qpid)))
                if D.qpid[j] == p:
                    continue
                cand = present_items(j, K, rng)
                if not np.any(D.gpid[cand] == p):                  # our target genuinely absent from the borrowed layout
                    it = cand; break
            if it is None:                                         # fallback (rare): K distractor identities
                pool = D.ids[D.ids != p]; sel = pool if K == 0 else rng.choice(pool, min(K, len(pool)), replace=False)
                it = np.concatenate([D.items[s] for s in sel])
        return it, (D.gpid[it] == p)
    D.sample = sample
    return D


def scene(enc, split, name="cuhk", qpp=1):
    """CUHK-SYSU / PRW with GT boxes (cuhk, prw) or frozen proposals (cuhkdet, prwdet: gallery = proposals, queries =
    GT boxes stored with gal = 0)"""
    z = np.load(os.path.join(O, f"{name}_index.npz")); f = np.load(os.path.join(O, "feats", f"{name}_{enc}.npz"))["f"].astype(np.float32)
    f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-12            # L2 so Fg @ Fq is cosine (features are already unit-norm; idempotent)
    img, pid, sp = z["img"], z["pid"], z["split"]; det = "gal" in z; gal = z["gal"] if det else np.ones(len(img), bool)
    boxes_of = {}
    for b in np.flatnonzero(gal):
        boxes_of.setdefault(int(img[b]), []).append(b)
    boxes_of = {k: np.array(v) for k, v in boxes_of.items()}
    D = Dom(); D.Fg, D.gpid, D.det = f, pid, det
    if split == "train":
        test_p = set(pid[(sp == 1) & (pid > 0)]); rng = np.random.default_rng(0); cand_q = []
        tr_imgs = np.array(sorted(k for k in boxes_of if (sp[boxes_of[k]] == 0).all()))
        src = ~gal if det else gal                                 # query / target-image source: GT boxes
        for p in np.unique(pid[(sp == 0) & (pid > 0) & src]):
            bs = np.flatnonzero((pid == p) & (sp == 0) & src)
            if p in test_p or len(bs) < 2:
                continue
            cand_q.extend(rng.choice(bs, min(qpp, len(bs)), replace=False))   # qpp query boxes per training person
        q = np.array(cand_q); D.pimgs = {}
        for b in q:
            D.pimgs[b] = np.unique(img[(pid == pid[b]) & (img != img[b]) & src])
        D.Fq, D.qpid, D.qbox = f[q], pid[q], q; D.ok = np.ones(len(q), bool)

        def present_items(ii, G, rng):                                   # G whole images: the focus identity's images + target-free fill (identical layout for every query)
            b = D.qbox[ii]; t = D.pimgs[b]; t = t[np.isin(t, tr_imgs)]
            free = tr_imgs[~np.isin(tr_imgs, np.r_[t, img[b]])]
            nfill = max(G - len(t), 0)
            ims = np.r_[t[:G], rng.choice(free, min(nfill, len(free)), replace=False)] if nfill else t[:G]
            return np.concatenate([boxes_of[int(k)] for k in ims]) if len(ims) else np.zeros(0, int)

        def sample(i, E, G, rng):                                        # 11-13 LEAK FIX v3: absent = the SAME layout built for ANOTHER query lacking this target -> one distribution for present/absent
            p = pid[D.qbox[i]]
            if E:
                it = present_items(i, G, rng)
            else:
                it = None
                for _ in range(64):
                    j = int(rng.integers(len(D.qbox)))
                    if pid[D.qbox[j]] == p:
                        continue
                    cand = present_items(j, G, rng)
                    if not np.any(pid[cand] == p):
                        it = cand; break
                if it is None:                                           # fallback (rare): G target-free whole images
                    free = tr_imgs[~np.isin(tr_imgs, np.r_[D.pimgs[D.qbox[i]], img[D.qbox[i]]])]
                    ims = rng.choice(free, min(G, len(free)), replace=False)
                    it = np.concatenate([boxes_of[int(k)] for k in ims]) if len(ims) else np.zeros(0, int)
            return it, (pid[it] == p)
        D.sample = sample
    else:
        D.test = {}; Gs = sorted(int(k[1:-2]) for k in z.files if k.startswith("G") and k.endswith("_q"))
        for G in Gs:
            D.test[G] = (z[f"G{G}_q"], z[f"G{G}_gal"], z[f"G{G}_has"])
        q = z[f"G{Gs[0]}_q"]; D.Fq, D.qpid, D.qbox = f[q], pid[q], q; D.ok = np.ones(len(q), bool); D.Gs = tuple(Gs)
        assert all((z[f"G{G}_q"] == q).all() for G in D.test), "TestG query order differs across gallery sizes"
        # 11-13 LEAK FIX v3g: reconstruct present AND absent by ONE identical law (query-blind-indistinguishable), over the
        # TEST split only, never the query's own image, and never a true target in an absent episode. Gallery FRAMES are
        # chosen by where identities APPEAR (GT boxes for det, so detection cannot bias which frames enter); the candidate
        # boxes in those frames are the gallery boxes (proposals for det, GT otherwise).
        tgal = np.flatnonzero(gal & (sp == 1))                   # candidate boxes: TEST split only (no train leakage)
        tboxes_of = {}
        for b in tgal:
            tboxes_of.setdefault(int(img[b]), []).append(b)
        tboxes_of = {k: np.array(v) for k, v in tboxes_of.items()}
        test_imgs = np.fromiter((k for k in tboxes_of), dtype=np.int64, count=len(tboxes_of))
        appb = np.flatnonzero((~gal) & (sp == 1)) if det else tgal   # where identities appear: GT boxes for det -> frame choice is detection-independent
        gtpid_t, gtimg_t = pid[appb], img[appb]
        appear = {}
        for b in appb:
            appear.setdefault(int(pid[b]), set()).add(int(img[b]))
        appear = {k: np.array(sorted(v), dtype=np.int64) for k, v in appear.items()}
        qids = np.array(sorted(set(int(p) for p in D.qpid)), dtype=np.int64)
        E0 = np.zeros(0, np.int64)

        def build(focus, G, excl, qimg, rng):                    # focus identity's appearance frames + distractor frames; the excl identity's frames (and the query frame) are fully excluded so excl can never appear as a candidate
            foc = appear.get(int(focus), E0); xf = appear.get(int(excl), E0); dropped = -1
            foc = foc[~np.isin(foc, xf)]                         # drop frames where focus co-occurs with excl -> no excl box leaks in (absent stays target-free)
            if (foc == qimg).any():
                foc = foc[foc != qimg]                           # present: the query's own frame is never a candidate
            elif len(foc) > 1:
                di = int(rng.integers(len(foc))); dropped = int(foc[di]); foc = np.delete(foc, di)   # absent: drop one focus frame to match present losing the query frame
            bad = np.r_[foc, xf, np.array([qimg, dropped], np.int64)]
            pool = test_imgs[~np.isin(test_imgs, bad)]
            incl = foc[:G]                                        # focus frames ACTUALLY placed in the gallery (xf-co-occurring frames and qimg already removed)
            nd = max(G - len(incl), 0)
            fill = rng.choice(pool, min(nd, len(pool)), replace=False) if nd and len(pool) else E0
            ims = np.r_[incl, fill]
            cb = [tboxes_of[int(k)] for k in ims if int(k) in tboxes_of]   # candidate boxes (a GT-only frame with no proposal contributes none, but still counts for recall)
            return (np.concatenate(cb).astype(np.int64) if cb else E0), incl

        def sample(i, E, G, rng):                                # symmetric decoy; present = target focus (decoy excluded), absent = decoy focus (target excluded)
            qp = int(D.qpid[i]); qimg = int(img[D.qbox[i]])
            def pick_decoy():
                d = qp
                while d == qp:
                    d = int(qids[rng.integers(len(qids))])
                return d
            if E:
                it = E0; ng = 0
                for _ in range(8):                               # A2: re-draw the decoy if it co-occurs with EVERY target frame (ngt==0 -> no GT target in the gallery). A genuine detection miss (ngt>0 but no proposal) is kept.
                    it, incl = build(qp, G, pick_decoy(), qimg, rng)
                    ng = int(((gtpid_t == qp) & np.isin(gtimg_t, incl)).sum())
                    if ng > 0:
                        break
                D._ngt_det = ng                                  # may stay 0 if the target has no gallery frame besides the query -> episodes() drops this present episode (not a detection miss)
            else:
                it, incl = build(pick_decoy(), G, qp, qimg, rng); D._ngt_det = 0
            return it, (pid[it] == qp)
        D.sample = sample
        if det:                                                  # det recall denominator (GT-defined, so detector misses still cost recall); computed per-episode from the exact included frames in sample() just above
            def ngt(i, G):
                return int(getattr(D, "_ngt_det", 0))
            D.ngt = ngt
    return D


TRK = dict(market=(10, 20, 50, 100, 200, 0), cuhk=(50, 100, 200, 500), cuhkdet=(50, 100, 200, 500), prw=(50, 100, 200, 500, 1000),
           prwdet=(50, 100, 200, 500, 1000), cuhkseq=(50, 100, 200, 500), prwseq=(50, 100, 200, 500, 1000))
TEK = dict(market=(10, 50, 100, 200, 0), cuhk=(50, 100, 500, 1000, 2000, 4000), rpif=(10, 100, 500, 1000, 0))
TRK["rpif"] = (10, 20, 50, 100, 200)                               # 11-04 CF: RPIfield confirmatory evaluation
TRK["msmt"] = (10, 20, 50, 100, 200, 0); TEK["msmt"] = (10, 100, 500, 1000, 0)   # 11-04 CF2: MSMT17 (0 = full gallery)


def restrict_ids(D, n, seed=0):
    """11-04 IDS: keep only n training identities as possible queries (their positives with them); impostors unchanged"""
    ids = np.unique(D.qpid[D.ok]); keep = np.random.default_rng(seed).choice(ids, min(n, len(ids)), replace=False)
    D.ok = D.ok & np.isin(D.qpid, keep); return D


def episodes(D, Ks, n, seed):
    g = np.random.default_rng(seed); qs = np.flatnonzero(D.ok); qs = qs if n is None else g.choice(qs, n, replace=n > len(qs))
    out = []
    for i in qs:
        for K in Ks:
            for E in (0, 1):
                it, tm = D.sample(i, E, K, g)
                assert E or not tm.any(), f"absent episode (K={K}) contains a true target"   # invariant: no-match episodes are genuinely target-free
                if len(it) < 2 or (E and not tm.any() and not getattr(D, "det", False)):
                    continue
                s = D.Fg[it] @ D.Fq[i]; o = np.argsort(-s)
                ngt = D.ngt(i, K) if hasattr(D, "ngt") and E else int(tm.sum())   # proposals: missed targets cost recall
                if getattr(D, "det", False) and E and ngt == 0:   # A2: a present episode with NO GT target in the gallery is not a detection miss -> drop it (keeps ngt>0 & tm.any()==False, the real detector miss)
                    continue
                out.append((i, it[o].astype(np.int32), tm[o], s[o].astype(np.float32), E, K, ngt))
    return out


# ---------------------------------------------------------------- model
class Blk(nn.Module):
    def __init__(self, h, nh=4, aff=False):
        super().__init__(); self.nh = nh; self.qkv = nn.Linear(h, 3 * h); self.o = nn.Linear(h, h)
        self.f = nn.Sequential(nn.Linear(h, 4 * h), nn.GELU(), nn.Linear(4 * h, h)); self.n1 = nn.LayerNorm(h); self.n2 = nn.LayerNorm(h)
        self.aff = nn.Parameter(torch.zeros(nh)) if aff else None

    def forward(self, t, A, pad):                                   # t [B, L, h]; A [B, L, L]; pad [B, L] True = pad
        B, L, h = t.shape; q, k, v = self.qkv(self.n1(t)).view(B, L, 3, self.nh, h // self.nh).permute(2, 0, 3, 1, 4)
        lg = q @ k.transpose(-1, -2) / (h // self.nh) ** 0.5
        if self.aff is not None:
            lg = lg + self.aff.view(1, -1, 1, 1) * A[:, None]
        lg = lg.masked_fill(pad[:, None, None, :], -1e4); t = t + self.o((lg.softmax(-1) @ v).transpose(1, 2).reshape(B, L, h))
        return t + self.f(self.n2(t))


class Net(nn.Module):
    def __init__(self, d, kind, h=128):
        super().__init__(); self.kind = kind; self.cal = kind.endswith("_cal"); kind = kind.removesuffix("_cal")
        self.cnull = kind == "attn_cn"; kind = kind.removesuffix("_cn")   # 11-04 ablation: attn with a constant (set-independent) no-match token
        self.setrel = kind in ("ctx", "attn", "setmatch")
        self.tok = nn.Linear(3 * d + 3, h); self.ctx = nn.Linear(7, h); self.null = nn.Parameter(torch.zeros(h))
        self.pool = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, h))
        if kind in ("attn", "setmatch"):
            self.blocks = nn.ModuleList([Blk(h, aff=kind == "setmatch") for _ in range(3)])
        else:
            self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 4 * h), nn.GELU(), nn.Linear(4 * h, h)) for _ in range(3)])
        self.out = nn.Linear(h, 1)

    def forward(self, X, C, A, pad):
        if not self.setrel:
            X = torch.cat([X[..., :-2], torch.zeros_like(X[..., -2:])], -1)
        e = self.tok(X)
        if self.setrel:
            e = e + self.ctx(C)[:, None]
            m = (~pad).float()[..., None]; mean = (e * m).sum(1) / m.sum(1); mx = e.masked_fill(pad[..., None], -1e4).max(1).values
            z = self.null.expand(len(e), -1) if self.cnull else self.null + self.pool(torch.cat([mean, mx], -1))
        else:
            z = self.null.expand(len(e), -1)
        t = torch.cat([e, z[:, None]], 1); padx = torch.cat([pad, torch.zeros_like(pad[:, :1])], 1)
        Ax = nn.functional.pad(A, (0, 1, 0, 1))
        for b in self.blocks:
            t = b(t, Ax, padx) if isinstance(b, Blk) else t + b(t)
        return self.out(t)[..., 0].masked_fill(padx, -1e4)


def tail(F0, s, N):                                                # -log10 P(max of N impostors >= s), scaled to ~[0, 2]
    F = np.clip(np.searchsorted(F0, s, side="right") / len(F0), 1e-9, 1 - 1e-9)
    return np.clip(-np.log10(-np.expm1(N * np.log(F))) / 6, 0, 2)


def batch(D, eps, cal=False):
    d = D.Fq.shape[1]; B = len(eps); X = np.zeros((B, M, 3 * d + 3), np.float32); C = np.zeros((B, 7), np.float32)
    A = np.zeros((B, M, M), np.float32); pad = np.ones((B, M), bool); P = np.zeros((B, M + 1), np.float32)
    for b, (i, it, tm, s, E, K, *_) in enumerate(eps):
        n = min(M, len(it)); c = D.Fg[it[:n]]; q = D.Fq[i]
        X[b, :n, :d] = q * c; X[b, :n, d:2 * d] = np.abs(q - c); X[b, :n, 2 * d:3 * d] = c; X[b, :n, 3 * d] = s[:n]
        X[b, :n, 3 * d + 1] = np.arange(n) / M; X[b, :n, 3 * d + 2] = np.arange(n) / len(s)
        if cal:
            t = tail(D.F0, s[:n], len(s)); X[b, :n, 3 * d + 2] = t
            C[b] = [t[0], t[1], (s[0] - s.mean()) / (s.std() + 1e-6) / 5, s.mean(), s.std(), np.median(s), s[0] - s[1]]
        else:
            C[b] = [np.log(len(s)), s.mean(), s.std(), np.median(s), np.quantile(s, 0.9), np.quantile(s, 0.99), s[0]]
        A[b, :n, :n] = c @ c.T; pad[b, :n] = False
        if E and tm[:n].any():
            P[b, :n] = tm[:n]
        else:
            P[b, M] = 1
    tt = lambda a: torch.from_numpy(a).to(dev)
    return tt(X), tt(C), tt(A), tt(pad), tt(P)


def loss_fn(lg, P, pad, bce):
    if bce:
        return nn.functional.binary_cross_entropy_with_logits(lg[:, :M][~pad], P[:, :M][~pad])
    return (torch.logsumexp(lg, 1) - torch.logsumexp(lg.masked_fill(P == 0, -1e4), 1)).mean()   # multi-positive CE


def dg_loss(lg, P, o):
    """11-06 Deep Gamblers (Liu et al., NeurIPS 2019) adapted to open-world matching: the no-match index M is the
    reservation (abstain) outcome with payoff o. Present episode (target in shortlist): accept a positive candidate or
    reserve -- maximise sum_{pos} p + p_null/o. Absent / target-outside-shortlist episode (P[:,M]=1): the only correct
    outcome is no-match -- maximise p_null. Disclosed adaptation: reservation here means 'no gallery match', not 'decline
    to predict'."""
    Z = torch.logsumexp(lg, 1); pres = P[:, M] == 0                 # present & target in shortlist
    pos = lg[:, :M].masked_fill(P[:, :M] == 0, -1e4)                # positive-candidate logits only
    num_pres = torch.logsumexp(torch.cat([pos, (lg[:, M] - math.log(o))[:, None]], 1), 1)
    num = torch.where(pres, num_pres, lg[:, M])
    return -(num - Z).mean()


def train(Dtr, kind, seed, steps, trk, bs=192):
    torch.manual_seed(seed); g = np.random.default_rng(seed); qs = np.flatnonzero(Dtr.ok)
    net = Net(Dtr.Fq.shape[1], kind).to(dev); opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps)
    for it in range(steps):
        eps = []
        while len(eps) < bs:
            i = g.choice(qs); E = int(g.random() < 0.5); K = int(g.choice(trk)); its, tm = Dtr.sample(i, E, K, g)
            if len(its) < 2 or (E and not tm.any()):
                continue
            s = Dtr.Fg[its] @ Dtr.Fq[i]; o = np.argsort(-s); eps.append((i, its[o], tm[o], s[o], E, K))
        X, C, A, pad, P = batch(Dtr, eps, net.cal); loss = loss_fn(net(X, C, A, pad), P, pad, kind == "pair_bce")
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 500 == 0:
            print(f"  {kind} s{seed} step {it} loss {loss.item():.4f}", flush=True)
    return net.eval()


def ap_r1(rank_tm):
    hits = np.flatnonzero(rank_tm)
    if not len(hits):
        return 0.0, 0.0
    return float(np.mean((np.arange(len(hits)) + 1) / (hits + 1))), float(rank_tm[0])


@torch.no_grad()
def run_model(net, D, eps):
    conf, corr, ap, r1 = [], [], [], []
    for b in range(0, len(eps), 256):
        ch = eps[b:b + 256]; X, C, A, pad, _ = batch(D, ch, net.cal); lg = net(X, C, A, pad)
        p = (torch.sigmoid(lg[:, :M]) if net.kind == "pair_bce" else torch.softmax(lg, 1)[:, :M]).cpu().numpy()
        for k, (i, it, tm, s, E, K, *ng) in enumerate(ch):
            n = min(M, len(it)); pk = p[k, :n]; j = int(pk.argmax()); conf.append(float(pk[j])); corr.append(float(E and tm[j]))
            if E:
                order = np.r_[np.argsort(-pk), np.arange(n, len(it))]; a, r = ap_r1(tm[order]); a *= recall(tm, ng)
            else:
                a, r = np.nan, np.nan
            ap.append(a); r1.append(r)
    return np.array(conf), np.array(corr), np.array(ap), np.array(r1)


def recall(tm, ng):                                                # fraction of gallery targets that are candidates at all
    return float(tm.sum()) / ng[0] if ng and ng[0] else 1.0


def set_stats(s):
    mu, sd = s.mean(), s.std() + 1e-6
    return [s[0], s[1], s[0] - s[1], mu, sd, np.median(s), (s[0] - mu) / sd, skew(s), kurtosis(s), s[:5].mean(), np.log(len(s))]


def wscore(s, n=10):
    t = s[1:1 + n]
    if len(t) < 3:
        return 0.5
    lo = t.min() - 1e-3; x = t - lo
    try:
        c, _, sc = weibull_min.fit(x, floc=0); return float(weibull_min.cdf(s[0] - lo, c, 0, sc))
    except Exception:
        return 0.5


def qnull(s):
    m = max(5, len(s) // 50); t = s[m:] if len(s) > m + 5 else s[1:]
    z = (s[0] - t.mean()) / (t.std() + 1e-6); lf = len(s) * norm.logcdf(z)
    return float(-np.log10(max(-np.expm1(lf), 1e-300)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--enc", required=True)
    ap.add_argument("--variants", default="base"); ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--ntrain_eval", type=int, default=1200); ap.add_argument("--trk", default=""); ap.add_argument("--tag", default="")
    ap.add_argument("--ntrain_ids", type=int, default=0)
    ap.add_argument("--qpp", type=int, default=1, help="11-04 fix: query boxes per training person, as in sm_v6 (meta must use the same train-eval episodes)")
    a = ap.parse_args(); trk = tuple(int(k) for k in a.trk.split(",")) if a.trk else TRK[a.data]
    load = (lambda enc, sp: market(enc, sp, a.data)) if a.data in ("market", "rpif", "msmt") else (lambda enc, sp: scene(enc, sp, a.data, a.qpp))
    Dtr, Dte = load(a.enc, "train"), load(a.enc, "test")
    if a.ntrain_ids:
        Dtr = restrict_ids(Dtr, a.ntrain_ids)
    od = os.path.join(O, "v5"); os.makedirs(od, exist_ok=True)
    ev_te = episodes(Dte, TEK.get(a.data, getattr(Dte, "Gs", None)), None, 2)
    print(a.data, a.enc, "train q", int(Dtr.ok.sum()), "test episodes", len(ev_te), flush=True)
    for v in a.variants.split(","):
        out = os.path.join(od, f"{a.data}_{a.enc}_{v}{a.tag}.npz")
        if v == "base":                                             # cosine-based rules + SDSE / p_null fitted on train episodes
            ev_tr = episodes(Dtr, trk, a.ntrain_eval, 1)            # use the --trk override (if any), same grid as the other variants below
            lab = lambda ev: np.array([float(E and tm[0]) for i, it, tm, s, E, K, *_ in ev])
            Vtr, Vte = lab(ev_tr), lab(ev_te)
            Xtr = np.nan_to_num(np.array([set_stats(e[3]) for e in ev_tr])); Xte = np.nan_to_num(np.array([set_stats(e[3]) for e in ev_te]))
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; w = np.zeros(Xtr.shape[1] + 1); Z = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
            for _ in range(50):
                q = 1 / (1 + np.exp(-Z @ w)); w -= np.linalg.solve((Z * (q * (1 - q))[:, None]).T @ Z + 1e-2 * np.eye(len(w)), Z.T @ (q - Vtr) + 1e-2 * w)
            F0 = np.sort(np.concatenate([e[3][~e[2]][:200] for e in ev_tr]))
            pn = lambda s: -np.log10(np.clip(1 - np.clip(np.searchsorted(F0, s[0], side="right") / len(F0), 1e-6, 1 - 1e-9) ** len(s), 1e-12, 1))
            res = {}
            for nm, ftr, fte in (("raw", [e[3][0] for e in ev_tr], [e[3][0] for e in ev_te]),
                                 ("margin", [e[3][0] - e[3][1] for e in ev_tr], [e[3][0] - e[3][1] for e in ev_te]),
                                 ("sdse", Z @ w, np.c_[(Xte - mu) / sd, np.ones(len(Xte))] @ w),
                                 ("pnull", [pn(e[3]) for e in ev_tr], [pn(e[3]) for e in ev_te])):
                res[nm + "_tr"] = np.asarray(ftr, float); res[nm + "_te"] = np.asarray(fte, float)
            tk = lambda ev: np.array([np.pad(np.asarray(e[3][:8], float), (0, max(0, 8 - len(e[3]))), constant_values=np.nan) for e in ev])
            res["topk_tr"] = tk(ev_tr); res["topk_te"] = tk(ev_te)     # top-8 scores/episode -> LocalScore (s0+s_k) post-hoc for any k
            res["stats_tr"] = Xtr; res["stats_te"] = Xte               # the 11 score-list stats, for faithful stat-based baselines (SDSE etc.)
            apr = [(ap_r1(e[2])[0] * recall(e[2], e[6:]), ap_r1(e[2])[1]) if e[4] else (np.nan, np.nan) for e in ev_te]
            np.savez(os.path.join(od, f"{a.data}_{a.enc}_meta{a.tag if (a.ntrain_ids or a.qpp != 1) else ''}.npz"), E=np.array([e[4] for e in ev_te]), K=np.array([e[5] for e in ev_te]),
                     Etr=np.array([e[4] for e in ev_tr]), Vtr=Vtr, Vte=Vte, cos_ap=np.array([x[0] for x in apr]), cos_r1=np.array([x[1] for x in apr]), **res)
            print("base saved", flush=True); continue
        if v == "evt":
            ev_tr = episodes(Dtr, trk, a.ntrain_eval, 1); res = {}
            for nm, f in (("wscore", wscore), ("qnull", qnull)):
                res[nm + "_tr"] = np.array([f(e[3]) for e in ev_tr]); res[nm + "_te"] = np.array([f(e[3]) for e in ev_te])
            np.savez(out + ".tmp.npz", **res); os.replace(out + ".tmp.npz", out); print("saved evt", flush=True); continue
        ev_tr = episodes(Dtr, trk, a.ntrain_eval, 1); r = {}
        neg = np.concatenate([e[3][~e[2]] for e in ev_tr])               # impostor-score CDF for the _cal inputs (train only)
        Dtr.F0 = Dte.F0 = np.sort(np.random.default_rng(0).choice(neg, min(len(neg), 200000), replace=False))
        for sd_ in range(a.seeds):
            net = train(Dtr, v, sd_, a.steps, trk)
            for nm, D, ev in (("tr", Dtr, ev_tr), ("te", Dte, ev_te)):
                c, k, apv, r1 = run_model(net, D, ev); r[f"conf_{nm}_s{sd_}"] = c; r[f"corr_{nm}_s{sd_}"] = k
                if nm == "te":
                    r[f"ap_s{sd_}"] = apv; r[f"r1_s{sd_}"] = r1
        np.savez(out + ".tmp.npz", **r); os.replace(out + ".tmp.npz", out); print("saved", v, flush=True)


if __name__ == "__main__":
    main()
