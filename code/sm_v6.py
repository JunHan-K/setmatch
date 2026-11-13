"""11-02 SetWiseMatch fast path (same models, data and episodes as sm_v5.py; outputs are read by sm_v5_eval.py).
sm_v5.py rebuilt 192 episodes on the CPU at every training step and the whole test set for every variant, so the GPU
waited on numpy. Here every episode set is built once per (data, encoder, training-K grid) and cached in compact form
(top-M indices, scores, both context vectors, hit positions beyond M); model inputs are gathered on the GPU from the
feature tables. Training samples from a fixed pool of --pool episodes (sm_v5: fresh episodes every step).
One process runs every requested variant and seed.
<kind>_qcal (v7 candidate): as _cal, but the null of the tail coordinate is query-local -- a Gaussian fitted to the
list below its top max(5, N/50) ranks -- instead of the global training F0.
<kind>_hyb (v7): _cal inputs plus cardinality inside the training support only: log N clamped to the largest
training N, and the raw top score. In-range galleries keep the information of ctx; beyond the training range the
cardinality input saturates and the tail coordinate carries the extrapolation.
--target T --nulls src,unlab,oracle,intrain (C3 T3): train on --data, test on dataset T; the competition coordinate of
the test episodes uses the chosen null F0 (src = source training impostors; unlab = co-occurring boxes in T's test
images, no identity labels; oracle = T's test query impostors; intrain = T's labelled train split). Writes
<T>_<enc>_<variant>_x<data>_<null>.npz, evaluated with the source training episodes' frozen threshold.
<kind>_eff (v8): effective competition. Competitors are not independent (several images of one person, similar
people), so the null of the top score is F0(s)^{N_eff} with N_eff = a N^gamma fitted by maximum likelihood on the
target-absent training episodes (-N_eff log F0(s_1) ~ Exp(1)). Inputs: the log-odds of u = F0(s)^{N_eff} per
candidate and for the two best scores, plus the N-free shape statistics; no log N. Writes results/setmatch/v5/<data>_<enc>_<variant><tag>.npz."""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

import sm_v5 as V
from scipy.stats import norm

from sm_v5 import M, Net, TEK, TRK, dg_loss, episodes, loss_fn, scene, market, tail


def qtail(sc, n):                                                  # query-local null: Gaussian on ranks below the top m
    m = max(5, len(sc) // 50); r = sc[m:] if len(sc) > m + 5 else sc[1:]
    z = (sc[:n] - r.mean()) / (r.std() + 1e-6)
    return np.clip(-np.log10(np.maximum(-np.expm1(len(sc) * norm.logcdf(z)), 1e-300)) / 6, 0, 2).astype(np.float32)

CACHE = os.path.join(V.O, "cache")


def pack(eps, F0):
    """episodes -> dict of arrays; context for raw (v5 ctx) and calibrated (_cal) inputs"""
    B = len(eps); idx = np.full((B, M), -1, np.int32); s = np.zeros((B, M), np.float32); t = np.zeros((B, M), np.float32)
    tq = np.zeros((B, M), np.float32); Cq = np.zeros((B, 7), np.float32)
    N = np.zeros(B, np.float32); Cr = np.zeros((B, 7), np.float32); Cc = np.zeros((B, 7), np.float32); qi = np.zeros(B, np.int32)
    tm = np.zeros((B, M), bool); P = np.zeros((B, M + 1), np.float32); E = np.zeros(B, np.int8); K = np.zeros(B, np.int32)
    rest, roff, ngt = [], [0], np.zeros(B, np.float32)
    for b, (i, it, m, sc, e, k, *ng) in enumerate(eps):
        n = min(M, len(it)); qi[b], E[b], K[b], N[b] = i, e, k, len(sc); idx[b, :n] = it[:n]; s[b, :n] = sc[:n]; tm[b, :n] = m[:n]
        tt = tail(F0, sc[:n], len(sc)); t[b, :n] = tt
        Cr[b] = [np.log(len(sc)), sc.mean(), sc.std(), np.median(sc), np.quantile(sc, 0.9), np.quantile(sc, 0.99), sc[0]]
        Cc[b] = [tt[0], tt[1], (sc[0] - sc.mean()) / (sc.std() + 1e-6) / 5, sc.mean(), sc.std(), np.median(sc), sc[0] - sc[1]]
        tl = qtail(sc, n); tq[b, :n] = tl; Cq[b] = [tl[0], tl[1], *Cc[b, 2:]]
        if e and m[:n].any():
            P[b, :n] = m[:n]
        else:
            P[b, M] = 1
        h = np.flatnonzero(m[n:]) + n; rest.append(h); roff.append(roff[-1] + len(h))
        ngt[b] = ng[0] if ng and ng[0] else m.sum()
    return dict(idx=idx, s=s, t=t, tq=tq, N=N, Cr=Cr, Cc=Cc, Cq=Cq, qi=qi, tm=tm, P=P, E=E, K=K, ngt=ngt,
                rest=np.concatenate(rest).astype(np.int32) if rest else np.zeros(0, np.int32), roff=np.array(roff, np.int64))


def pool_eps(D, n, trk, seed):
    """the sm_v5 training sampler, run once: n episodes with E ~ Bernoulli(.5), K uniform over trk"""
    g = np.random.default_rng(seed); qs = np.flatnonzero(D.ok); out = []
    while len(out) < n:
        i = g.choice(qs); E = int(g.random() < 0.5); K = int(g.choice(trk)); its, tm = D.sample(i, E, K, g)
        if len(its) < 2 or (E and not tm.any()):
            continue
        s = D.Fg[its] @ D.Fq[i]; o = np.argsort(-s); out.append((i, its[o], tm[o], s[o], E, K))
    return out


def gs_stats(name, z, make_eps):
    """11-04 GSC: GS statistics (sm_v5.set_stats) of every episode's full score list, stored beside the cache in
    <name>_gsc.npz. The episodes are regenerated with the cache's seeds and must match it exactly (query, presence, N)."""
    p = os.path.join(CACHE, name + "_gsc.npz")
    if os.path.exists(p):
        return np.load(p)["Cg"]
    eps = make_eps(); assert len(eps) == len(z["qi"]), (name, len(eps), len(z["qi"]))
    assert all(e[0] == q and int(e[4]) == E_ and len(e[3]) == n for e, q, E_, n in zip(eps, z["qi"], z["E"], z["N"])), name + ": regenerated episodes differ from the cache"
    Cg = np.nan_to_num(np.array([V.set_stats(e[3]) for e in eps], np.float32)); np.savez(p + ".tmp.npz", Cg=Cg); os.replace(p + ".tmp.npz", p); return Cg


def cached(name, build):
    p = os.path.join(CACHE, name + ".npz")
    if os.path.exists(p):
        z = dict(np.load(p))
        if "tq" in z:                                              # caches from before _qcal are rebuilt
            return z
    t0 = time.time(); z = build(); os.makedirs(CACHE, exist_ok=True); tmp = f"{p}.{os.getpid()}.tmp.npz"; np.savez(tmp, **z); os.replace(tmp, p)
    print(f"  built {name} in {time.time() - t0:.0f}s", flush=True); return z


class G:
    """packed episodes + feature tables on the GPU"""

    def __init__(self, z, D, share=None):
        self.z = z; self.n = len(z["qi"]); T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(V.dev)
        self.Fg, self.Fq = (share.Fg, share.Fq) if share is not None else (T(D.Fg.astype(np.float32)), T(D.Fq.astype(np.float32)))
        for k in ("idx", "s", "t", "tq", "N", "Cr", "Cc", "Cq", "qi", "P"):
            setattr(self, k, T(z[k]))
        self.idx = self.idx.long(); self.qi = self.qi.long(); self.F0e = None   # per-set null for _eff (cross-dataset)
        self.Cg = T(z["Cg"].astype(np.float32)) if "Cg" in z else None   # 11-04 GSC: GS statistics of the full score list
        self.ga = T(z["ga"].astype(np.float32)) if "ga" in z else None   # 11-04 GSR: logit of the GS anchor (logistic regression on the pool)

    lmax = None                                                    # log of the largest training N (set from the pool)
    lncut = None; train_mode = False; qhead = None; qt = None      # M1 cardinality dropout / M2 query-conditional null
    F0 = a = gam = None                                            # effective-competition null (set in main)
    gmu = gsd = None                                               # GSC: standardisation of [Cr, Cg] (training pool)

    def batch(self, sel, cal):                                     # cal: False | True (global F0) | "q" (query-local) | "h" (hybrid) | "g" (GSC)
        gsc = cal == "g"; cal = False if gsc else cal
        idx = self.idx[sel]; pad = idx < 0; c = self.Fg[idx.clamp(min=0)] * (~pad)[..., None]; q = self.Fq[self.qi[sel]][:, None]
        n = (~pad).sum(1, keepdim=True); ar = torch.arange(M, device=V.dev)[None].float()
        third = (self.tq if cal == "q" else self.t)[sel] if cal else ar / self.N[sel][:, None]
        X = torch.cat([q * c, (q - c).abs() * (~pad)[..., None], c, self.s[sel][..., None], (ar / M).expand(len(sel), -1)[..., None],
                       third[..., None]], -1) * (~pad)[..., None]
        A = c @ c.transpose(1, 2); C = (self.Cq if cal == "q" else self.Cc)[sel] if cal else self.Cr[sel]
        if cal == "e":
            lo = self.eff_logodds(self.s[sel], self.N[sel]) * (~pad); third = lo
            C = torch.cat([lo[:, :2], self.Cc[sel][:, 2:]], 1)
            X = torch.cat([X[..., :-1], third[..., None]], -1)
        if cal == "h":
            C = torch.cat([C, (self.N[sel].log().clamp(max=G.lmax) / 10)[:, None], self.s[sel][:, :1]], 1)
        if cal == "d":                                             # M1: nominal tail + raw size context, raw masked out of support
            lN = self.N[sel].log()
            mask = ((lN > G.lncut) | (torch.rand(len(sel), device=V.dev) < 0.2)) if G.train_mode else (lN > G.lmax + 1e-6)
            C = torch.cat([self.Cc[sel], self.Cr[sel] * (~mask)[:, None].float(), mask[:, None].float()], 1)
        if cal == "qn":                                            # M2: tail from predicted query-conditional impostor quantiles
            Q = self.qquant(sel); t = self.qtail(self.s[sel], self.N[sel], Q) * (~pad); X = torch.cat([X[..., :-1], t[..., None]], -1)
            C = torch.cat([t[:, :2], self.Cc[sel][:, 2:]], 1); self._Q = Q
        if gsc:                                                    # GSC: raw context + GS statistics, standardised
            C = (torch.cat([self.Cr[sel], self.Cg[sel]], 1) - G.gmu) / G.gsd
        return X, C, A, pad, self.P[sel]

    def qquant(self, sel):
        h = G.qhead(torch.cat([self.Fq[self.qi[sel]], self.Cr[sel][:, 1:6]], 1))
        q90 = h[:, 0]; q99 = q90 + nn.functional.softplus(h[:, 1]) + 1e-3; q999 = q99 + nn.functional.softplus(h[:, 2]) + 1e-3
        return torch.stack([q90, q99, q999], 1)

    @staticmethod
    def qtail(s, N, Q):
        l1, l2, l3 = np.log(0.1), np.log(0.01), np.log(0.001); q90, q99, q999 = Q[:, :1], Q[:, 1:2], Q[:, 2:3]
        lp = torch.where(s <= q99, l1 + (l2 - l1) / (q99 - q90) * (s - q90), l2 + (l3 - l2) / (q999 - q99) * (s - q99)).clamp(max=-1e-6)
        lF = torch.log1p(-lp.exp()); x = (N[:, None] * lF).clamp(min=-80)
        return (-torch.log10((-torch.expm1(x)).clamp(min=1e-30)) / 6).clamp(0, 2)


    def eff_logodds(self, s, N):                                   # log u - log(1 - u), u = F0(s)^(a N^gamma); scaled, clipped
        F0 = self.F0e if self.F0e is not None else G.F0
        F = (torch.searchsorted(F0, s.contiguous(), right=True).float() / len(F0)).clamp(1e-9, 1 - 1e-9)
        nx = G.a * N[:, None] ** G.gam * -F.log(); lu = -nx; l1u = torch.log(-torch.expm1(-nx)).clamp(min=-60)
        return ((lu - l1u) / 5).clamp(-4, 4)


@torch.no_grad()
def query_targets(D, zpl):
    """true 0.9 / 0.99 / 0.999 quantiles of every training query's impostor scores over the training gallery items"""
    items = np.unique(zpl["idx"][zpl["idx"] >= 0]); Fg = torch.from_numpy(D.Fg[items].astype(np.float32)).to(V.dev); gp = torch.from_numpy(D.gpid[items]).to(V.dev)
    out = []
    for b in range(0, len(D.Fq), 128):
        q = torch.from_numpy(D.Fq[b:b + 128].astype(np.float32)).to(V.dev); S = q @ Fg.T
        S = S.masked_fill(gp[None] == torch.from_numpy(D.qpid[b:b + 128]).to(V.dev)[:, None], float("nan"))
        out.append(torch.nanquantile(S, torch.tensor([0.9, 0.99, 0.999], device=V.dev), dim=1).T)
    Q = torch.cat(out); print("query-null targets", tuple(Q.shape), "mean", Q.mean(0).tolist(), flush=True); return Q


def fit_neff(z):
    """N_eff = a N^gamma by maximum likelihood on target-absent train episodes"""
    e0 = z["E"] == 0; F = np.clip(np.searchsorted(z["F0"], z["s"][e0, 0], side="right") / len(z["F0"]), 1e-9, 1 - 1e-9)
    x, N = -np.log(F), z["N"][e0]; best = None
    for g in np.linspace(0.2, 1.3, 111):
        a = len(x) / np.sum(N ** g * x); n = a * N ** g; ll = np.sum(np.log(n) - n * x)
        if best is None or ll > best[0]:
            best = (ll, g, a)
    return best[2], best[1]


def fwd(net, Gx, sel, X, C, A, pad):
    """11-04 GSR: the no-match logit is offset by -lam * (GS anchor logit); the layer learns a correction to GS"""
    lg = net(X, C, A, pad)
    if getattr(net, "gsr", False):
        lg = torch.cat([lg[:, :M], lg[:, M:] - net.lam * Gx.ga[sel][:, None]], 1)
    return lg


def gs_fit(Z, y, lam=1e-2):
    """GS logistic regression (Newton), as in cross_gs"""
    Z = np.c_[Z, np.ones(len(Z))]; w = np.zeros(Z.shape[1])
    for _ in range(50):
        q = 1 / (1 + np.exp(-Z @ w)); w -= np.linalg.solve((Z * (q * (1 - q))[:, None]).T @ Z + lam * np.eye(len(w)), Z.T @ (q - y) + lam * w)
    return w


def gsmlp(zpl, zs, qpid, seed, steps=3000):
    """11-04 NG: nonlinear GS. MLP on the eleven GS statistics (log N included) of the SetWiseMatch training pool, label =
    cosine top-1 is a true match; width x weight decay chosen by BCE on pool episodes of 20% held-out training
    identities, then refitted on the whole pool. Returns the accept probability for each packed set in zs."""
    mu, sd = zpl["Cg"].mean(0), zpl["Cg"].std(0) + 1e-6; T = lambda a: torch.from_numpy(((a - mu) / sd).astype(np.float32)).to(V.dev)
    y = torch.from_numpy(((zpl["E"] == 1) & zpl["tm"][:, 0]).astype(np.float32)).to(V.dev); X = T(zpl["Cg"])
    ids = np.unique(qpid); ho = np.isin(qpid, np.random.default_rng(seed).choice(ids, max(1, len(ids) // 5), replace=False)); ho_t = torch.from_numpy(ho).to(V.dev)

    def fit(h, wd, m):
        torch.manual_seed(seed); g = torch.Generator(device="cpu").manual_seed(seed); f = nn.Sequential(nn.Linear(X.shape[1], h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1)).to(V.dev)
        opt = torch.optim.AdamW(f.parameters(), 1e-3, weight_decay=wd); ix = torch.nonzero(m)[:, 0]
        for _ in range(steps):
            b = ix[torch.randint(len(ix), (1024,), generator=g).to(V.dev)]; loss = nn.functional.binary_cross_entropy_with_logits(f(X[b])[:, 0], y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        return f.eval()
    best = None
    for h in (32, 64, 128):
        for wd in (1e-4, 1e-2):
            f = fit(h, wd, ~ho_t)
            with torch.no_grad():
                v = nn.functional.binary_cross_entropy_with_logits(f(X[ho_t])[:, 0], y[ho_t]).item()
            best = min(best or (9e9,), (v, h, wd))
    f = fit(best[1], best[2], torch.ones_like(ho_t)); print(f"  gsmlp s{seed} chose h{best[1]} wd{best[2]} val {best[0]:.4f}", flush=True)
    with torch.no_grad():
        return [torch.sigmoid(f(T(z["Cg"]))[:, 0]).double().cpu().numpy() for z in zs]


def sch_feats(z, j, p):
    """11-05 SCH inputs per set: selector posterior of its choice, its margin and entropy, the chosen candidate's cosine,
    its gap to the cosine top-1, its rank, the cosine top-1/top-2 gap, the eleven GS statistics and the raw context"""
    n = np.arange(len(j)); q = np.sort(np.clip(p, 0, None), 1)[:, ::-1]; pn = np.clip(p, 1e-9, None)
    ent = -np.where(p >= 0, pn * np.log(pn), 0.0).sum(1) / np.log(np.maximum((p >= 0).sum(1), 2)); sc = z["s"]
    return np.nan_to_num(np.c_[q[:, 0], q[:, 0] - q[:, 1], ent, sc[n, j], sc[:, 0] - sc[n, j], j / M, sc[:, 0] - sc[:, 1], z["Cg"], z["Cr"]]).astype(np.float32)


def corr_head(Xp, yp, Xs, qpid, seed, steps=3000):
    """11-05 SCH: MLP on sch_feats, trained on the pool sets to predict that the frozen selector's choice is a true match;
    width x weight decay chosen by BCE on pool sets of 20% held-out training identities, then refitted on the whole pool"""
    mu, sd = Xp.mean(0), Xp.std(0) + 1e-6; T = lambda a: torch.from_numpy(((a - mu) / sd).astype(np.float32)).to(V.dev)
    X, y = T(Xp), torch.from_numpy(yp.astype(np.float32)).to(V.dev)
    ids = np.unique(qpid); ho = torch.from_numpy(np.isin(qpid, np.random.default_rng(seed).choice(ids, max(1, len(ids) // 5), replace=False))).to(V.dev)

    def fit(h, wd, m):
        torch.manual_seed(seed); g = torch.Generator(device="cpu").manual_seed(seed); f = nn.Sequential(nn.Linear(X.shape[1], h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1)).to(V.dev)
        opt = torch.optim.AdamW(f.parameters(), 1e-3, weight_decay=wd); ix = torch.nonzero(m)[:, 0]
        for _ in range(steps):
            b = ix[torch.randint(len(ix), (1024,), generator=g).to(V.dev)]; loss = nn.functional.binary_cross_entropy_with_logits(f(X[b])[:, 0], y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        return f.eval()
    best = None
    for h in (32, 64, 128):
        for wd in (1e-4, 1e-2):
            f = fit(h, wd, ~ho)
            with torch.no_grad():
                v = nn.functional.binary_cross_entropy_with_logits(f(X[ho])[:, 0], y[ho]).item()
            best = min(best or (9e9,), (v, h, wd))
    f = fit(best[1], best[2], torch.ones_like(ho)); print(f"  selch s{seed} chose h{best[1]} wd{best[2]} val {best[0]:.4f}", flush=True)
    with torch.no_grad():
        return [torch.sigmoid(f(T(x))[:, 0]).double().cpu().numpy() for x in Xs]


def train(Gp, kind, seed, steps, bs=192, sub=None):
    """sub (11-05 SCH cross-fitting): train only on these pool rows"""
    torch.manual_seed(seed); g = torch.Generator(device="cpu").manual_seed(seed)
    net = Net(Gp.Fq.shape[1], kind.replace("_qcal", "_cal").replace("_hyb", "_cal").replace("_eff", "_cal").replace("_sel", "").replace("_gsc", "")).to(V.dev); net.kind = kind
    if kind.endswith("_gsr"):
        net = Net(Gp.Fq.shape[1], kind.replace("_gsr", "")).to(V.dev); net.kind = kind; net.gsr = True; net.lam = nn.Parameter(torch.ones((), device=V.dev))
    net.selonly = kind.endswith("_sel")                            # 11-04 DS: candidate selector only (no-match logit blocked; present, in-shortlist episodes)
    if kind.endswith("_gsc"):
        net.cal = "g"; net.ctx = nn.Linear(18, net.ctx.out_features).to(V.dev)
    if kind.endswith("_eff"):
        net.cal = "e"
    if kind.endswith("_qcal"):
        net.cal = "q"
    if kind.endswith("_hyb"):
        net.cal = "h"; net.ctx = nn.Linear(9, net.ctx.out_features).to(V.dev)
    if kind.endswith("_dual"):
        net = Net(Gp.Fq.shape[1], kind.replace("_dual", "_cal")).to(V.dev); net.kind = kind; net.cal = "d"; net.ctx = nn.Linear(15, net.ctx.out_features).to(V.dev)
    if kind.endswith("_qn"):
        net = Net(Gp.Fq.shape[1], kind.replace("_qn", "_cal")).to(V.dev); net.kind = kind; net.cal = "qn"
        net.qhead = nn.Sequential(nn.Linear(Gp.Fq.shape[1] + 5, 128), nn.GELU(), nn.Linear(128, 3)).to(V.dev); G.qhead = net.qhead
        with torch.no_grad():
            net.qhead[2].bias.copy_(torch.tensor([0.3, -2.0, -2.0]))
    if kind.endswith("_dg"):                                        # 11-06 Deep Gamblers: attn architecture, reservation loss on the no-match index
        net = Net(Gp.Fq.shape[1], "attn").to(V.dev); net.kind = kind
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps)
    G.train_mode = True
    for it in range(steps):
        sel = torch.randint(Gp.n, (bs,), generator=g).to(V.dev) if sub is None else sub[torch.randint(len(sub), (bs,), generator=g).to(V.dev)]
        X, C, A, pad, P = Gp.batch(sel, net.cal)
        if getattr(net, "selonly", False):
            keep = P[:, :M].sum(1) > 0; lg_ = net(X, C, A, pad)[keep]; lg_[:, M] = -1e4
            loss = loss_fn(lg_, P[keep], pad[keep], False)
        elif kind.endswith("_dg"):
            loss = dg_loss(fwd(net, Gp, sel, X, C, A, pad), P, float(os.environ.get("DG_O", "2")))
        else:
            loss = loss_fn(fwd(net, Gp, sel, X, C, A, pad), P, pad, kind == "pair_bce")
        if net.cal == "qn" and G.qt is not None:                   # supervise the predicted quantiles with the true ones
            loss = loss + 100 * ((Gp._Q - G.qt[Gp.qi[sel]]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 500 == 0:
            print(f"  {kind} s{seed} step {it} loss {loss.item():.4f}", flush=True)
    G.train_mode = False; return net.eval()


@torch.no_grad()
def run(net, Ge):
    z = Ge.z; conf, choice, ps, p0s = [], [], [], []; G.train_mode = False
    if getattr(net, "qhead", None) is not None:
        G.qhead = net.qhead
    for b in range(0, Ge.n, 1024):
        sel = torch.arange(b, min(b + 1024, Ge.n), device=V.dev); X, C, A, pad, _ = Ge.batch(sel, net.cal); lg = fwd(net, Ge, sel, X, C, A, pad)
        if getattr(net, "selonly", False):
            lg[:, M] = -1e4
        p = torch.sigmoid(lg[:, :M]) if net.kind == "pair_bce" else torch.softmax(lg, 1)[:, :M]
        p = p.masked_fill(pad, -1.0); c, j = p.max(1); conf.append(c.cpu()); choice.append(j.cpu()); ps.append(p.cpu())
        p0s.append(torch.softmax(lg, 1)[:, M].cpu() if net.kind != "pair_bce" else torch.zeros(len(sel)))
    conf, j, p = torch.cat(conf).numpy(), torch.cat(choice).numpy(), torch.cat(ps).numpy()
    Ge.last = (j, p)                                               # 11-05 SCH: choice and candidate posteriors of the last run
    Ge.p0 = torch.cat(p0s).numpy().astype(np.float64)                # 11-05 DRA: no-match probability of the last run
    tm, E = z["tm"], z["E"]; corr = (E == 1) & tm[np.arange(len(j)), j]; ap = np.full(len(j), np.nan); r1 = np.full(len(j), np.nan)
    for b in np.flatnonzero(E == 1):
        n = int((z["idx"][b] >= 0).sum()); o = np.argsort(-p[b, :n], kind="stable"); h = np.r_[np.flatnonzero(tm[b, o]), z["rest"][z["roff"][b]:z["roff"][b + 1]]]
        if len(h):
            ap[b] = np.mean((np.arange(len(h)) + 1) / (h + 1)) * (len(h) / z["ngt"][b] if z["ngt"][b] else 1.0); r1[b] = float(h[0] == 0)
        else:
            ap[b] = r1[b] = 0.0
    return conf.astype(np.float64), corr.astype(np.float64), ap, r1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--enc", required=True)
    ap.add_argument("--variants", required=True); ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--pool", type=int, default=150000); ap.add_argument("--ntrain_eval", type=int, default=1200)
    ap.add_argument("--trk", default=""); ap.add_argument("--tag", default=""); ap.add_argument("--save_null", action="store_true"); ap.add_argument("--qpp", type=int, default=1)
    ap.add_argument("--target", default=""); ap.add_argument("--nulls", default="src,unlab,oracle,intrain")
    ap.add_argument("--hardtrain", action="store_true", help="cross: half of the source pool shortlisted (m in {4, 16}), source only")
    ap.add_argument("--law", default="fit", help="multiplicity of the _eff coordinate: fit (a N^gamma) | nominal (N) | none (F0 only)")
    ap.add_argument("--ntrain_ids", type=int, default=0, help="IDS: keep only this many training identities as queries")
    a = ap.parse_args()
    trk = tuple(int(k) for k in a.trk.split(",")) if a.trk else TRK[a.data]; kt = "-".join(map(str, trk)); t0 = time.time()
    load = (lambda enc, sp: market(enc, sp, a.data)) if a.data in ("market", "rpif", "msmt") else (lambda enc, sp: scene(enc, sp, a.data, a.qpp))
    Dtr, Dte = load(a.enc, "train"), load(a.enc, "test"); pre = f"{a.data}_{a.enc}"; cpre = pre + (f"_q{a.qpp}" if a.qpp != 1 else "")
    if a.ntrain_ids:
        Dtr = V.restrict_ids(Dtr, a.ntrain_ids); cpre += f"_id{a.ntrain_ids}"
    ev_tr = None

    def build_tr():
        nonlocal ev_tr
        ev_tr = episodes(Dtr, trk, a.ntrain_eval, 1); neg = np.concatenate([e[3][~e[2]] for e in ev_tr])
        F0 = np.sort(np.random.default_rng(0).choice(neg, min(len(neg), 200000), replace=False)); z = pack(ev_tr, F0); z["F0"] = F0; return z
    ztr = cached(f"{cpre}_k{kt}_tr", build_tr); F0 = ztr["F0"]
    if a.target:
        return cross(a, Dtr, ztr, trk, kt, cpre, t0)
    zte = cached(f"{cpre}_k{kt}_te", lambda: pack(episodes(Dte, TEK.get(a.data, getattr(Dte, "Gs", None)), None, 2), F0))
    zpl = cached(f"{cpre}_k{kt}_pool{a.pool}", lambda: pack(pool_eps(Dtr, a.pool, trk, 7), F0))
    if any(v.endswith("_gsc") for v in a.variants.split(",")):
        for nm, z_, mk in (("tr", ztr, lambda: episodes(Dtr, trk, a.ntrain_eval, 1)), ("te", zte, lambda: episodes(Dte, TEK.get(a.data, getattr(Dte, "Gs", None)), None, 2)),
                           (f"pool{a.pool}", zpl, lambda: pool_eps(Dtr, a.pool, trk, 7))):
            z_["Cg"] = gs_stats(f"{cpre}_k{kt}_{nm}", z_, mk)
        cat = np.concatenate([zpl["Cr"], zpl["Cg"]], 1); G.gmu = torch.from_numpy(cat.mean(0).astype(np.float32)).to(V.dev)
        G.gsd = torch.from_numpy((cat.std(0) + 1e-6).astype(np.float32)).to(V.dev)
    if any(v.endswith("_gsr") or v in ("gsmlp", "selch") for v in a.variants.split(",")):
        for nm, z_, mk in (("tr", ztr, lambda: episodes(Dtr, trk, a.ntrain_eval, 1)), ("te", zte, lambda: episodes(Dte, TEK.get(a.data, getattr(Dte, "Gs", None)), None, 2)),
                           (f"pool{a.pool}", zpl, lambda: pool_eps(Dtr, a.pool, trk, 7))):
            if "Cg" not in z_:
                z_["Cg"] = gs_stats(f"{cpre}_k{kt}_{nm}", z_, mk)
        mu, sd = zpl["Cg"].mean(0), zpl["Cg"].std(0) + 1e-6; w = gs_fit((zpl["Cg"] - mu) / sd, ((zpl["E"] == 1) & zpl["tm"][:, 0]).astype(float))
        for z_ in (ztr, zte, zpl):
            z_["ga"] = np.c_[(z_["Cg"] - mu) / sd, np.ones(len(z_["Cg"]))] @ w
    print(pre, "train-eval", len(ztr["qi"]), "test", len(zte["qi"]), "pool", len(zpl["qi"]), f"data ready {time.time() - t0:.0f}s", flush=True)
    Gtr, Gte, Gpl = G(ztr, Dtr), G(zte, Dte), G(zpl, Dtr); G.lmax = float(np.log(zpl["N"].max()))
    G.lncut = float(np.log(np.quantile(zpl["N"], 0.7)))
    if any(v.endswith("_qn") for v in a.variants.split(",")):
        G.qt = query_targets(Dtr, zpl)
    G.a, G.gam = fit_neff(ztr) if a.law == "fit" else ((1.0, 1.0) if a.law == "nominal" else (1.0, 0.0))
    G.F0 = torch.from_numpy(ztr["F0"].astype(np.float32)).to(V.dev); print(f"law {a.law}: N_eff = {G.a:.3f} N^{G.gam:.2f}", flush=True); od = os.path.join(V.O, "v5")
    for v in a.variants.split(","):
        r = {}; t1 = time.time()
        if v == "selch":                                           # SCH: frozen selector (attn_sel) + correctness head of its choice
            for sd in range(a.seeds):
                # head inputs on the pool are out-of-fold: two selectors, each trained on the pool sets of half the training
                # identities, score the other half; the selector applied at test time is trained on the whole pool
                qp = Dtr.qpid[zpl["qi"]]; ids = np.unique(qp); half = np.isin(qp, np.random.default_rng(100 + sd).choice(ids, len(ids) // 2, replace=False))
                Xp, yp = None, None
                for m in (half, ~half):
                    nh = train(Gpl, "attn_sel", sd, a.steps, sub=torch.from_numpy(np.flatnonzero(m)).to(V.dev)); _, k, _, _ = run(nh, Gpl); j, pp = Gpl.last; f = sch_feats(zpl, j, pp)
                    if Xp is None:
                        Xp, yp = np.zeros_like(f), np.zeros_like(k)
                    Xp[~m], yp[~m] = f[~m], k[~m]
                net = train(Gpl, "attn_sel", sd, a.steps); fz = []
                for Gx, z_ in ((Gtr, ztr), (Gte, zte)):
                    cf, k, apv, r1 = run(net, Gx); j, pp = Gx.last; fz.append((sch_feats(z_, j, pp), k, apv, r1))
                fz = [(Xp, yp, None, None)] + fz
                ctr, cte = corr_head(fz[0][0], fz[0][1], (fz[1][0], fz[2][0]), qp, sd)
                r.update({f"conf_tr_s{sd}": ctr, f"corr_tr_s{sd}": fz[1][1], f"conf_te_s{sd}": cte, f"corr_te_s{sd}": fz[2][1], f"ap_s{sd}": fz[2][2], f"r1_s{sd}": fz[2][3]})
            r["Etr"] = ztr["E"]; out = os.path.join(od, f"{pre}_{v}{a.tag}.npz"); np.savez(out + ".tmp.npz", **r); os.replace(out + ".tmp.npz", out)
            print("saved", v, f"{time.time() - t1:.0f}s", flush=True); continue
        if v == "gsmlp":                                           # NG: no SetWiseMatch; accept = nonlinear GS, choice = cosine top-1
            for sd in range(a.seeds):
                ctr, cte = gsmlp(zpl, (ztr, zte), Dtr.qpid[zpl["qi"]], sd)
                r.update({f"conf_tr_s{sd}": ctr, f"corr_tr_s{sd}": ((ztr["E"] == 1) & ztr["tm"][:, 0]).astype(float), f"conf_te_s{sd}": cte,
                          f"corr_te_s{sd}": ((zte["E"] == 1) & zte["tm"][:, 0]).astype(float)})
            r["Etr"] = ztr["E"]; out = os.path.join(od, f"{pre}_{v}{a.tag}.npz"); np.savez(out + ".tmp.npz", **r); os.replace(out + ".tmp.npz", out)
            print("saved", v, f"{time.time() - t1:.0f}s", flush=True); continue
        for sd in range(a.seeds):
            net = train(Gpl, v, sd, a.steps)
            for nm, Gx in (("tr", Gtr), ("te", Gte)):
                c, k, apv, r1 = run(net, Gx); r[f"conf_{nm}_s{sd}"] = c; r[f"corr_{nm}_s{sd}"] = k
                if a.save_null:                                    # 11-05 DRA: keep p(no-match) for the decision-rule audit
                    r[f"null_{nm}_s{sd}"] = Gx.p0
                if nm == "te":
                    r[f"ap_s{sd}"] = apv; r[f"r1_s{sd}"] = r1
        r["Etr"] = ztr["E"]; out = os.path.join(od, f"{pre}_{v}{a.tag}.npz"); np.savez(out + ".tmp.npz", **r); os.replace(out + ".tmp.npz", out)
        print("saved", v, f"{time.time() - t1:.0f}s", flush=True)


def target_null(mode, tgt, enc, F0src):
    import sm_comp as SC
    rng = np.random.default_rng(0)
    if mode == "src":
        return F0src
    if mode == "intrain":
        return SC.labelled_null(tgt, enc, rng)
    z = np.load(os.path.join(V.O, f"{tgt}_index.npz")); f = SC.feats(tgt, enc); test = z["split"] == 1
    if "gal" in z.files:                                           # proposals: only detector boxes exist at deployment
        test = test & z["gal"].astype(bool)
    if "_t" in mode:                                               # time axis: only the first t test frames in (sequence, frame) order
        mode, t = mode.split("_t"); names = z["imgs"]; ti = np.unique(z["img"][test])
        key = [(int(str(names[k]).split("s")[1].split("_")[0]), int(str(names[k]).split("_")[1].split(".")[0])) for k in ti]
        first = ti[np.lexsort(np.array(key).T[::-1])][:int(t)]; test = test & np.isin(z["img"], first)
    if mode.startswith("xunlab"):                                  # C3x: random pairs of boxes from DIFFERENT target test images
        tb = np.flatnonzero(test); i, j = rng.choice(tb, 800000), rng.choice(tb, 800000); ok = z["img"][i] != z["img"][j]
        if mode.startswith("xunlab_n"):                            # sample efficiency: only n images
            keep = rng.choice(np.unique(z["img"][tb]), int(mode[8:]), replace=False); ok &= np.isin(z["img"][i], keep) & np.isin(z["img"][j], keep)
        return SC.sub(np.einsum("nd,nd->n", f[i[ok]], f[j[ok]]), 300000, rng)
    if mode.startswith("unlab"):                                   # unlab | unlab_iou<x> | unlab_n<images> | unlab_pos<percent>
        arg = mode[5:]; iou = float(arg[4:]) if arg.startswith("_iou") else 0.1; n = int(arg[2:]) if arg.startswith("_n") else None
        F = SC.cooccur(z, f, test, rng, iou, n)
        if arg.startswith("_pos"):                                 # contamination: same-identity pairs (labels used only to inject)
            tb = np.flatnonzero(test & (z["pid"] > 0)); m = int(len(F) * float(arg[4:]) / 100); i, j = rng.choice(tb, 40 * m), rng.choice(tb, 40 * m)
            ok = (z["pid"][i] == z["pid"][j]) & (i != j); i, j = i[ok][:m], j[ok][:m]; F = np.r_[F, np.einsum("nd,nd->n", f[i], f[j])]
        return F
    q = z[sorted(k for k in z.files if k.startswith("G") and k.endswith("_q"))[0]]; q = rng.choice(q, min(400, len(q)), replace=False)
    tb = np.flatnonzero(test); S = f[q] @ f[tb].T; return SC.sub(S[z["pid"][q][:, None] != z["pid"][tb][None]], 200000, rng)


def retail(z, F0):
    """same packed episodes, competition coordinate recomputed under another null"""
    z = dict(z); t = (tail(F0, z["s"], z["N"][:, None]) * (z["idx"] >= 0)).astype(np.float32); Cc = z["Cc"].copy()
    Cc[:, 0], Cc[:, 1] = t[:, 0], t[:, 1]; z["t"], z["Cc"], z["F0t"] = t, Cc, F0; return z


def cross_gs(a, Dtr, trk, eps):
    """GS (eleven set statistics + logistic regression) and the cosine top-1 rule fitted on SOURCE training episodes and
    applied unchanged to the target test episodes; saved in the layout of the learned rules (one 'seed')"""
    ev_tr = episodes(Dtr, trk, a.ntrain_eval, 1); lab = lambda ev: np.array([float(e[4] and e[2][0]) for e in ev])
    Vtr, Vte = lab(ev_tr), lab(eps); Xtr = np.nan_to_num(np.array([V.set_stats(e[3]) for e in ev_tr])); Xte = np.nan_to_num(np.array([V.set_stats(e[3]) for e in eps]))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; w = np.zeros(Xtr.shape[1] + 1); Z = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
    for _ in range(50):
        q = 1 / (1 + np.exp(-Z @ w)); w -= np.linalg.solve((Z * (q * (1 - q))[:, None]).T @ Z + 1e-2 * np.eye(len(w)), Z.T @ (q - Vtr) + 1e-2 * w)
    sig = lambda x: 1 / (1 + np.exp(-x)); apr = np.array([(V.ap_r1(e[2])[0] * V.recall(e[2], e[6:]), V.ap_r1(e[2])[1]) if e[4] else (np.nan, np.nan) for e in eps])
    Etr = np.array([e[4] for e in ev_tr]); od = os.path.join(V.O, "v5")
    for nm, ftr, fte in (("gs", sig(Z @ w), sig(np.c_[(Xte - mu) / sd, np.ones(len(Xte))] @ w)), ("cos", np.array([e[3][0] for e in ev_tr]), np.array([e[3][0] for e in eps]))):
        r = dict(conf_tr_s0=ftr.astype(np.float64), corr_tr_s0=Vtr, conf_te_s0=fte.astype(np.float64), corr_te_s0=Vte, ap_s0=apr[:, 0], r1_s0=apr[:, 1], Etr=Etr)
        out = os.path.join(od, f"{a.target}_{a.enc}_{nm}_x{a.data}_src.npz"); np.savez(out + ".tmp.npz", **r); os.replace(out + ".tmp.npz", out)
    print("saved gs, cos", flush=True)


def cross(a, Dtr, ztr, trk, kt, cpre, t0):
    Dte = market(a.enc, "test", a.target) if a.target in ("market", "rpif", "msmt") else scene(a.enc, "test", a.target); eps = episodes(Dte, TEK.get(a.target, getattr(Dte, "Gs", None)), None, 2); zte = {}
    base = pack(eps, ztr["F0"])
    for m in a.nulls.split(","):
        F0t = np.sort(target_null(m, a.target, a.enc, ztr["F0"]).astype(np.float32)); zte[m] = retail(base, F0t)
    zpl = cached(f"{cpre}_k{kt}_pool{a.pool}", lambda: pack(pool_eps(Dtr, a.pool, trk, 7), ztr["F0"]))
    if "gs" in a.variants.split(","):
        cross_gs(a, Dtr, trk, eps); a.variants = ",".join(v for v in a.variants.split(",") if v != "gs")
        if not a.variants:
            return
    if a.hardtrain:                                                # shift-aware training, built from SOURCE training images only
        import sm_hshift as SH
        zr = pack(SH.hard_pool(Dtr, a.data, 60000, trk, 11), ztr["F0"]); zh = SH.shortlist_inputs(zr, ztr["F0"], zr["K"])
        keys = ("idx", "s", "t", "tq", "N", "Cr", "Cc", "Cq", "qi", "P", "E", "tm"); half = len(zpl["qi"]) // 2
        zpl = {k: np.concatenate([zpl[k][:half], zh[k]]) for k in keys}
    print(a.data, "->", a.target, a.enc, "test", len(eps), f"data ready {time.time() - t0:.0f}s", flush=True)
    Gtr, Gpl = G(ztr, Dtr), G(zpl, Dtr); Gte = {}
    for m, z in zte.items():
        Gte[m] = G(z, Dte, share=next(iter(Gte.values()), None))
    G.lmax = float(np.log(zpl["N"].max())); od = os.path.join(V.O, "v5"); G.lncut = float(np.log(np.quantile(zpl["N"], 0.7)))
    if any(v.endswith("_qn") for v in a.variants.split(",")):
        G.qt = query_targets(Dtr, zpl)
    G.a, G.gam = fit_neff(ztr); G.F0 = torch.from_numpy(ztr["F0"].astype(np.float32)).to(V.dev)  # N_eff law transferred from the source
    for m, Gx in Gte.items():
        Gx.F0e = torch.from_numpy(zte[m]["F0t"]).to(V.dev)
    if any(v in ("gsmlp", "selch") for v in a.variants.split(",")):    # 11-05 EXT: strongest baselines in the transfer setting
        for nm, z_, mk in (("tr", ztr, lambda: episodes(Dtr, trk, a.ntrain_eval, 1)), (f"pool{a.pool}", zpl, lambda: pool_eps(Dtr, a.pool, trk, 7))):
            if "Cg" not in z_:
                z_["Cg"] = gs_stats(f"{cpre}_k{kt}_{nm}", z_, mk)
        Cgt = np.nan_to_num(np.array([V.set_stats(e[3]) for e in eps], np.float32))
        for z_ in zte.values():
            assert len(Cgt) == len(z_["qi"]); z_["Cg"] = Cgt
    ms = list(zte); qp = Dtr.qpid[zpl["qi"]]; lab = lambda z_: ((z_["E"] == 1) & z_["tm"][:, 0]).astype(float)
    for v in a.variants.split(","):
        r = {m: {} for m in zte}
        for sd in range(a.seeds):
            if v == "gsmlp":                                       # NG rule, source pool only; choice = cosine top-1
                o = gsmlp(zpl, (ztr, *[zte[m] for m in ms]), qp, sd)
                for i, m in enumerate(ms):
                    r[m].update({f"conf_tr_s{sd}": o[0], f"corr_tr_s{sd}": lab(ztr), f"conf_te_s{sd}": o[1 + i], f"corr_te_s{sd}": lab(zte[m])})
                continue
            if v == "selch":                                       # SCH rule exactly as in main(), source pool only
                ids = np.unique(qp); half = np.isin(qp, np.random.default_rng(100 + sd).choice(ids, len(ids) // 2, replace=False)); Xp = yp = None
                for mk_ in (half, ~half):
                    nh = train(Gpl, "attn_sel", sd, a.steps, sub=torch.from_numpy(np.flatnonzero(mk_)).to(V.dev)); _, k, _, _ = run(nh, Gpl); j, pp = Gpl.last; f = sch_feats(zpl, j, pp)
                    if Xp is None:
                        Xp, yp = np.zeros_like(f), np.zeros_like(k)
                    Xp[~mk_], yp[~mk_] = f[~mk_], k[~mk_]
                net = train(Gpl, "attn_sel", sd, a.steps); _, ktr, _, _ = run(net, Gtr); j, pp = Gtr.last; ftr = sch_feats(ztr, j, pp); fte = {}
                for m, Gx in Gte.items():
                    _, kk, apv, r1 = run(net, Gx); j, pp = Gx.last; fte[m] = (sch_feats(zte[m], j, pp), kk, apv, r1)
                o = corr_head(Xp, yp, (ftr, *[fte[m][0] for m in ms]), qp, sd)
                for i, m in enumerate(ms):
                    r[m].update({f"conf_tr_s{sd}": o[0], f"corr_tr_s{sd}": ktr, f"conf_te_s{sd}": o[1 + i], f"corr_te_s{sd}": fte[m][1], f"ap_s{sd}": fte[m][2], f"r1_s{sd}": fte[m][3]})
                continue
            net = train(Gpl, v, sd, a.steps); c, k, _, _ = run(net, Gtr); ntr = Gtr.p0
            for m, Gx in Gte.items():
                r[m][f"conf_tr_s{sd}"] = c; r[m][f"corr_tr_s{sd}"] = k; ct, kt_, apv, r1 = run(net, Gx)
                r[m].update({f"conf_te_s{sd}": ct, f"corr_te_s{sd}": kt_, f"ap_s{sd}": apv, f"r1_s{sd}": r1})
                if a.save_null:                                    # 11-05 EXT-M: p(no-match) for the margin rule
                    r[m].update({f"null_tr_s{sd}": ntr, f"null_te_s{sd}": Gx.p0})
        for m in zte:
            r[m]["Etr"] = ztr["E"]; r[m].update(E=zte[m]["E"], K=zte[m]["K"], N=zte[m]["N"]); out = os.path.join(od, f"{a.target}_{a.enc}_{v}{a.tag}_x{a.data}_{m}.npz"); np.savez(out + ".tmp.npz", **r[m]); os.replace(out + ".tmp.npz", out)
        print("saved", v, flush=True)


if __name__ == "__main__":
    main()
