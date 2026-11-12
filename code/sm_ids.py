"""11-04 IDS verdict (PREREG): strict frozen accuracy of SetWiseMatch and GS, and selective worst regret, as the number of
training identities used as queries shrinks (N in {56, 150, 400} vs all). Full level = existing GS base and SetWiseMatch
s10main seeds 0-4. Uses sm_ctrl.evaluate unchanged. Writes results/setmatch/ids.json and new_paper/figs/ids.pdf."""
import json
import os

import numpy as np

H = os.path.dirname(os.path.abspath(__file__)); src = open(os.path.join(H, "sm_ctrl.py")).read(); exec(src[:src.index("\nres, dropped = {}, {}")])   # noqa: S102
NS = (56, 150, 400, "all"); COMB = [(d, e) for d in ("market", "cuhk") for e in ("osnet_ain_x1_0_msmt17", "solider_msmt", "clip")]; res = {}
for d, e in COMB:
    key = f"{d}_{e}"; r = {}
    for n in NS:
        tag = "" if n == "all" else f"_id{n}"; z = np.load(os.path.join(V5, f"{key}_meta{tag}.npz")); y = np.load(os.path.join(V5, f"{key}_attn{'_s10main' if n == 'all' else tag}.npz"))
        E, K = z["E"], z["K"]; S = [s for s in range(5) if f"conf_te_s{s}" in y.files]
        gs = evaluate(z["sdse_tr"], z["Vtr"], z["Etr"], z["sdse_te"], z["Vte"], E, K, z["cos_ap"], z["cos_r1"], "logit")
        sm = [evaluate(y[f"conf_tr_s{s}"], y[f"corr_tr_s{s}"], y["Etr"], y[f"conf_te_s{s}"], y[f"corr_te_s{s}"], E, K, y[f"ap_s{s}"], y[f"r1_s{s}"], "prob") for s in S]
        r[str(n)] = dict(gs_acc=gs["acc_mean"], sm_acc=float(np.mean([o["acc_mean"] for o in sm])), gs_sel=gs["sel_worst"], sm_sel=float(np.mean([o["sel_worst"] for o in sm])),
                         sm_oracle=float(np.mean([o["oracle_mean"] for o in sm])), gs_oracle=gs["oracle_mean"], seeds=len(S))
        r[str(n)]["delta"] = r[str(n)]["sm_acc"] - r[str(n)]["gs_acc"]
    res[key] = r; print(key.ljust(28), "  ".join(f"N={n}: d {v['delta']:+.3f} (sm {v['sm_acc']:.3f} gs {v['gs_acc']:.3f}) sel {v['sm_sel']:.3f}/{v['gs_sel']:.3f}" for n, v in r.items()), flush=True)
c1 = sum(res[k]["56"]["delta"] <= 0 for k in res); c2 = sum(res[k]["all"]["delta"] - res[k]["56"]["delta"] >= .01 for k in res)
V = dict(c1_delta56_le0=c1, c2_gain_ge01=c2, verdict="few training identities favour GS (consistent with RPIfield)" if c1 >= 4 and c2 >= 4 else "training identities alone do not explain RPIfield",
         med_delta={str(n): float(np.median([res[k][str(n)]["delta"] for k in res])) for n in NS})
print(json.dumps(V)); json.dump(dict(cells=res, verdict=V), open(os.path.join(O, "ids.json"), "w"), indent=1)
import sys
sys.path.insert(0, H); import figstyle as F
from figstyle import plt
fig, axs = plt.subplots(1, 2, figsize=(F.COL, 1.45), sharey=True); xs = [56, 150, 400]
for ax, d, t in zip(axs, ("market", "cuhk"), ("Market-1501", "CUHK-SYSU")):
    full = {"market": 751, "cuhk": 5532}[d]
    for e, mk in (("osnet_ain_x1_0_msmt17", "o"), ("solider_msmt", "s"), ("clip", "^")):
        v = res[f"{d}_{e}"]; ax.plot(xs + [full], [v[str(n)]["delta"] for n in NS], marker=mk, c=F.C["sm"], lw=1.1, ms=3, label={"osnet_ain_x1_0_msmt17": "OSNet", "solider_msmt": "SOLIDER", "clip": "CLIP"}[e])
    ax.axhline(0, c=F.INK2, lw=0.6); ax.axvline(56, c=F.MUTED, lw=0.5, ls=":"); ax.set_xscale("log"); ax.set_title(t, pad=2); ax.set_xlabel("training identities")
axs[0].set_ylabel("accuracy: SetWiseMatch $-$ GS"); axs[0].legend(loc="lower right", handletextpad=0.3); fig.tight_layout(w_pad=0.6)
P = os.path.join(R, "new_paper/figs"); fig.savefig(os.path.join(P, "ids.pdf")); fig.savefig(os.path.join(P, "ids.png"), dpi=200)
