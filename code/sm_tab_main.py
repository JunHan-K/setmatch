"""11-03 main table of the module paper, generated from results/setmatch/v5_eval.json: worst set-expansion regret /
mAP per (dataset, encoder) for cosine, GS, raw-context MLP, raw-context attention and the proposed module.
OURS selects the module variant (set after the M1/M2 verdict). Writes new_paper/tab_main.tex."""
import json
import os
import sys

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OURS = sys.argv[1] if len(sys.argv) > 1 else "ctx_eff"
DS = [("cuhk", "CUHK-SYSU"), ("cuhkdet", "CUHK prop."), ("prw", "PRW"), ("prwdet", "PRW prop."), ("market", "Market")]
ENC = [("osnet_ain_x1_0_msmt17", "OSNet"), ("transreidssl_msmt", "TReID"), ("solider_msmt", "SOLIDER"), ("dinov2", "DINOv2"), ("clip", "CLIP")]


def pick(S, base):
    for suf in ("_s10", "_q20s10", "_M", "_s5", "_cmpraw", "_cmpnom", ""):
        if base + suf in S:
            return S[base + suf]


def row(S):
    m = [S.get("raw"), S.get("sdse"), pick(S, "ctx"), S.get("attn_cmpraw") or S.get("attn_s5"), pick(S, OURS)]
    r = [x["regret_worst"] if x else None for x in m]; best = min(v for v in r if v is not None)
    f = lambda x, v: "--" if x is None else ("\\textbf{%.3f}" % v if v == best else "%.3f" % v).replace("0.", ".", 1) + "/" + ("%.3f" % x["mAP"]).replace("0.", ".", 1)
    return " & ".join(f(x, v) for x, v in zip(m, r))


def main():
    ev = json.load(open(os.path.join(R, "results/setmatch/v5_eval.json"))); L = []
    for d, dn in DS:
        cells = [(en, ev.get(f"{d}_{e}")) for e, en in ENC]; cells = [(en, c) for en, c in cells if c]
        for i, (en, c) in enumerate(cells):
            L.append((f"\\multirow{{{len(cells)}}}{{*}}{{{dn}}}" if i == 0 else "") + f" & {en} & " + row(c["summary"]) + "\\\\")
        L.append("\\midrule")
    for d, dn in (("cuhkseq", "CUHK-SYSU"), ("prwseq", "PRW")):
        c = ev.get(f"{d}_seqnet")
        if c:
            L.append(f"{dn} & SeqNet & " + row(c["summary"]) + "\\\\")
    out = ("\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{3pt}\n\\caption{Worst set-expansion regret (lower is better, best in bold) / mAP. "
           "Every learned rule uses the same frozen features, pair encoder and training episodes; they differ only in the decision layer. "
           "CUHK-SYSU and its proposals: training galleries $\\le500$, test up to 4000 ($8\\times$); PRW: training $\\le1000$, test 6111 ($6\\times$); "
           "SeqNet: official person-search embeddings.}\\label{tab:main}\n\\begin{tabular}{llccccc}\\toprule\n"
           " & & cosine & GS & raw ctx (MLP) & raw ctx (attn) & \\textbf{ours}\\\\\\midrule\n" + "\n".join(L) + "\\bottomrule\n\\end{tabular}\\end{table*}\n")
    open(os.path.join(R, "new_paper/tab_main.tex"), "w").write(out); print(out)


if __name__ == "__main__":
    main()
