# Figure data for GPT to draw (no cluster / npz needed)

All values are from the saved result JSONs; medians over 27 settings where noted. SetMatch's
acceptance rule is the candidate–no-match **margin**; `p_max` is the no-match ablation.

## Fig 4(a) — Joint learning improves candidate choice
`fig4a_cho_hard.csv`: one row per setting (27). Columns:
- `CHO_diff_r1_sm_minus_sel` — forced-choice accuracy of SetMatch minus an architecture-matched
  selector without the no-match output (threshold-independent). Paired median **+0.0064**, SetMatch
  higher in **24/27**.
- `HARD_diff_sm_minus_sch` — on present queries whose cosine top-1 is a distractor but the true
  match is in the shortlist, the accept-and-correct rate of SetMatch minus SCH (at each rule's own
  frozen threshold). Paired median **+0.0360**, SetMatch higher in **23/27**.
Plot as two separate dot columns (different events, independent y-axes), points around a zero line,
coloured by `encoder_family` (ReID / general / SeqNet). Label CHO = forced choice, HARD = accept+correct.
Do NOT merge the two scales; do not add confidence intervals (per-setting CIs not exported here).

## Fig 4(b) — No-match evidence improves rejection
`fig4b_dir.csv`: DIR at matched **test** FAR 1% and 5% for GS-MLP, SCH, SetMatch (margin), and the
SetMatch p_max ablation. Dot/lollipop; emphasise margin .610 vs p_max .284 at 1% FAR. Caption must say
"matched test FAR" (threshold set on test absent queries — not a deployable point; see Table 2 for the
source-fixed version).

## Fig 1 / Fig 5 — real person-matching images (need raw crops)
Not here. Raw CUHK-SYSU images are on the cluster at `data_raw/cuhksysu/cuhk_sysu/Image/SSM/*.jpg`
(license: non-commercial research, **no redistribution** → PDF only, never commit jpgs; face-blur).
`results/setmatch/teaser_examples.json` holds query ids/scores/ranks but not pixels. To build these,
Claude maps the query id + gallery image ids + boxes from the scene index and crops deterministically
(checkpoint & seed fixed); pending because it needs the scene index + light image I/O on the cluster.
