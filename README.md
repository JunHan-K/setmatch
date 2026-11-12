# SetWiseMatch

Code and result files for **"Same Score, Different Set: Candidate-Set Shift in Open-World Person Matching."**
> **Note (provisional results).** The code reflects a fix to the evaluation's episode
> construction (present/absent galleries are now size-matched so candidate-set size cannot
> leak presence; see `code/sm_v5.py`). The `results/` JSONs and `figdata/` CSVs here are from
> the pre-fix run and are being **regenerated**; treat the numbers as provisional until updated.


SetWiseMatch is a small, frozen-encoder-agnostic decision layer for open-world person matching
(person re-identification and person search). It turns a cosine shortlist into **one
categorical distribution over `{no-match ∅, candidate 1 … L}`** and makes two decisions from it:

- **choose** `ĵ = argmax_i p_i` (which candidate), and
- **accept** `ĵ` iff the candidate–no-match margin `z_ĵ − z_∅ > τ`, with a single threshold `τ`
  fixed on *training* episodes and reused for every test gallery size.

The problem it targets is **candidate-set shift**: the same query and the same score can mean a
match in a small gallery and a non-match in a large one, so a fixed cosine threshold silently
drifts as gallery size changes. SetWiseMatch sees the whole shortlist (plus gallery-level score
statistics `g(C)` and an explicit no-match token) and stays stable across sizes.

## Method in one paragraph

A frozen ReID / person-search encoder embeds the query and gallery (never updated). The gallery
is cosine-ranked and the top `L = min(64, N)` form a shortlist. Each candidate becomes a token

```
e_i = W_e [ q⊙c_i , |q−c_i| , c_i , s_i , r_i/64 , r_i/N ] + W_g g(C)
```

together with a learned, set-pooled no-match token `z_∅ = null + pool(mean_i e_i, max_i e_i)`.
Three set-attention blocks (LayerNorm, 4-head self-attention with a pairwise affinity bias
`c_i^T c_j`, MLP, residuals) mix the tokens; a linear head + softmax give `P(y | q, C)`.
At `d=512` the layer is ~0.84M parameters (~1.1M at `d=1280`) and adds ~1.5 ms per query.

## Repository layout

```
code/      Python scripts that compute the paper's numbers and draw its figures
results/   small JSON result files the figures and tables are built from
figdata/   CSV extracts used by a couple of figures
```

The large per-episode `.npz` prediction dumps and the datasets themselves are **not** included
(see *Datasets* below); the JSON files in `results/` hold the final numbers (medians and
per-cell values) each figure/table needs.

## Running the figure / table scripts

Needs `numpy` and `matplotlib` (`code/figstyle.py` sets a shared plotting style). The scripts
resolve a results directory via a variable near the top (e.g. `O = ...` / `R = ...`); point it at
this repo's `results/` directory. A few figures/tables (the DIR–FAR curves and the per-cell
main table) additionally need the per-episode `.npz` dumps, which are not redistributed here.

Each result JSON is produced by one preregistered "verdict" script, e.g. `sm_ctrl.py → ctrl.json`,
`sm_cho.py → cho.json`, `sm_hard.py → hard.json`, `sm_dra.py → dra.json`, `sm_src.py → src.json`,
`sm_rank.py → rank.json`. The experiment was preregistered before its results were seen.

## Headline numbers (27 dataset–encoder settings, medians)

- Strict accuracy: SetWiseMatch **.835** vs SCH .826, GS-MLP .822, Gallery-stats .818, size-aware
  cosine .792; better than Gallery-stats in 26/27 settings.
- Worst-size selective regret: `.182 → .059`.
- Margin acceptance: DIR @ 1% false-accept rate `.284 → .610` (same model, acceptance score only).
- Source-fixed operating points (size-aware): DIR `.621 / .775` at 1% / 5% targets vs SCH `.474 / .673`.

## Datasets

The method runs on **frozen features** extracted from standard benchmarks (CUHK-SYSU, PRW,
Market-1501, MSMT17 and transfers). Their images are governed by each dataset's own licence and
are **not** redistributed here — obtain them from the original sources and extract features with
any frozen ReID / person-search encoder.

## Licence

Code is released under the MIT Licence (see `LICENSE`). Dataset images and any third-party
encoder weights keep their own licences.
