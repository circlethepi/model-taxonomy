# LoRA rank sweep — canonical levels against the mixture simplex

**rsweep** — a sweep over the LoRA rank, holding the corpus, the mixture grid,
the training draw, the test-query draw, the seeds and the optimizer fixed. See
`docs/terminology.md`.

**surrogate** — a read-time view of one taxonomy level (a choice of layers,
projections, pooling and metric), as opposed to the level itself. The eight
canonical surrogates are this project's standing per-level defaults, defined in
`figures/simplex_collection_size/sweep_group_size.py` and imported, not copied.

Eight collections of 16 adapters — the 25% yahoo 3-group simplex on
OLMo-2-0425-1B-Instruct at ranks 1, 2, 4, 8, 16, 32, 64, 128, all trained on the
same 1000-row draw (seed 0, 5008 samples seen) and read on the same 100-query
33/33/33 test draw. `lora_alpha = 2r`, so PEFT's `alpha / rank` scaling is the
constant 2 and the axis varies capacity alone. 64 rows, scored in 78 s on one
core.

Reproduce:

```
python figures/fig_structural_sweep/sweep_rsweep.py          # -> rsweep_scores.csv
python figures/fig_structural_sweep/make_figures.py --score both
```

## What the numbers say

**The structural level is the one that responds to rank, and it responds
hard.** Structural (a), all layers · o_proj, moves from disparity 0.557 /
dCor\* 0.580 at r=1 to 0.010 / 0.966 at r=16 — a 55× reduction in disparity —
and is then **flat from r=8 to r=128** (0.008–0.012). Structural (b), all four
projections, starts far better at r=1 (0.091 / 0.846) because four projections
at rank 1 already carry four directions, and saturates at the same place. So
rank buys structural legibility only up to r≈8, and nothing after.

**Structural (c), the last layer alone, is non-monotone and should not be read
as a curve.** 0.666 at r=1, **0.010 at r=2**, back up to **0.319 at r=4**, then
down through 0.073 / 0.016 / 0.010 to 0.0085 at r=128. The r=2 point is better
than r=128; the r=4 point is worse than r=1 on dCor\*. One layer's o_proj at
small rank is a handful of directions, and which ones the initialisation lands
on is not controlled here — there is one init seed per rank, so each of these is
a single sample from a distribution the figure does not measure. It needs
replicate init seeds before it supports any claim.

**The dataset-embedding level is flat to four decimals** — 0.0152 disparity,
0.9797 dCor\* at every rank — exactly as it must be: its surrogate is the mean
embedding of the training draw, which no adapter touches. It is the internal
check that the pins are right, and the reference the other levels climb toward.

**Functional is nearly flat, with a shallow optimum in the middle.** All hidden
states: 0.023 at r=1, best 0.006 at r=32, back to 0.010 at r=128. Final hidden
state behaves the same (0.020 → 0.005 at r=64 → 0.009). The all-versus-last
contrast that is the widest gap in the collection-size suite is again a **null
result** here, as it was in the nsweep.

**Behavioral does not improve with rank, and at r=128 it breaks.** The R=16 read
sits between 0.048 and 0.075 disparity across r=1…64 with no trend, then jumps
to **0.433 at r=128** while its dCor\* barely moves (0.935 → 0.918). Greedy does
the same thing, larger: 0.075–0.235 across r=1…64, then **0.583 at r=128**.

## Two things to resolve before this backs a claim

1. **The r=128 behavioral disparity spike is estimator disagreement, not a
   finding yet.** Disparity moves 6–8× while dCor\* moves 0.02 over the same
   matrices. That is the same pattern the nsweep left open — "greedy Procrustes
   disparity is non-monotonic while its dCor\* stays flat; the MDS fit is the
   likeliest suspect" — now reproduced on a second axis. Two estimators over one
   matrix disagreeing about direction needs explaining before either panel is
   quotable.

2. **There are no replicates.** One init seed per rank, so every point is a
   single measurement and the wobble is unquantified. This is what makes
   structural (c) unreadable, and it is also why the small functional and
   behavioral movements should not be read as trends.

## Files

| file | what |
|---|---|
| `sweep_rsweep.py` | scores the eight collections; writes `rsweep_scores.csv` |
| `make_figures.py` | plots it; writes the two PNGs |
| `rsweep_scores.csv` | 64 rows: perspective × rank, with `dcor`, `disparity`, `lora_alpha` |
| `fig_rsweep_lora_rank.png` | panel grid, one column per level, column order shared with the nsweep and collection-size figures |
| `fig_rsweep_lora_rank_overlay.png` | the same curves on one axes per estimator |
