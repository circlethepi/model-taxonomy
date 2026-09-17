# LoRA initialisation-seed sweep — how much of a taxonomy score is seed luck

**initsweep** — a sweep over `lora_init_seed`, the seed passed to
`torch.manual_seed()` immediately before PEFT initialises the LoRA `A` and `B`
matrices, holding the corpus, the mixture grid, the training draw, the test-query
draw, the dataset seeds, the rank and every other optimizer setting fixed. See
`docs/terminology.md` and `docs/notes/init_seed_sweep.md`.

**surrogate** — a read-time view of one taxonomy level (a choice of layers,
projections, pooling and metric), as opposed to the level itself. The canonical
surrogates are defined in
`figures/simplex_collection_size/sweep_group_size.py` and imported, not copied.

**seed-siblings** — the ten adapters of one mixture trained from ten
initialisations. The **seed cloud** is where a surrogate puts them.

160 adapters: the same 16-point 25% yahoo 3-group simplex on
OLMo-2-0425-1B-Instruct, rank 16, one 1000-row draw (seed 0), from ten
initialisations `i00 .. i09`. The `i00` sixteen are `simplex3_olmo2`'s own
adapters — content-addressed caching made that seed a cache hit rather than a
retrain — so this sweep passes *through* the standing experiment rather than
beside it.

Four surrogates, no dataset row: `structural_all_o` (every layer, output
projections), `functional_all` (every hidden state), `behavioral` (per-query mean
over R=16 replicates) and `behavioral_greedy` (R=1 deterministic). The two
behavioral rows differ **by decoding alone** and are never drawn on one axes;
every figure is written twice, `_sampled` and `_greedy`.

Reproduce (all compute goes through Slurm; the 160-model pools are too heavy for
a login node):

```
sbatch jobs/initsweep_score.sh                               # -> the three CSVs
python figures/fig_structural_sweep/make_initsweep_figures.py   # -> ten PDFs
python figures/fig_structural_sweep/make_initsweep_figures.py \
    --figure overlay --seed-alpha 0.18        # fade the seeds under the means
```

Cold, the run is dominated by assembling the 160-model pools (the structural one
measured 916 s); warm it is about 10 s.

## Headline

**The structural level is dominated by initialisation and the functional level is
nearly immune to it.** Two adapters of the *same* mixture from different seeds sit
43% further apart, structurally, than two adapters of *different* mixtures that
share a seed. On the functional read the same comparison runs a factor of 13 the
other way.

**But the shape survives anyway.** Every seed's 16-point configuration agrees
with the mixture simplex to dCor\* 0.966–0.976 and Procrustes disparity
0.006–0.010, and the ten configurations land on top of each other once
superimposed. Initialisation moves *where* the adapters are; it does not move
*what shape they make*.

## Analysis A — ten replicates of the standing score

Score the 16-model simplex once per seed against one fixed ground truth (the
mixture simplex, which does not depend on the initialisation). Ten independent
values per surrogate. `mean*` is the **before-embedding mean**: the ten distance
matrices averaged and the average embedded and scored as one collection.

| surrogate | dCor\* min–max | range | mean\* | disparity min–max | range | mean\* |
|---|---|---|---|---|---|---|
| structural · all layers · o_proj | 0.9663–0.9714 | 0.0050 | 0.9695 | 0.0087–0.0104 | 0.0017 | 0.0091 |
| functional · all hidden states | 0.9741–0.9764 | 0.0024 | 0.9754 | 0.0064–0.0082 | 0.0018 | 0.0073 |
| behavioral · R=16 per query | 0.3107–0.9383 | 0.6277 | 0.7361 | 0.0747–0.7291 | 0.6544 | 0.1573 |
| behavioral · greedy | 0.8210–0.8821 | 0.0611 | 0.9159 | 0.1033–0.5326 | 0.4293 | 0.1074 |

Read: **the error bar on a structural or functional score is in the fourth
decimal place.** A single-seed structural dCor\* of 0.97 is good to ±0.003; the
project can keep quoting it.

The behavioral rows are a different story. Sampled, dCor\* over ten seeds spans
0.31 to 0.94 — the number is nearly uninformative from one seed. Greedy narrows
dCor\* to 0.061 but leaves disparity spanning 0.43, so **neither behavioral read
is safe to quote from one initialisation.** Trap 4 of the design note applies:
the sampled spread mixes initialisation with decode noise, because
`_seed_for_batch` hashes only the generation seed and the batch start, so all 160
models decode under the same RNG stream. Greedy is the clean read of
initialisation alone, and it is the one that still moves.

Both behavioral `mean*` values sit **outside** the range of the ten — 0.7361
against a median of 0.4119 sampled, 0.9159 against 0.8821 greedy. Averaging the
ten distance matrices before embedding cancels per-seed noise that no single seed
can cancel, which is exactly the property the before-embedding mean was carried
for.

Figure: `fig_initsweep_replicates_{sampled,greedy}.pdf`.

## Analysis B — separation in the 160-model pool

One collection, all ten seeds. **Three cells, not two.** A pair of adapters from
different mixtures may or may not share an initialisation, and those two cases
sit far apart, so the plain within/between ratio is not a statistic of the
surrogate at all — it is a statistic of the surrogate *and* of how many seeds
happen to be in the pool. Measured on `structural_all_o`, `between_mean` climbed
0.6021 → 0.6603 → 0.7079 → 0.7406 across pools of 2, 3, 5 and 10 seeds while
`within_mean` held at 0.617, purely because the share of same-seed pairs falls as
`1/n_seeds`. A two-cell fit to the 2- and 3-seed pools predicts the 5- and
10-seed pools to within 0.0012. **Do not quote `ratio`; quote the three cells and
`ratio*`.**

| surrogate | diff. mixture, one seed (n=1200) | seed-siblings (n=720) | diff. mixture, two seeds (n=10800) | ratio\* |
|---|---|---|---|---|
| structural · all layers · o_proj | 0.4305 | 0.6167 | 0.7751 | **1.432** |
| functional · all hidden states | 0.0188 | 0.0015 | 0.0194 | **0.079** |
| behavioral · R=16 per query | 0.0254 | 0.0199 | 0.0261 | 0.785 |
| behavioral · greedy | 0.1716 | 0.0923 | 0.1735 | 0.538 |

`ratio*` = seed-siblings ÷ one-seed neighbours: the comparison pool size does not
move. Above 1 the initialisation moves an adapter further than changing what it
was trained on.

The ordering is the finding. **Structural, 1.43: re-seeding beats re-mixing.**
Functional, 0.079: the seed is almost invisible and the mixture is almost
everything. The two behavioral reads sit between, and greedy is the tighter of
the two — consistent with sampling noise inflating the sibling distance.

Per-mixture rows show the same picture everywhere on the grid, with no vertex /
centre structure worth reporting: structural `ratio*` runs 1.11–1.71 (median
1.44), functional 0.059–0.101 (median 0.078), behavioral 0.714–0.846, greedy
0.488–0.615. **Every structural mixture is above 1**, so the headline is not
carried by a few cells.

Not reported: Procrustes or dCor\* against the truth on this pool. The truth is
degenerate here — `ground_truth_weights` derives barycentric coordinates from the
*recipe*, so all ten siblings get byte-identical weights and 720 of the 12,720
pairs are pinned at exactly 0. A dataset row cannot enter this pool at all: its
identity is the recipe, which is ten-to-one across the seeds, and `relabel()`
refuses to let distinct models collide.

Figure: `fig_initsweep_separation_{sampled,greedy}.pdf`.

## Analysis C — geometry over the full collection

**One joint fit or ten separate ones.** This is the distinction the three figures
turn on, and the two answers can differ without either being wrong.

`fig_initsweep_pool_*.pdf` — **one** 160×160 dissimilarity matrix over all 160
adapters and **one** MDS fit; the ten seeds of a mixture are placed by that joint
fit. The ringed point is not a selected adapter: it is that mixture's
**after-embedding mean**, the arithmetic centroid of its ten `pool160`
coordinates. This is analysis B as a picture — the structural panel smears each
mixture into a band wider than the spacing between mixtures, while the functional
panel collapses each mixture's ten seeds to a single visible point.

`fig_initsweep_overlay_*.pdf` — **ten** 16×16 matrices and **ten** MDS fits, one
per seed, each seed's sixteen adapters compared against each other and nobody
else. The ten configurations start in ten unrelated frames and are
Procrustes-superimposed before they are drawn. The diamond is that mixture's
**mean over the ten aligned points** (`seed_mean`) — the centre of exactly the
cloud drawn around it. `--seed-alpha` fades the individual seeds against it; the
default is 0.55, and the right value depends on the level, since no one setting
serves both a structural panel where ten points coincide and a sampled behavioral
panel that is a solid mass.

**Structural and functional: every seed lands within a marker width of the
mean.** This is the second half of the headline and it is not in tension with
analysis B. B asks where adapters sit in a joint embedding of all ten seeds; the
overlay asks whether each seed's own 16-point configuration has the same *shape*.
It does. Initialisation moves where the adapters are, not what shape they make.

`fig_initsweep_means_*.pdf` — the two means over seeds superimposed and joined per
mixture. They differ because MDS is not linear. Functional's two coincide;
structural's do not. Note there are now **three** means over seeds in the CSV and
they are three different objects: `mean_after` (centroid inside the joint fit),
`mean_before` (embedding of averaged distances) and `seed_mean` (average of ten
separate fits, after reconciling their frames).

**House orientation.** Every geometry kind is written with the pure-g1 vertex due
north of its centre and the pure-g2 vertex in positive x. MDS leaves rotation and
reflection free, so without this two panels of the same simplex can be mirror
images of each other and a reader comparing levels side by side has to re-derive
which way is which in every panel. Kinds that share a frame are turned together,
by one map computed from the sixteen mixture points in that frame, so the
superpositions are not disturbed. Scale is left alone.

## Traps

1. **The rank pin is the one that is easy to forget.** `i00` exists at eight
   ranks from the rsweep, so a filter on mixture and seed alone silently pulls in
   128 extra adapters. `sweep_initsweep.py` pins `lora_rank=16` with
   `n_samples=1000, seed=0`.
2. **`ratio` is pool-size dependent** — see analysis B. It is kept in the CSV for
   continuity and should not be quoted.
3. **Alignment rescales.** `procrustes_compare` centres and scales *both*
   configurations to unit Frobenius norm, so an aligned configuration is never on
   the same scale as the raw one it was aligned to. The raw scale differs by
   three orders of magnitude between the structural and functional rows, so
   drawing an aligned configuration against a raw one shows that ratio and
   nothing else. The geometry CSV therefore carries both sides of every
   superposition (`seed_reference`, `mean_after_aligned`) and the figure script
   pairs like with like.
4. **Dropping the dataset row drops the determinism null control.** The
   initialisation seed is not in the draw path, so all ten seeds read the same
   sixteen draws and a dataset row's ten scores must come out *exactly* equal — a
   spread there would be a finding about pipeline determinism rather than about
   initialisation. It is not in the default four. Take the reading with
   `sbatch jobs/initsweep_score.sh --perspective dataset_embedding --skip-pool`;
   it is seconds of CPU, since no adapter is touched.
5. **Behavioral spread mixes initialisation with decode noise.** See analysis A.
   Greedy is the clean read.

## Open

- Whether the structural result is a rank-16 fact or holds across the rsweep's
  eight ranks. One seed pair at each rank would answer it cheaply.
- Whether `ratio*` above 1 on the structural row means the structural level
  should be read at a scope less sensitive to the initialisation
  (`structural_all_qkvo`, `structural_last_o`) rather than being discounted.
