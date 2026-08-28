# Visualization

`src/plots/simplex.py` is the colour system and panel grid for 3-group simplex
experiments, and `scripts/make_simplex3_figures.py` regenerates the whole figure suite
from the shared cache.

## Why a barycentric colour system

Earlier suites mixed *two* topic groups, so a model's composition was a scalar and a 1-D
ramp (`plasma` keyed to `% topic 0`) could carry it. The simplex3 suites mix *three*
groups, so composition is a point in a 2-simplex and no single ramp can represent it
without discarding an axis.

The replacement is a **barycentric blend**: a model's colour *is* its own mixture. Three
anchor hues sit at the pure vertices and every interior point is their weight-average,
mixed in **Oklab** so equal weight steps look like equal colour steps. The legend is
therefore the simplex itself, drawn as a filled triangle, rather than a bar.

```python
ANCHORS = {"g1": "#1E6FE8",   # blue
           "g2": "#F02B3A",   # red
           "g3": "#FFC220"}   # gold
```

Chosen to stay separable under deuteranopia and to share no hue family with `copper_r`,
which distance matrices keep — so a distance matrix and an embedding can never read as
the same encoding.

The anchors were brightened on 2026-08-24 from a more muted set. The hues are unchanged;
this is the same blue/red/gold tricolor pushed up in chroma and lightness. That matters
more here than in a categorical palette: 13 of the 16 models sit in the *interior* of the
simplex, and an interior point is a three-way blend, which is always less saturated than
any of its parents. Muted anchors left the interior nearly grey.

## The plotting API

| Function | Purpose |
|---|---|
| `mixture_weights(model_id)` | The `(w1, w2, w3)` a model's name encodes |
| `mixture_label(model_id)` | Its human-readable mixture label |
| `sort_by_mixture(model_ids)` | Canonical row/panel order |
| `barycentric_color(w)` / `model_colors(ids)` | Oklab blend of the anchors |
| `ternary_legend(...)` | The filled triangle that is the legend |
| `align_to_simplex(geometry)` | Put an embedding in the simplex's own frame |
| `dm_grid(...)` / `mds_grid(...)` | Dense rung × metric grids for one level |
| `crosslevel_mds(...)` | One MDS panel per taxonomy, in a shared frame |
| `rgb_to_oklab` / `oklab_to_rgb` / `oklab_delta_e` | The colour-space primitives |

### `crosslevel_mds` differs from `mds_grid` in three ways

- **A shared frame.** Every panel goes through `align_to_simplex`, so the centre mixture
  is at the origin and the pure-g1 model is straight up in all of them. Without it, four
  independently fitted embeddings arrive in four arbitrary orientations and cannot be
  compared by eye at all. It is a similarity transform, so nothing measurable moves.
- **True 1:1 axes.** `adjustable="box"` with symmetric limits rather than
  `adjustable="datalim"`. Under `datalim`, matplotlib satisfies the aspect by stretching
  the data limits to fit whatever box the layout gives it — equal scaling in a non-square
  frame, which reads as a distorted simplex.
- **Four labels, not sixteen.** Only the three vertices and the centre are annotated;
  interior mixtures are identified by colour and named in the legend beside the panels.

The ternary legend sits **first**, before the panels, because it is the key the panels
are read through. Panel limits are **per panel**: the levels differ by roughly an order
of magnitude in absolute MDS scale (structural ~0.6 against dataset ~0.1), so a shared
limit would render three of four as a dot at the origin. What is comparable across
panels is the arrangement, not the size.

A panel may carry a Procrustes disparity in addition to its dCor. It is passed in rather
than computed inside, because `crosslevel_mds` fits its own MDS under `random_state` and
a caller that scored a *different* fit would have the figure and its own tables reporting
two numbers for one configuration. Pass the disparity computed under the same seed, or
leave it off.

## Regenerating the suite

```bash
python scripts/make_simplex3_figures.py                  # everything
python scripts/make_simplex3_figures.py --level functional
python scripts/make_simplex3_figures.py --skip-sweep     # omit the 33-layer sweep
python scripts/make_simplex3_figures.py --skip-detail    # omit per-metric detail figures
python scripts/make_simplex3_figures.py --skip-surrogate # raw rungs only
python scripts/make_simplex3_figures.py --outdir figures/simplex3_qwen_v3
```

Four levels — `dataset_embedding`, `structural`, `functional`, `behavioral` — over the
16 Qwen3.5-4B adapters spanning the 3-group topic simplex. Distance matrices in
`copper_r`; MDS embeddings coloured by each model's own mixture.

### Output directories are versioned, not overwritten

`figures/simplex3_qwen/` holds the **row-permuted** figures produced before the `ids`
ordering bug was fixed; `_v2/` and `_v3/` hold the corrected suites. They are kept side
by side so the correction can be checked rather than taken on trust.

**Only the agreement tables are tracked in git** — `crosslevel_agreement.md`,
`crosslevel_agreement_dataset_cosine.md`, `functional_layers.md` and the matching
`crosslevel_scores*.csv`. The PNGs are gitignored, and are regenerated by the command
above. These tables are *generated artifacts*: edit the script, not the markdown.

### Missing cells are drawn, not dropped

Not every (rung, metric) cell exists, and the gaps are structural rather than accidental:

- **CKA**, **MMD** and **energy** all need more than one row, so none can run on a
  `model mean` representation; and `cka_distance_matrix` takes a single
  (layer, projection), so CKA cannot span a layer grouping.
- **Bures-Wasserstein** stacks per-block factors before its SVD, so every block must
  share an input dim — a selection mixing 2560-input and 4096-input projections has no BW
  value. It is also rank-1 on a single-row representation, and so carries nothing cosine
  does not.

Absent cells are drawn with the reason in place, so the constraint stays visible in the
figure.

### Surrogate rungs

Several rows apply a fleet-level transform from `src.analysis.surrogates` before
distancing. These are not alternative metrics — they change what is being compared. See
[Cross-Level Comparison](cross_level_comparison.md).

### The `dataset_cosine` variant

`cross_level(..., metric_override={"dataset_embedding": "cosine"})` asks what a level
looks like when held to the same metric as the rest of the figure, rather than to
whichever metric happens to score best on it. The dataset level is the only one whose
unrestricted winner is not `cosine`.

It is a question, not a correction: pinning the dataset level to `cosine` costs 0.043
dCor and takes that panel's MDS stress from 0.014 to 0.253. Both variants are written
for that reason, and the override variant carries a filename suffix so it sits beside
the unrestricted one rather than overwriting it.

## See also

- [Cross-Level Comparison](cross_level_comparison.md) — the scores printed in the panel
  titles and the agreement tables
- [Geometry Methods](geometry_methods.md) — MDS, PCA, UMAP
