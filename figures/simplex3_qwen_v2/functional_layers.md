# Functional level — what each layer row is

The functional level compares the 16 simplex3 adapters by their **hidden states**
on the shared 100-question draw (`recipe_hash 6149cf8055bac2c1`, `seed 1`,
prompt format `ea27ccee`), mean-pooled over tokens, per-layer normalized, and
concatenated across the selected layers:

```python
{"mode": "input", "pooling": "mean", "view": "concat",
 "normalize": "layer", "layers": <the row>, "max_new_tokens": None}
```

Only `layers` changes between rows. Everything below is a *selection of hidden
states*, not a different metric.

## Indexing convention

Qwen3.5-4B has 32 transformer layers, so there are **33 hidden states**:
`h0` is the embedding output, and `h_{L+1}` is the output of transformer layer
`L`. The model is hybrid (`full_attention_interval: 4`): layer `L` is softmax
("full") attention when `L % 4 == 3`, and gated-delta-rule **linear** attention
otherwise.

| | layers | hidden states |
|---|---|---|
| full-attention | 3, 7, 11, …, 31 (8) | 4, 8, 12, …, 32 |
| linear-attention | all others (24) | 1, 2, 3, 5, 6, 7, 9, … |

## Individual-layer rows

Figures: `fig_functional_layers_dm_grid.png`, `fig_functional_layers_mds_grid.png`.

| row | hidden state | what it is |
|---|---|---|
| `h0 · embeddings (control)` | 0 | Embedding-matrix output. LoRA never touches the embeddings, so all 16 models are **identical** here — whatever distance a metric reports is its own noise floor. Negative control. |
| `h1 · first linear-attn` | 1 | Output of layer 0, the first (linear-attention) block. Earliest state LoRA can move at all. |
| `h4 · first full-attn` | 4 | Output of layer 3, the first softmax-attention block. |
| `h16 · mid-stack (full-attn)` | 16 | Output of layer 15, also a full-attention block — the mid-depth read. |
| `h32 · final hidden state` | 32 | Output of layer 31, after the model's final RMSNorm — the residual stream as the LM head receives it. A 2560-d hidden vector, **not** logits: `lm_head` is never applied. |

## Layer-grouping rows

Figures: `fig_functional_groups_dm_grid.png`, `fig_functional_groups_mds_grid.png`.
Groupings exclude `h0` (the frozen control); the thirds split `h1…h32`
contiguously into near-equal depth bands.

| row | hidden states | count |
|---|---|---|
| `all 33 layers (reference)` | `layers=None` → h0–h32 | 33 |
| `early third` | h1–h10 | 10 |
| `middle third` | h11–h21 | 11 |
| `late third` | h22–h32 | 11 |
| `full-attn outputs` | h4, h8, …, h32 | 8 |
| `linear-attn outputs` | h1, h2, h3, h5, … | 24 |
| `all 33 layers · centered` | h0–h32, then `centered("rowwise")` | 33 |
| `late third · centered` | h22–h32, then `centered("rowwise")` | 11 |

`centered("rowwise")` subtracts, from row *i* of every model, the fleet mean of
row *i*. Row *i* is question *i* of the same draw in all 16 models, so this
removes the base model's own reading of each prompt — most of a hidden state,
and identical across the fleet by construction since LoRA only perturbs it. It
is not an alternative metric; it changes what is being compared.

`all 33 layers (reference)` is also the rung used for the per-metric detail
panels `fig_functional_dm_<metric>.png` / `fig_functional_mds_<metric>.png`.

## Layer sweep

`fig_functional_layer_sweep.png` runs every one of the 33 hidden states
individually (h0 … h32), scoring each metric's distance matrix by distance
correlation against the ground-truth simplex. Dashed vertical rules mark the
full-attention outputs. h0 is expected to be flat/absent — some metrics fail
outright there because the representation is all-zero after centering by
construction.

## Absent cells

The functional grids are complete except for CKA over a grouping:
`cka_distance_matrix` takes a single (layer, projection), so it cannot span a
multi-layer selection — those cells are drawn with the reason in place.
Euclidean cells on a `centered` row duplicate their raw twin (translation
invariance) and are labelled as such rather than recomputed.
