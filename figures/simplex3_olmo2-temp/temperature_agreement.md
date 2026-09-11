# Behavioral level across the temperature sweep — OLMo-2-1B, per query · cosine

## Distance correlation vs the ground-truth simplex

Higher is better. dCor scores the distance matrix directly and never embeds, so it is unaffected by the MDS fit.

| slice · surrogate | cosine |
|---|---|
| greedy · per generation (R=1: no per-query mean) | 0.8210 |
| T=0.1 · per query | 0.9162 |
| T=0.2 · per query | 0.8972 |
| T=0.3 · per query | 0.8956 |
| T=0.4 · per query | 0.9107 |
| T=0.5 · per query | 0.9053 |
| T=0.6 · per query | 0.9437 |
| T=0.7 · per query | 0.9338 |
| T=0.8 · per query | 0.9409 |
| T=0.9 · per query | 0.4074 |
| T=1.0 · per query | 0.4097 |

## Procrustes residual vs the ground-truth simplex

**Lower** is better. The residual scores the 2-D MDS configuration each panel draws, so unlike dCor it inherits the distortion `stress` reports below.

| slice · surrogate | cosine |
|---|---|
| greedy · per generation (R=1: no per-query mean) | 0.2351 |
| T=0.1 · per query | 0.0433 |
| T=0.2 · per query | 0.0446 |
| T=0.3 · per query | 0.0489 |
| T=0.4 · per query | 0.0427 |
| T=0.5 · per query | 0.0646 |
| T=0.6 · per query | 0.0639 |
| T=0.7 · per query | 0.0715 |
| T=0.8 · per query | 0.0459 |
| T=0.9 · per query | 0.6707 |
| T=1.0 · per query | 0.4777 |

## Kruskal stress of the MDS fit

Lower is better. This is the fit each Procrustes residual above describes — a high stress means that row's residual is scoring a configuration that represents its distance matrix poorly.

| slice · surrogate | cosine |
|---|---|
| greedy · per generation (R=1: no per-query mean) | 0.3044 |
| T=0.1 · per query | 0.2147 |
| T=0.2 · per query | 0.2088 |
| T=0.3 · per query | 0.2138 |
| T=0.4 · per query | 0.2190 |
| T=0.5 · per query | 0.2407 |
| T=0.6 · per query | 0.2511 |
| T=0.7 · per query | 0.2688 |
| T=0.8 · per query | 0.2785 |
| T=0.9 · per query | 0.2814 |
| T=1.0 · per query | 0.2900 |

## Cell per slice

The panels of `fig_behavioral_canonical_mds.png`: the single perspective this run reads the behavioral level at, per slice.

| slice | surrogate | metric | dCor | Procrustes | stress |
|---|---|---|---|---|---|
| greedy | per generation (R=1: no per-query mean) | cosine | 0.8210 | 0.2351 | 0.3044 |
| T=0.1 | per query | cosine | 0.9162 | 0.0433 | 0.2147 |
| T=0.2 | per query | cosine | 0.8972 | 0.0446 | 0.2088 |
| T=0.3 | per query | cosine | 0.8956 | 0.0489 | 0.2138 |
| T=0.4 | per query | cosine | 0.9107 | 0.0427 | 0.2190 |
| T=0.5 | per query | cosine | 0.9053 | 0.0646 | 0.2407 |
| T=0.6 | per query | cosine | 0.9437 | 0.0639 | 0.2511 |
| T=0.7 | per query | cosine | 0.9338 | 0.0715 | 0.2688 |
| T=0.8 | per query | cosine | 0.9409 | 0.0459 | 0.2785 |
| T=0.9 | per query | cosine | 0.4074 | 0.6707 | 0.2814 |
| T=1.0 | per query | cosine | 0.4097 | 0.4777 | 0.2900 |

