# Behavioral level across the temperature sweep

dCor runs 0→1, better higher: it scores the distance matrix and never embeds. The Procrustes residual runs 1→0, better lower: it scores the 2-D MDS configuration each panel draws, so it inherits the distortion `stress` reports. Both are against the ground-truth simplex.

| slice | cosine dCor | frobenius dCor | euclidean dCor | cka dCor | cosine Procr. | frobenius Procr. | euclidean Procr. | cka Procr. |
|---|---|---|---|---|---|---|---|---|
| greedy · per generation | 0.7462 | 0.7437 | 0.7437 | 0.6450 | 0.7458 | 0.9057 | 0.9057 | 0.8282 |
| T=0.1 · R=8 · per query | 0.7886 | 0.7841 | 0.7841 | 0.7329 | 0.3581 | 0.7267 | 0.7267 | 0.2832 |
| T=0.2 · R=8 · per query | 0.8031 | 0.8010 | 0.8010 | 0.7502 | 0.2166 | 0.6875 | 0.6875 | 0.4735 |
| T=0.3 · R=8 · per query | 0.8095 | 0.8029 | 0.8029 | 0.7811 | 0.1923 | 0.7292 | 0.7292 | 0.2014 |
| T=0.4 · R=8 · per query | 0.8001 | 0.7979 | 0.7979 | 0.7956 | 0.4171 | 0.7216 | 0.7216 | 0.4053 |
| T=0.5 · R=8 · per query | 0.7746 | 0.7742 | 0.7742 | 0.7631 | 0.3348 | 0.3514 | 0.3514 | 0.3419 |
| T=0.6 · R=8 · per query | 0.7628 | 0.7646 | 0.7646 | 0.7778 | 0.2472 | 0.5863 | 0.5863 | 0.1841 |
| T=0.7 · R=8 · per query | 0.7702 | 0.7722 | 0.7722 | 0.7959 | 0.3044 | 0.5283 | 0.5283 | 0.3384 |
| T=0.8 · R=8 · per query | 0.7474 | 0.7494 | 0.7494 | 0.7280 | 0.5045 | 0.7091 | 0.7091 | 0.6195 |
| T=0.9 · R=8 · per query | 0.7323 | 0.7339 | 0.7339 | 0.7318 | 0.4278 | 0.7337 | 0.7337 | 0.7599 |
| T=1.0 · R=8 · per query | 0.8203 | 0.8234 | 0.8234 | 0.7794 | 0.2284 | 0.4310 | 0.4310 | 0.3792 |

## Kruskal stress of the MDS fit each Procrustes residual describes

| slice | cosine | frobenius | euclidean | cka |
|---|---|---|---|---|
| greedy · per generation | 0.3404 | 0.3570 | 0.3570 | 0.3304 |
| T=0.1 · R=8 · per query | 0.3143 | 0.3438 | 0.3438 | 0.3046 |
| T=0.2 · R=8 · per query | 0.2927 | 0.3381 | 0.3381 | 0.3044 |
| T=0.3 · R=8 · per query | 0.2868 | 0.3395 | 0.3395 | 0.2783 |
| T=0.4 · R=8 · per query | 0.3001 | 0.3402 | 0.3402 | 0.2923 |
| T=0.5 · R=8 · per query | 0.3015 | 0.3332 | 0.3332 | 0.2970 |
| T=0.6 · R=8 · per query | 0.3002 | 0.3398 | 0.3398 | 0.2917 |
| T=0.7 · R=8 · per query | 0.3044 | 0.3425 | 0.3425 | 0.3040 |
| T=0.8 · R=8 · per query | 0.3208 | 0.3483 | 0.3483 | 0.3162 |
| T=0.9 · R=8 · per query | 0.3225 | 0.3495 | 0.3495 | 0.3189 |
| T=1.0 · R=8 · per query | 0.3153 | 0.3463 | 0.3463 | 0.3272 |
