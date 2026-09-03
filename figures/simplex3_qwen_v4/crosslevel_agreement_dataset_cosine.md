| level | dCor vs ground truth | Procrustes residual (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.8164 | 0.1973 | R=16 · per query | cka |
| functional | 0.9746 | **0.0075** | h32 · final hidden state | cosine |
| structural | 0.9683 | 0.0082 | full-attn · late third | cosine |
| dataset_embedding | 0.9371 | 0.0760 | dataset text · mean · n1000_s00 | cosine |


## Every surrogate, per level

### behavioral

| dCor | Procrustes residual (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.8164 | **0.1973** | R=16 · per query | cka |
| 0.7942 | **0.3010** | R=16 · per query | frobenius |
| 0.7942 | **0.3010** | R=16 · per query | euclidean |
| 0.7895 | 0.3210 | R=16 · per query | cosine |
| 0.7462 | 0.7458 | greedy · per generation | cosine |
| 0.7437 | 0.9057 | greedy · per generation | frobenius |
| 0.7437 | 0.9057 | greedy · per generation | euclidean |
| 0.7392 | 0.6144 | R=16 · per query | bw |
| 0.7070 | 0.8685 | R=16 · per generation | frobenius |
| 0.7070 | 0.8685 | R=16 · per generation | euclidean |
| 0.7039 | 0.8850 | greedy · per generation | bw |
| 0.7017 | 0.7137 | R=16 · per generation | cosine |
| 0.6729 | 0.6339 | R=16 · per generation | cka |
| 0.6450 | 0.8282 | greedy · per generation | cka |
| 0.6077 | 0.6677 | R=16 · per generation | bw |

### functional

| dCor | Procrustes residual (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9746 | **0.0075** | h32 · final hidden state | cosine |
| 0.9729 | **0.0098** | late third | cosine |
| 0.9701 | **0.0107** | full-attn outputs | cosine |
| 0.9687 | 0.0118 | all 33 layers (reference) | cosine |
| 0.9640 | 0.0240 | h32 · final hidden state | frobenius |
| 0.9640 | 0.0240 | h32 · final hidden state | euclidean |
| 0.9640 | 0.0237 | h32 · final hidden state | bw |
| 0.9630 | 0.0272 | late third | bw |
| 0.9630 | 0.0274 | late third | frobenius |
| 0.9630 | 0.0274 | late third | euclidean |
| 0.9599 | 0.0296 | full-attn outputs | bw |
| 0.9598 | 0.0297 | full-attn outputs | euclidean |
| 0.9598 | 0.0297 | full-attn outputs | frobenius |
| 0.9585 | 0.0307 | all 33 layers (reference) | bw |
| 0.9584 | 0.0308 | all 33 layers (reference) | euclidean |
| 0.9584 | 0.0308 | all 33 layers (reference) | frobenius |
| 0.9440 | 0.0437 | h32 · final hidden state | cka |
| 0.9168 | 0.0386 | full-attn outputs | cka |
| 0.9154 | 0.0411 | late third | cka |
| 0.9046 | 0.0407 | all 33 layers (reference) | cka |

### structural

| dCor | Procrustes residual (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9683 | **0.0082** | full-attn · late third | cosine |
| 0.9650 | **0.0099** | full-attn · q_proj (whole) | cosine |
| 0.9645 | **0.0097** | all layers · all projections | cosine |
| 0.9643 | 0.0100 | full-attn · q,k,v (d_in 2560) | cosine |
| 0.9619 | 0.0105 | full-attn · middle third | cosine |
| 0.9601 | 0.0109 | full-attn · k_proj | cosine |
| 0.9558 | 0.0281 | full-attn · late third | frobenius |
| 0.9554 | 0.0118 | full-attn · v_proj | cosine |
| 0.9526 | 0.0121 | output projections (d_in 4096) | cosine |
| 0.9525 | 0.0150 | full-attn · early third | cosine |
| 0.9522 | 0.0316 | full-attn · q_proj (whole) | frobenius |
| 0.9515 | 0.0299 | all layers · all projections | frobenius |
| 0.9510 | 0.0318 | full-attn · q,k,v (d_in 2560) | frobenius |
| 0.9474 | 0.0338 | full-attn · middle third | frobenius |
| 0.9454 | 0.0324 | full-attn · k_proj | frobenius |
| 0.9387 | 0.0577 | full-attn · early third | frobenius |
| 0.9386 | 0.0382 | full-attn · v_proj | frobenius |
| 0.9352 | 0.0341 | output projections (d_in 4096) | frobenius |

### dataset_embedding

| dCor | Procrustes residual (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9800 | **0.0153** | dataset text · mean · n1000_s00 | frobenius |
| 0.9797 | **0.0152** | dataset text · mean · n1000_s00 | euclidean |
| 0.9371 | **0.0760** | dataset text · mean · n1000_s00 | cosine |

