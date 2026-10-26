| level | dCor vs ground truth | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.8167 | 0.5582 | R=16 · per generation | cosine |
| functional | 0.9437 | 0.0317 | all 33 layers (reference) | cosine |
| structural | 0.9640 | 0.0141 | full-attn · late third | cosine |
| dataset_embedding | 0.9900 | **0.0103** | dataset text · mean · n500_s00 | frobenius |


## Every surrogate, per level

### behavioral

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.8167 | 0.5582 | R=16 · per generation | cosine |
| 0.8152 | 0.7822 | R=16 · per generation | frobenius |
| 0.8152 | 0.7822 | R=16 · per generation | euclidean |
| 0.7974 | 0.6005 | greedy · per generation | bw |
| 0.7945 | 0.2003 | R=16 · per query | euclidean |
| 0.7945 | 0.2003 | R=16 · per query | frobenius |
| 0.7933 | **0.1093** | R=16 · per query | cosine |
| 0.7904 | **0.1824** | R=16 · per query | bw |
| 0.7792 | 0.4332 | R=16 · per generation | bw |
| 0.7702 | 0.2434 | greedy · per generation | cosine |
| 0.7667 | 0.5565 | greedy · per generation | frobenius |
| 0.7667 | 0.5565 | greedy · per generation | euclidean |
| 0.7489 | 0.6648 | R=16 · per generation | cka |
| 0.7416 | **0.1802** | R=16 · per query | cka |
| 0.5117 | 0.4935 | greedy · per generation | cka |

### functional

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9437 | **0.0317** | all 33 layers (reference) | cosine |
| 0.9431 | 0.0530 | all 33 layers (reference) | bw |
| 0.9431 | 0.0523 | all 33 layers (reference) | euclidean |
| 0.9431 | 0.0523 | all 33 layers (reference) | frobenius |
| 0.9403 | 0.0534 | full-attn outputs | frobenius |
| 0.9403 | 0.0534 | full-attn outputs | euclidean |
| 0.9402 | 0.0541 | full-attn outputs | bw |
| 0.9401 | **0.0333** | full-attn outputs | cosine |
| 0.9376 | 0.0558 | late third | frobenius |
| 0.9376 | 0.0558 | late third | euclidean |
| 0.9375 | 0.0565 | late third | bw |
| 0.9361 | **0.0391** | late third | cosine |
| 0.9193 | 0.0639 | h32 · final hidden state | frobenius |
| 0.9193 | 0.0639 | h32 · final hidden state | euclidean |
| 0.9189 | 0.0641 | h32 · final hidden state | bw |
| 0.9161 | 0.0497 | h32 · final hidden state | cosine |
| 0.8187 | 0.1381 | full-attn outputs | cka |
| 0.8128 | 0.1454 | all 33 layers (reference) | cka |
| 0.7566 | 0.1950 | late third | cka |
| 0.6954 | 0.4041 | h32 · final hidden state | cka |

### structural

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9640 | **0.0141** | full-attn · late third | cosine |
| 0.9614 | **0.0156** | full-attn · q_proj (whole) | cosine |
| 0.9606 | **0.0161** | full-attn · q,k,v (d_in 2560) | cosine |
| 0.9606 | 0.0168 | all layers · all projections | cosine |
| 0.9565 | 0.0413 | full-attn · late third | frobenius |
| 0.9555 | 0.0175 | full-attn · middle third | cosine |
| 0.9553 | 0.0207 | full-attn · k_proj | cosine |
| 0.9553 | 0.0209 | output projections (d_in 4096) | cosine |
| 0.9531 | 0.0225 | full-attn · early third | cosine |
| 0.9529 | 0.0199 | full-attn · v_proj | cosine |
| 0.9528 | 0.0436 | all layers · all projections | frobenius |
| 0.9526 | 0.0438 | full-attn · q_proj (whole) | frobenius |
| 0.9519 | 0.0443 | full-attn · q,k,v (d_in 2560) | frobenius |
| 0.9469 | 0.0481 | full-attn · k_proj | frobenius |
| 0.9453 | 0.0515 | output projections (d_in 4096) | frobenius |
| 0.9447 | 0.0466 | full-attn · middle third | frobenius |
| 0.9434 | 0.0488 | full-attn · v_proj | frobenius |
| 0.9426 | 0.0524 | full-attn · early third | frobenius |
| 0.9078 | 0.0351 | full-attn · layer 31 · o_proj | cosine |
| 0.8867 | 0.0801 | full-attn · layer 31 · o_proj | frobenius |

### dataset_embedding

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9900 | **0.0103** | dataset text · mean · n500_s00 | frobenius |
| 0.9850 | **0.0113** | dataset text · mean · n500_s00 | euclidean |
| 0.9628 | **0.0543** | dataset text · mean · n500_s00 | cosine |

