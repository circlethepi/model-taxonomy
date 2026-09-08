| level | dCor vs ground truth | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.8370 | 0.3016 | R=16 · per query | bw |
| functional | 0.9187 | 0.0680 | all 41 layers (reference) | bw |
| structural | 0.9633 | 0.0199 | v_proj | cosine |
| dataset_embedding | 0.9900 | **0.0103** | dataset text · mean · n500_s00 | frobenius |


## Every surrogate, per level

### behavioral

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.8370 | 0.3016 | R=16 · per query | bw |
| 0.8251 | **0.0977** | R=16 · per query | cosine |
| 0.8242 | 0.2802 | R=16 · per query | euclidean |
| 0.8242 | 0.2802 | R=16 · per query | frobenius |
| 0.8174 | 0.5891 | R=16 · per generation | bw |
| 0.7904 | **0.1275** | R=16 · per query | cka |
| 0.7893 | 0.6193 | greedy · per generation | bw |
| 0.7683 | **0.2790** | greedy · per generation | cosine |
| 0.7627 | 0.5721 | greedy · per generation | euclidean |
| 0.7627 | 0.5721 | greedy · per generation | frobenius |
| 0.6479 | 0.5820 | R=16 · per generation | cosine |
| 0.6479 | 0.8698 | R=16 · per generation | frobenius |
| 0.6479 | 0.8698 | R=16 · per generation | euclidean |
| 0.6466 | 0.7597 | R=16 · per generation | cka |
| 0.5721 | 0.6069 | greedy · per generation | cka |

### functional

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9187 | **0.0680** | all 41 layers (reference) | bw |
| 0.9185 | 0.0688 | all 41 layers (reference) | frobenius |
| 0.9185 | 0.0688 | all 41 layers (reference) | euclidean |
| 0.9169 | **0.0602** | all 41 layers (reference) | cosine |
| 0.9145 | 0.0700 | late third | bw |
| 0.9144 | 0.0712 | h40 · final hidden state | bw |
| 0.9143 | 0.0701 | late third | frobenius |
| 0.9143 | 0.0701 | late third | euclidean |
| 0.9124 | **0.0644** | late third | cosine |
| 0.9121 | 0.0722 | h40 · final hidden state | frobenius |
| 0.9121 | 0.0722 | h40 · final hidden state | euclidean |
| 0.9073 | 0.0736 | h40 · final hidden state | cosine |
| 0.7581 | 0.2397 | late third | cka |
| 0.6827 | 0.4079 | all 41 layers (reference) | cka |
| 0.6824 | 0.4113 | h40 · final hidden state | cka |

### structural

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9633 | **0.0199** | v_proj | cosine |
| 0.9631 | **0.0195** | late third | cosine |
| 0.9622 | 0.0210 | all layers · all projections | cosine |
| 0.9620 | 0.0218 | q,k,v (dim-pure) | cosine |
| 0.9619 | 0.0217 | q_proj (whole) | cosine |
| 0.9612 | **0.0206** | output projections | cosine |
| 0.9587 | 0.0244 | k_proj | cosine |
| 0.9585 | 0.0207 | middle third | cosine |
| 0.9556 | 0.0509 | v_proj | frobenius |
| 0.9545 | 0.0517 | all layers · all projections | frobenius |
| 0.9544 | 0.0511 | late third | frobenius |
| 0.9541 | 0.0522 | q,k,v (dim-pure) | frobenius |
| 0.9541 | 0.0314 | early third | cosine |
| 0.9539 | 0.0518 | q_proj (whole) | frobenius |
| 0.9538 | 0.0518 | output projections | frobenius |
| 0.9511 | 0.0542 | k_proj | frobenius |
| 0.9503 | 0.0493 | middle third | frobenius |
| 0.9468 | 0.0602 | early third | frobenius |
| 0.9332 | 0.0337 | layer 39 · o_proj | cosine |
| 0.9194 | 0.0674 | layer 39 · o_proj | frobenius |

### dataset_embedding

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9900 | **0.0103** | dataset text · mean · n500_s00 | frobenius |
| 0.9850 | **0.0113** | dataset text · mean · n500_s00 | euclidean |
| 0.9628 | **0.0543** | dataset text · mean · n500_s00 | cosine |

