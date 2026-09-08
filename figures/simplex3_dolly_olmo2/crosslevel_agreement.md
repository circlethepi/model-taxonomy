| level | dCor vs ground truth | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.8292 | 0.2102 | greedy · per generation | bw |
| functional | 0.9403 | 0.0562 | all 17 layers (reference) | frobenius |
| structural | 0.9414 | **0.0375** | output projections | cosine |
| dataset_embedding | 0.9119 | 0.0612 | dataset text · mean · n1000_s00 | euclidean |


## Every surrogate, per level

### behavioral

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.8292 | 0.2102 | greedy · per generation | bw |
| 0.8177 | **0.1221** | greedy · per generation | cosine |
| 0.8116 | 0.2030 | greedy · per generation | frobenius |
| 0.8116 | 0.2030 | greedy · per generation | euclidean |
| 0.7832 | 0.1400 | R=16 · per query | cosine |
| 0.7792 | **0.1192** | R=16 · per query | bw |
| 0.7748 | **0.1232** | R=16 · per query | frobenius |
| 0.7748 | 0.1232 | R=16 · per query | euclidean |
| 0.7708 | 0.1524 | R=16 · per query | cka |
| 0.7628 | 0.2093 | greedy · per generation | cka |
| 0.7532 | 0.3634 | R=16 · per generation | bw |
| 0.3814 | 0.6579 | R=16 · per generation | cka |
| 0.3284 | 0.7152 | R=16 · per generation | frobenius |
| 0.3284 | 0.7152 | R=16 · per generation | euclidean |
| 0.3221 | 0.5288 | R=16 · per generation | cosine |

### functional

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9403 | **0.0562** | all 17 layers (reference) | frobenius |
| 0.9403 | **0.0562** | all 17 layers (reference) | euclidean |
| 0.9403 | **0.0562** | all 17 layers (reference) | bw |
| 0.9386 | 0.0613 | all 17 layers (reference) | cosine |
| 0.9336 | 0.0609 | late third | euclidean |
| 0.9336 | 0.0609 | late third | frobenius |
| 0.9332 | 0.0610 | late third | bw |
| 0.9282 | 0.0786 | late third | cosine |
| 0.9186 | 0.0910 | late third | cka |
| 0.9121 | 0.1244 | h16 · final hidden state | cka |
| 0.8978 | 0.0805 | h16 · final hidden state | euclidean |
| 0.8978 | 0.0805 | h16 · final hidden state | frobenius |
| 0.8971 | 0.0810 | h16 · final hidden state | bw |
| 0.8895 | 0.1124 | h16 · final hidden state | cosine |
| 0.8326 | 0.1416 | all 17 layers (reference) | cka |

### structural

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9414 | 0.0375 | output projections | cosine |
| 0.9403 | **0.0277** | late third | cosine |
| 0.9400 | **0.0320** | q_proj (whole) | cosine |
| 0.9392 | **0.0306** | k_proj | cosine |
| 0.9390 | 0.0344 | all layers · all projections | cosine |
| 0.9385 | 0.0326 | middle third | cosine |
| 0.9377 | 0.0323 | q,k,v (dim-pure) | cosine |
| 0.9369 | 0.0510 | output projections | frobenius |
| 0.9346 | 0.0509 | q_proj (whole) | frobenius |
| 0.9343 | 0.0475 | late third | frobenius |
| 0.9335 | 0.0491 | all layers · all projections | frobenius |
| 0.9333 | 0.0488 | k_proj | frobenius |
| 0.9319 | 0.0491 | q,k,v (dim-pure) | frobenius |
| 0.9315 | 0.0528 | middle third | frobenius |
| 0.9302 | 0.0400 | v_proj | cosine |
| 0.9259 | 0.0500 | early third | cosine |
| 0.9241 | 0.0529 | v_proj | frobenius |
| 0.9218 | 0.0572 | early third | frobenius |

### dataset_embedding

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9119 | **0.0612** | dataset text · mean · n1000_s00 | euclidean |
| 0.9038 | **0.0645** | dataset text · mean · n1000_s00 | frobenius |
| 0.8527 | **0.2248** | dataset text · mean · n1000_s00 | cosine |

