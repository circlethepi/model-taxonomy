| level | dCor vs ground truth | Procrustes residual (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.9051 | 0.7217 | R=16 · per query | bw |
| functional | 0.9752 | **0.0088** | all 41 layers (reference) | cosine |
| structural | 0.9552 | 0.0139 | output projections | cosine |
| dataset_embedding | 0.9371 | 0.0760 | dataset text · mean · n1000_s00 | cosine |


## Every surrogate, per level

### behavioral

| dCor | Procrustes residual (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9051 | 0.7217 | R=16 · per query | bw |
| 0.9043 | 0.7330 | R=16 · per generation | bw |
| 0.9010 | **0.0981** | R=16 · per query | cosine |
| 0.8969 | 0.7206 | R=16 · per query | frobenius |
| 0.8969 | **0.7206** | R=16 · per query | euclidean |
| 0.8314 | **0.1033** | R=16 · per query | cka |
| 0.7123 | 0.7826 | greedy · per generation | bw |
| 0.6969 | 0.8443 | greedy · per generation | cosine |
| 0.6950 | 0.7948 | greedy · per generation | euclidean |
| 0.6950 | 0.7948 | greedy · per generation | frobenius |
| 0.6218 | 0.7706 | R=16 · per generation | frobenius |
| 0.6218 | 0.7706 | R=16 · per generation | euclidean |
| 0.6165 | 0.7792 | R=16 · per generation | cosine |
| 0.5581 | 0.8365 | greedy · per generation | cka |
| 0.5484 | 0.7365 | R=16 · per generation | cka |

### functional

| dCor | Procrustes residual (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9752 | **0.0088** | all 41 layers (reference) | cosine |
| 0.9751 | **0.0092** | late third | cosine |
| 0.9674 | 0.0241 | late third | euclidean |
| 0.9674 | 0.0241 | late third | frobenius |
| 0.9674 | 0.0240 | late third | bw |
| 0.9673 | 0.0244 | all 41 layers (reference) | bw |
| 0.9673 | 0.0245 | all 41 layers (reference) | euclidean |
| 0.9673 | 0.0245 | all 41 layers (reference) | frobenius |
| 0.9672 | **0.0135** | h40 · final hidden state | cosine |
| 0.9596 | 0.0232 | h40 · final hidden state | euclidean |
| 0.9596 | 0.0232 | h40 · final hidden state | frobenius |
| 0.9590 | 0.0234 | h40 · final hidden state | bw |
| 0.8894 | 0.0463 | h40 · final hidden state | cka |
| 0.8634 | 0.0922 | late third | cka |
| 0.8326 | 0.1139 | all 41 layers (reference) | cka |

### structural

| dCor | Procrustes residual (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9552 | **0.0139** | output projections | cosine |
| 0.9539 | **0.0147** | v_proj | cosine |
| 0.9494 | 0.0159 | all layers · all projections | cosine |
| 0.9493 | **0.0152** | middle third | cosine |
| 0.9459 | 0.0166 | late third | cosine |
| 0.9441 | 0.0176 | q,k,v (dim-pure) | cosine |
| 0.9424 | 0.0174 | k_proj | cosine |
| 0.9418 | 0.0184 | q_proj (whole) | cosine |
| 0.9394 | 0.0379 | output projections | frobenius |
| 0.9390 | 0.0366 | v_proj | frobenius |
| 0.9336 | 0.0391 | all layers · all projections | frobenius |
| 0.9332 | 0.0374 | middle third | frobenius |
| 0.9298 | 0.0643 | late third | frobenius |
| 0.9282 | 0.0402 | q,k,v (dim-pure) | frobenius |
| 0.9275 | 0.0251 | early third | cosine |
| 0.9260 | 0.0408 | q_proj (whole) | frobenius |
| 0.9255 | 0.0415 | k_proj | frobenius |
| 0.9118 | 0.1297 | early third | frobenius |

### dataset_embedding

| dCor | Procrustes residual (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9800 | **0.0153** | dataset text · mean · n1000_s00 | frobenius |
| 0.9797 | **0.0152** | dataset text · mean · n1000_s00 | euclidean |
| 0.9371 | **0.0760** | dataset text · mean · n1000_s00 | cosine |

