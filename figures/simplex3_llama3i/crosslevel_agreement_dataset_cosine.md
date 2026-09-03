| level | dCor vs ground truth | Procrustes residual (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.9343 | 0.4623 | R=16 · per generation | bw |
| functional | 0.9763 | 0.0112 | h32 · final hidden state | cosine |
| structural | 0.9629 | **0.0107** | output projections | cosine |
| dataset_embedding | 0.9371 | 0.0760 | dataset text · mean · n1000_s00 | cosine |


## Every surrogate, per level

### behavioral

| dCor | Procrustes residual (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9343 | 0.4623 | R=16 · per generation | bw |
| 0.9328 | 0.7275 | R=16 · per generation | cka |
| 0.9313 | **0.3708** | R=16 · per query | bw |
| 0.9289 | **0.1406** | R=16 · per query | cosine |
| 0.9284 | 0.3709 | R=16 · per query | euclidean |
| 0.9284 | 0.3709 | R=16 · per query | frobenius |
| 0.9273 | 0.5991 | R=16 · per generation | cosine |
| 0.9248 | 0.7457 | R=16 · per generation | euclidean |
| 0.9248 | 0.7457 | R=16 · per generation | frobenius |
| 0.8946 | **0.2808** | R=16 · per query | cka |
| 0.7906 | 0.7531 | greedy · per generation | bw |
| 0.7751 | 0.4143 | greedy · per generation | cosine |
| 0.7631 | 0.7313 | greedy · per generation | frobenius |
| 0.7631 | 0.7313 | greedy · per generation | euclidean |
| 0.6740 | 0.6669 | greedy · per generation | cka |

### functional

| dCor | Procrustes residual (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9763 | **0.0112** | h32 · final hidden state | cosine |
| 0.9733 | **0.0106** | late third | cosine |
| 0.9722 | **0.0098** | all 33 layers (reference) | cosine |
| 0.9673 | 0.0267 | h32 · final hidden state | bw |
| 0.9673 | 0.0267 | h32 · final hidden state | euclidean |
| 0.9673 | 0.0267 | h32 · final hidden state | frobenius |
| 0.9639 | 0.0283 | late third | euclidean |
| 0.9639 | 0.0283 | late third | frobenius |
| 0.9639 | 0.0283 | late third | bw |
| 0.9623 | 0.0285 | all 33 layers (reference) | frobenius |
| 0.9623 | 0.0285 | all 33 layers (reference) | euclidean |
| 0.9623 | 0.0285 | all 33 layers (reference) | bw |
| 0.8423 | 0.1470 | late third | cka |
| 0.7838 | 0.1958 | h32 · final hidden state | cka |
| 0.7812 | 0.2802 | all 33 layers (reference) | cka |

### structural

| dCor | Procrustes residual (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9629 | **0.0107** | output projections | cosine |
| 0.9620 | **0.0111** | v_proj | cosine |
| 0.9575 | 0.0123 | all layers · all projections | cosine |
| 0.9573 | **0.0118** | middle third | cosine |
| 0.9541 | 0.0133 | q,k,v (dim-pure) | cosine |
| 0.9540 | 0.0145 | late third | cosine |
| 0.9530 | 0.0137 | q_proj (whole) | cosine |
| 0.9505 | 0.0149 | k_proj | cosine |
| 0.9477 | 0.0329 | output projections | frobenius |
| 0.9472 | 0.0324 | v_proj | frobenius |
| 0.9456 | 0.0145 | early third | cosine |
| 0.9425 | 0.0334 | all layers · all projections | frobenius |
| 0.9419 | 0.0343 | middle third | frobenius |
| 0.9401 | 0.0348 | late third | frobenius |
| 0.9393 | 0.0338 | q,k,v (dim-pure) | frobenius |
| 0.9387 | 0.0338 | q_proj (whole) | frobenius |
| 0.9338 | 0.0375 | k_proj | frobenius |
| 0.9289 | 0.0376 | early third | frobenius |

### dataset_embedding

| dCor | Procrustes residual (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9800 | **0.0153** | dataset text · mean · n1000_s00 | frobenius |
| 0.9797 | **0.0152** | dataset text · mean · n1000_s00 | euclidean |
| 0.9371 | **0.0760** | dataset text · mean · n1000_s00 | cosine |

