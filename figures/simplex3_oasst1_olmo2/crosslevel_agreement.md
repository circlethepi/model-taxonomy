| level | dCor vs ground truth | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.8118 | 0.4520 | greedy · per generation | bw |
| functional | 0.9366 | 0.0661 | h16 · final hidden state | bw |
| structural | 0.9685 | 0.0151 | late third | cosine |
| dataset_embedding | 0.9900 | **0.0103** | dataset text · mean · n500_s00 | frobenius |


## Every surrogate, per level

### behavioral

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.8118 | 0.4520 | greedy · per generation | bw |
| 0.8022 | **0.2496** | greedy · per generation | cosine |
| 0.7983 | 0.4852 | greedy · per generation | euclidean |
| 0.7983 | 0.4852 | greedy · per generation | frobenius |
| 0.6839 | 0.3600 | greedy · per generation | cka |
| 0.6827 | 0.6563 | R=16 · per generation | bw |
| 0.6742 | 0.4359 | R=16 · per query | bw |
| 0.6704 | 0.3827 | R=16 · per query | euclidean |
| 0.6704 | 0.3827 | R=16 · per query | frobenius |
| 0.6683 | **0.2623** | R=16 · per query | cosine |
| 0.6515 | **0.2918** | R=16 · per query | cka |
| 0.4323 | 0.6285 | R=16 · per generation | cka |
| 0.4280 | 0.7302 | R=16 · per generation | frobenius |
| 0.4280 | 0.7302 | R=16 · per generation | euclidean |
| 0.4142 | 0.6580 | R=16 · per generation | cosine |

### functional

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9366 | 0.0661 | h16 · final hidden state | bw |
| 0.9362 | 0.0669 | h16 · final hidden state | frobenius |
| 0.9362 | 0.0669 | h16 · final hidden state | euclidean |
| 0.9288 | **0.0620** | all 17 layers (reference) | bw |
| 0.9287 | **0.0621** | all 17 layers (reference) | euclidean |
| 0.9287 | **0.0621** | all 17 layers (reference) | frobenius |
| 0.9239 | 0.0790 | h16 · final hidden state | cosine |
| 0.9223 | 0.0703 | late third | bw |
| 0.9222 | 0.0705 | late third | euclidean |
| 0.9222 | 0.0705 | late third | frobenius |
| 0.9191 | 0.0686 | all 17 layers (reference) | cosine |
| 0.9093 | 0.0887 | late third | cosine |
| 0.7726 | 0.2412 | h16 · final hidden state | cka |
| 0.6855 | 0.3513 | all 17 layers (reference) | cka |
| 0.6795 | 0.3803 | late third | cka |

### structural

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9685 | 0.0151 | late third | cosine |
| 0.9683 | 0.0157 | v_proj | cosine |
| 0.9680 | **0.0140** | q,k,v (dim-pure) | cosine |
| 0.9679 | 0.0148 | all layers · all projections | cosine |
| 0.9667 | **0.0146** | k_proj | cosine |
| 0.9656 | **0.0136** | middle third | cosine |
| 0.9633 | 0.0203 | output projections | cosine |
| 0.9623 | 0.0169 | q_proj (whole) | cosine |
| 0.9590 | 0.0444 | late third | frobenius |
| 0.9589 | 0.0201 | early third | cosine |
| 0.9587 | 0.0445 | all layers · all projections | frobenius |
| 0.9586 | 0.0448 | k_proj | frobenius |
| 0.9586 | 0.0435 | q,k,v (dim-pure) | frobenius |
| 0.9577 | 0.0402 | middle third | frobenius |
| 0.9575 | 0.0467 | v_proj | frobenius |
| 0.9544 | 0.0471 | output projections | frobenius |
| 0.9522 | 0.0448 | q_proj (whole) | frobenius |
| 0.9472 | 0.0513 | early third | frobenius |
| 0.8917 | 0.0756 | layer 15 · o_proj | cosine |
| 0.8898 | 0.0749 | layer 15 · o_proj | frobenius |

### dataset_embedding

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9900 | **0.0103** | dataset text · mean · n500_s00 | frobenius |
| 0.9850 | **0.0113** | dataset text · mean · n500_s00 | euclidean |
| 0.9628 | **0.0543** | dataset text · mean · n500_s00 | cosine |

