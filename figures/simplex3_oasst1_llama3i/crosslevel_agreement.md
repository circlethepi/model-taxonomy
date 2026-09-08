| level | dCor vs ground truth | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.7978 | 0.3559 | R=16 · per query | bw |
| functional | 0.9206 | 0.0628 | all 33 layers (reference) | bw |
| structural | 0.9664 | 0.0207 | q_proj (whole) | cosine |
| dataset_embedding | 0.9900 | **0.0103** | dataset text · mean · n500_s00 | frobenius |


## Every surrogate, per level

### behavioral

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.7978 | 0.3559 | R=16 · per query | bw |
| 0.7894 | **0.1859** | R=16 · per query | cosine |
| 0.7880 | 0.3478 | R=16 · per query | frobenius |
| 0.7880 | 0.3478 | R=16 · per query | euclidean |
| 0.7790 | 0.4180 | greedy · per generation | bw |
| 0.7672 | 0.5803 | R=16 · per generation | bw |
| 0.7548 | **0.2897** | greedy · per generation | cosine |
| 0.7545 | **0.2392** | R=16 · per query | cka |
| 0.7495 | 0.5404 | greedy · per generation | euclidean |
| 0.7495 | 0.5404 | greedy · per generation | frobenius |
| 0.5697 | 0.4405 | greedy · per generation | cka |
| 0.5490 | 0.5138 | R=16 · per generation | cka |
| 0.5423 | 0.7155 | R=16 · per generation | euclidean |
| 0.5423 | 0.7155 | R=16 · per generation | frobenius |
| 0.5380 | 0.5653 | R=16 · per generation | cosine |

### functional

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9206 | **0.0628** | all 33 layers (reference) | bw |
| 0.9205 | **0.0628** | all 33 layers (reference) | frobenius |
| 0.9205 | **0.0628** | all 33 layers (reference) | euclidean |
| 0.9182 | 0.0631 | late third | bw |
| 0.9180 | 0.0632 | late third | frobenius |
| 0.9180 | 0.0632 | late third | euclidean |
| 0.9104 | 0.0889 | all 33 layers (reference) | cosine |
| 0.9070 | 0.0941 | late third | cosine |
| 0.8939 | 0.0685 | h32 · final hidden state | bw |
| 0.8934 | 0.0689 | h32 · final hidden state | frobenius |
| 0.8934 | 0.0689 | h32 · final hidden state | euclidean |
| 0.8812 | 0.1226 | h32 · final hidden state | cosine |
| 0.6770 | 0.4392 | h32 · final hidden state | cka |
| 0.6329 | 0.4385 | all 33 layers (reference) | cka |
| 0.6294 | 0.4806 | late third | cka |

### structural

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9664 | 0.0207 | q_proj (whole) | cosine |
| 0.9654 | 0.0219 | late third | cosine |
| 0.9652 | **0.0202** | q,k,v (dim-pure) | cosine |
| 0.9639 | **0.0185** | middle third | cosine |
| 0.9634 | **0.0196** | all layers · all projections | cosine |
| 0.9632 | 0.0213 | k_proj | cosine |
| 0.9589 | 0.0438 | q_proj (whole) | frobenius |
| 0.9573 | 0.0216 | output projections | cosine |
| 0.9573 | 0.0453 | q,k,v (dim-pure) | frobenius |
| 0.9565 | 0.0443 | late third | frobenius |
| 0.9561 | 0.0450 | middle third | frobenius |
| 0.9554 | 0.0456 | all layers · all projections | frobenius |
| 0.9552 | 0.0232 | early third | cosine |
| 0.9540 | 0.0506 | k_proj | frobenius |
| 0.9506 | 0.0230 | v_proj | cosine |
| 0.9483 | 0.0470 | output projections | frobenius |
| 0.9459 | 0.0538 | early third | frobenius |
| 0.9403 | 0.0496 | v_proj | frobenius |
| 0.8515 | 0.1148 | layer 31 · o_proj | cosine |
| 0.8460 | 0.0721 | layer 31 · o_proj | frobenius |

### dataset_embedding

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9900 | **0.0103** | dataset text · mean · n500_s00 | frobenius |
| 0.9850 | **0.0113** | dataset text · mean · n500_s00 | euclidean |
| 0.9628 | **0.0543** | dataset text · mean · n500_s00 | cosine |

