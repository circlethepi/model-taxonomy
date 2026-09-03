| level | dCor vs ground truth | Procrustes residual (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.9398 | 0.0869 | R=16 · per generation | cosine |
| functional | 0.9790 | **0.0063** | late third | cosine |
| structural | 0.9706 | 0.0087 | v_proj | cosine |
| dataset_embedding | 0.9800 | 0.0153 | dataset text · mean · n1000_s00 | frobenius |


## Every surrogate, per level

### behavioral

| dCor | Procrustes residual (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9398 | **0.0869** | R=16 · per generation | cosine |
| 0.9383 | **0.0747** | R=16 · per query | cosine |
| 0.9360 | 0.7456 | R=16 · per generation | euclidean |
| 0.9360 | 0.7456 | R=16 · per generation | frobenius |
| 0.9358 | 0.7024 | R=16 · per query | euclidean |
| 0.9358 | 0.7024 | R=16 · per query | frobenius |
| 0.9328 | 0.7331 | R=16 · per query | bw |
| 0.9197 | 0.4053 | R=16 · per generation | bw |
| 0.9109 | 0.6479 | R=16 · per generation | cka |
| 0.8757 | **0.0844** | R=16 · per query | cka |
| 0.8210 | 0.2351 | greedy · per generation | cosine |
| 0.8191 | 0.7206 | greedy · per generation | bw |
| 0.8153 | 0.7400 | greedy · per generation | frobenius |
| 0.8153 | 0.7400 | greedy · per generation | euclidean |
| 0.7735 | 0.2530 | greedy · per generation | cka |

### functional

| dCor | Procrustes residual (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9790 | **0.0063** | late third | cosine |
| 0.9770 | **0.0073** | h16 · final hidden state | cosine |
| 0.9751 | **0.0074** | all 17 layers (reference) | cosine |
| 0.9708 | 0.0191 | late third | euclidean |
| 0.9708 | 0.0191 | late third | frobenius |
| 0.9708 | 0.0190 | late third | bw |
| 0.9693 | 0.0174 | h16 · final hidden state | frobenius |
| 0.9693 | 0.0174 | h16 · final hidden state | euclidean |
| 0.9692 | 0.0174 | h16 · final hidden state | bw |
| 0.9655 | 0.0211 | all 17 layers (reference) | frobenius |
| 0.9655 | 0.0211 | all 17 layers (reference) | euclidean |
| 0.9654 | 0.0208 | all 17 layers (reference) | bw |
| 0.9412 | 0.0460 | all 17 layers (reference) | cka |
| 0.9393 | 0.0550 | late third | cka |
| 0.9229 | 0.0847 | h16 · final hidden state | cka |

### structural

| dCor | Procrustes residual (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9706 | **0.0087** | v_proj | cosine |
| 0.9705 | **0.0088** | late third | cosine |
| 0.9703 | **0.0074** | middle third | cosine |
| 0.9685 | 0.0096 | q,k,v (dim-pure) | cosine |
| 0.9683 | 0.0094 | all layers · all projections | cosine |
| 0.9675 | 0.0090 | k_proj | cosine |
| 0.9663 | 0.0100 | output projections | cosine |
| 0.9644 | 0.0135 | q_proj (whole) | cosine |
| 0.9581 | 0.0264 | late third | frobenius |
| 0.9575 | 0.0266 | v_proj | frobenius |
| 0.9568 | 0.0198 | early third | cosine |
| 0.9568 | 0.0265 | middle third | frobenius |
| 0.9559 | 0.0278 | q,k,v (dim-pure) | frobenius |
| 0.9555 | 0.0276 | all layers · all projections | frobenius |
| 0.9542 | 0.0284 | k_proj | frobenius |
| 0.9526 | 0.0302 | q_proj (whole) | frobenius |
| 0.9522 | 0.0277 | output projections | frobenius |
| 0.9441 | 0.0357 | early third | frobenius |

### dataset_embedding

| dCor | Procrustes residual (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9800 | **0.0153** | dataset text · mean · n1000_s00 | frobenius |
| 0.9797 | **0.0152** | dataset text · mean · n1000_s00 | euclidean |
| 0.9371 | **0.0760** | dataset text · mean · n1000_s00 | cosine |

