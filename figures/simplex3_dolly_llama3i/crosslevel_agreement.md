| level | dCor vs ground truth | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.8401 | 0.2353 | R=16 · per generation | bw |
| functional | 0.9011 | 0.0868 | all 33 layers (reference) | frobenius |
| structural | 0.9245 | **0.0418** | middle third | cosine |
| dataset_embedding | 0.9119 | 0.0612 | dataset text · mean · n1000_s00 | euclidean |


## Every surrogate, per level

### behavioral

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.8401 | 0.2353 | R=16 · per generation | bw |
| 0.8278 | **0.1334** | R=16 · per query | bw |
| 0.8217 | **0.1175** | R=16 · per query | cosine |
| 0.8176 | **0.1411** | R=16 · per query | euclidean |
| 0.8176 | 0.1411 | R=16 · per query | frobenius |
| 0.7480 | 0.4386 | greedy · per generation | bw |
| 0.7475 | 0.2362 | R=16 · per query | cka |
| 0.7306 | 0.1777 | greedy · per generation | cosine |
| 0.7293 | 0.4671 | greedy · per generation | frobenius |
| 0.7293 | 0.4671 | greedy · per generation | euclidean |
| 0.6312 | 0.3180 | greedy · per generation | cka |
| 0.3685 | 0.6480 | R=16 · per generation | cka |
| 0.3664 | 0.7008 | R=16 · per generation | euclidean |
| 0.3664 | 0.7008 | R=16 · per generation | frobenius |
| 0.3631 | 0.5978 | R=16 · per generation | cosine |

### functional

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9011 | **0.0868** | all 33 layers (reference) | frobenius |
| 0.9011 | **0.0868** | all 33 layers (reference) | euclidean |
| 0.9006 | **0.0862** | all 33 layers (reference) | bw |
| 0.8991 | 0.1040 | all 33 layers (reference) | cosine |
| 0.8948 | 0.1347 | h32 · final hidden state | frobenius |
| 0.8948 | 0.1347 | h32 · final hidden state | euclidean |
| 0.8948 | 0.0932 | late third | frobenius |
| 0.8948 | 0.0932 | late third | euclidean |
| 0.8946 | 0.1351 | h32 · final hidden state | bw |
| 0.8941 | 0.0933 | late third | bw |
| 0.8897 | 0.1221 | late third | cosine |
| 0.8804 | 0.1754 | h32 · final hidden state | cosine |
| 0.8270 | 0.1839 | h32 · final hidden state | cka |
| 0.6777 | 0.3323 | all 33 layers (reference) | cka |
| 0.6411 | 0.3782 | late third | cka |

### structural

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9245 | **0.0418** | middle third | cosine |
| 0.9226 | 0.0490 | q_proj (whole) | cosine |
| 0.9190 | 0.0627 | middle third | frobenius |
| 0.9184 | 0.0494 | q,k,v (dim-pure) | cosine |
| 0.9182 | 0.0646 | q_proj (whole) | frobenius |
| 0.9151 | 0.0570 | late third | cosine |
| 0.9150 | 0.0491 | all layers · all projections | cosine |
| 0.9138 | 0.0650 | q,k,v (dim-pure) | frobenius |
| 0.9137 | **0.0490** | k_proj | cosine |
| 0.9117 | 0.0660 | late third | frobenius |
| 0.9101 | 0.0736 | all layers · all projections | frobenius |
| 0.9099 | 0.0755 | k_proj | frobenius |
| 0.9087 | **0.0488** | output projections | cosine |
| 0.9074 | 0.0553 | early third | cosine |
| 0.9028 | 0.0913 | output projections | frobenius |
| 0.9018 | 0.0929 | early third | frobenius |
| 0.8990 | 0.0542 | v_proj | cosine |
| 0.8913 | 0.1118 | v_proj | frobenius |

### dataset_embedding

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9119 | **0.0612** | dataset text · mean · n1000_s00 | euclidean |
| 0.9038 | **0.0645** | dataset text · mean · n1000_s00 | frobenius |
| 0.8527 | **0.2248** | dataset text · mean · n1000_s00 | cosine |

