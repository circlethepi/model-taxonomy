| level | dCor vs ground truth | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.8703 | 0.1571 | R=16 · per generation | cosine |
| functional | 0.8908 | 0.0834 | all 41 layers (reference) | cosine |
| structural | 0.9193 | **0.0431** | middle third | cosine |
| dataset_embedding | 0.9119 | 0.0612 | dataset text · mean · n1000_s00 | euclidean |


## Every surrogate, per level

### behavioral

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.8703 | 0.1571 | R=16 · per generation | cosine |
| 0.8690 | 0.1605 | R=16 · per generation | bw |
| 0.8683 | 0.5926 | R=16 · per generation | frobenius |
| 0.8683 | 0.5926 | R=16 · per generation | euclidean |
| 0.8524 | **0.1102** | R=16 · per query | bw |
| 0.8392 | **0.1079** | R=16 · per query | cosine |
| 0.8376 | **0.1179** | R=16 · per query | euclidean |
| 0.8376 | 0.1179 | R=16 · per query | frobenius |
| 0.8170 | 0.1838 | R=16 · per generation | cka |
| 0.8096 | 0.5779 | greedy · per generation | bw |
| 0.7954 | 0.5777 | greedy · per generation | euclidean |
| 0.7954 | 0.5777 | greedy · per generation | frobenius |
| 0.7948 | 0.1711 | greedy · per generation | cosine |
| 0.7726 | 0.1936 | R=16 · per query | cka |
| 0.6899 | 0.2254 | greedy · per generation | cka |

### functional

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.8908 | **0.0834** | all 41 layers (reference) | cosine |
| 0.8905 | **0.0841** | all 41 layers (reference) | euclidean |
| 0.8905 | **0.0841** | all 41 layers (reference) | frobenius |
| 0.8901 | 0.0843 | all 41 layers (reference) | bw |
| 0.8878 | 0.0910 | late third | euclidean |
| 0.8878 | 0.0910 | late third | frobenius |
| 0.8877 | 0.0972 | h40 · final hidden state | euclidean |
| 0.8877 | 0.0972 | h40 · final hidden state | frobenius |
| 0.8873 | 0.0916 | late third | bw |
| 0.8865 | 0.0912 | late third | cosine |
| 0.8865 | 0.1004 | h40 · final hidden state | bw |
| 0.8822 | 0.1216 | h40 · final hidden state | cosine |
| 0.8437 | 0.1106 | h40 · final hidden state | cka |
| 0.7726 | 0.2891 | late third | cka |
| 0.7596 | 0.2151 | all 41 layers (reference) | cka |

### structural

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9193 | **0.0431** | middle third | cosine |
| 0.9154 | **0.0456** | q_proj (whole) | cosine |
| 0.9138 | 0.0502 | k_proj | cosine |
| 0.9134 | 0.0673 | middle third | frobenius |
| 0.9130 | 0.0466 | q,k,v (dim-pure) | cosine |
| 0.9107 | 0.0809 | k_proj | frobenius |
| 0.9105 | **0.0461** | all layers · all projections | cosine |
| 0.9090 | 0.0732 | q_proj (whole) | frobenius |
| 0.9086 | 0.0464 | late third | cosine |
| 0.9076 | 0.0548 | early third | cosine |
| 0.9070 | 0.0731 | q,k,v (dim-pure) | frobenius |
| 0.9068 | 0.0467 | output projections | cosine |
| 0.9050 | 0.1265 | all layers · all projections | frobenius |
| 0.9047 | 0.2069 | early third | frobenius |
| 0.9020 | 0.1214 | output projections | frobenius |
| 0.9008 | 0.1426 | late third | frobenius |
| 0.8988 | 0.0509 | v_proj | cosine |
| 0.8915 | 0.1761 | v_proj | frobenius |
| 0.8669 | 0.0802 | layer 39 · o_proj | cosine |
| 0.8614 | 0.1795 | layer 39 · o_proj | frobenius |

### dataset_embedding

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9119 | **0.0612** | dataset text · mean · n1000_s00 | euclidean |
| 0.9038 | **0.0645** | dataset text · mean · n1000_s00 | frobenius |
| 0.8527 | **0.2248** | dataset text · mean · n1000_s00 | cosine |

