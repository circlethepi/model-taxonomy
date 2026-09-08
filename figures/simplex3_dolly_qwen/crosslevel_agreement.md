| level | dCor vs ground truth | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.7974 | 0.2315 | R=16 · per generation | bw |
| functional | 0.9086 | 0.0678 | all 33 layers (reference) | cosine |
| structural | 0.9340 | **0.0316** | full-attn · late third | cosine |
| dataset_embedding | 0.9119 | 0.0612 | dataset text · mean · n1000_s00 | euclidean |


## Every surrogate, per level

### behavioral

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.7974 | 0.2315 | R=16 · per generation | bw |
| 0.7894 | **0.1779** | R=16 · per query | bw |
| 0.7789 | **0.1784** | R=16 · per query | cosine |
| 0.7715 | 0.2054 | R=16 · per query | frobenius |
| 0.7715 | 0.2054 | R=16 · per query | euclidean |
| 0.7497 | 0.6573 | greedy · per generation | bw |
| 0.7280 | **0.2034** | greedy · per generation | cosine |
| 0.7256 | 0.6241 | greedy · per generation | frobenius |
| 0.7256 | 0.6241 | greedy · per generation | euclidean |
| 0.6541 | 0.4405 | R=16 · per query | cka |
| 0.5742 | 0.6305 | greedy · per generation | cka |
| 0.4845 | 0.5826 | R=16 · per generation | cosine |
| 0.4834 | 0.8744 | R=16 · per generation | euclidean |
| 0.4834 | 0.8744 | R=16 · per generation | frobenius |
| 0.4567 | 0.6049 | R=16 · per generation | cka |

### functional

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9086 | **0.0678** | all 33 layers (reference) | cosine |
| 0.9072 | **0.0735** | all 33 layers (reference) | frobenius |
| 0.9072 | **0.0735** | all 33 layers (reference) | euclidean |
| 0.9070 | 0.0750 | all 33 layers (reference) | bw |
| 0.9043 | 0.0809 | full-attn outputs | euclidean |
| 0.9043 | 0.0809 | full-attn outputs | frobenius |
| 0.9039 | 0.0815 | full-attn outputs | bw |
| 0.9037 | 0.0777 | full-attn outputs | cosine |
| 0.8954 | 0.0854 | late third | frobenius |
| 0.8954 | 0.0854 | late third | euclidean |
| 0.8949 | 0.0862 | late third | bw |
| 0.8921 | 0.0959 | late third | cosine |
| 0.8859 | 0.1276 | h32 · final hidden state | frobenius |
| 0.8859 | 0.1276 | h32 · final hidden state | euclidean |
| 0.8850 | 0.1297 | h32 · final hidden state | bw |
| 0.8741 | 0.1395 | h32 · final hidden state | cosine |
| 0.8338 | 0.1589 | h32 · final hidden state | cka |
| 0.8070 | 0.1765 | full-attn outputs | cka |
| 0.7993 | 0.1841 | late third | cka |
| 0.7719 | 0.2089 | all 33 layers (reference) | cka |

### structural

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9340 | **0.0316** | full-attn · late third | cosine |
| 0.9288 | 0.0356 | full-attn · q_proj (whole) | cosine |
| 0.9282 | 0.0578 | full-attn · late third | frobenius |
| 0.9280 | **0.0345** | full-attn · q,k,v (d_in 2560) | cosine |
| 0.9270 | 0.0356 | full-attn · k_proj | cosine |
| 0.9256 | **0.0346** | full-attn · middle third | cosine |
| 0.9246 | 0.0626 | full-attn · k_proj | frobenius |
| 0.9218 | 0.0625 | full-attn · q_proj (whole) | frobenius |
| 0.9215 | 0.0621 | full-attn · q,k,v (d_in 2560) | frobenius |
| 0.9167 | 0.0666 | full-attn · middle third | frobenius |
| 0.9138 | 0.0413 | full-attn · v_proj | cosine |
| 0.9093 | 0.0773 | full-attn · v_proj | frobenius |
| 0.9023 | 0.0469 | all layers · all projections | cosine |
| 0.9014 | 0.0470 | output projections (d_in 4096) | cosine |
| 0.8999 | 0.0493 | full-attn · early third | cosine |
| 0.8970 | 0.1075 | full-attn · early third | frobenius |
| 0.8966 | 0.1149 | output projections (d_in 4096) | frobenius |
| 0.8965 | 0.0953 | all layers · all projections | frobenius |
| 0.8763 | 0.1059 | full-attn · layer 31 · o_proj | cosine |
| 0.8716 | 0.1067 | full-attn · layer 31 · o_proj | frobenius |

### dataset_embedding

| dCor | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|
| 0.9119 | **0.0612** | dataset text · mean · n1000_s00 | euclidean |
| 0.9038 | **0.0645** | dataset text · mean · n1000_s00 | frobenius |
| 0.8527 | **0.2248** | dataset text · mean · n1000_s00 | cosine |

