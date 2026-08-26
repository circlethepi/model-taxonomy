| level | dCor vs ground truth | rung | metric |
|---|---|---|---|
| behavioral | 0.8164 | R=16 · per query | cka |
| functional | 0.9746 | h32 · final hidden state | cosine |
| structural | 0.9684 | linear-attn · late third | cosine |
| dataset_embedding | 0.9371 | dataset text · mean · n1000_s00 | cosine |


## Every rung, per level

### behavioral

| dCor | rung | metric |
|---|---|---|
| 0.8164 | R=16 · per query | cka |

### functional

| dCor | rung | metric |
|---|---|---|
| 0.9746 | h32 · final hidden state | cosine |

### structural

| dCor | rung | metric |
|---|---|---|
| 0.9684 | linear-attn · late third | cosine |

### dataset_embedding

| dCor | rung | metric |
|---|---|---|
| 0.9371 | dataset text · mean · n1000_s00 | cosine |

