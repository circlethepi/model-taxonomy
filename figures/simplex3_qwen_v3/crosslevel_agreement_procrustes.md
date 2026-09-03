| level | dCor vs ground truth | Procrustes residual (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.8164 | 0.1973 | R=16 · per query | cka |
| functional | 0.9746 | **0.0075** | h32 · final hidden state | cosine |
| structural | 0.9684 | 0.0078 | linear-attn · late third | cosine |
| dataset_embedding | 0.9797 | 0.0152 | dataset text · mean · n1000_s00 | euclidean |


The full per-level ranking these winners are drawn from is in `crosslevel_agreement.md`.
This variant picks different rows out of that ranking; it does not change how any row scores.
