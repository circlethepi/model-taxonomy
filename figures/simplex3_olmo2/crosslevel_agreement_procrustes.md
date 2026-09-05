| level | dCor vs ground truth | Procrustes residual at d=2 (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.9383 | 0.0747 | R=16 · per query | cosine |
| functional | 0.9790 | **0.0063** | late third | cosine |
| structural | 0.9703 | 0.0074 | middle third | cosine |
| dataset_embedding | 0.9797 | 0.0152 | dataset text · mean · n1000_s00 | euclidean |


The full per-level ranking these winners are drawn from is in `crosslevel_agreement.md`.
This variant picks different rows out of that ranking; it does not change how any row scores.
