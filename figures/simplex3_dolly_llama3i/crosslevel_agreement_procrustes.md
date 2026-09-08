| level | dCor vs ground truth | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.8217 | 0.1175 | R=16 · per query | cosine |
| functional | 0.9006 | 0.0862 | all 33 layers (reference) | bw |
| structural | 0.9245 | **0.0418** | middle third | cosine |
| dataset_embedding | 0.9119 | 0.0612 | dataset text · mean · n1000_s00 | euclidean |


The full per-level ranking these winners are drawn from is in `crosslevel_agreement.md`.
This variant picks different rows out of that ranking; it does not change how any row scores.
