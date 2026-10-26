| level | dCor vs ground truth | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.8392 | 0.1079 | R=16 · per query | cosine |
| functional | 0.8908 | 0.0834 | all 41 layers (reference) | cosine |
| structural | 0.9193 | **0.0431** | middle third | cosine |
| dataset_embedding | 0.9119 | 0.0612 | dataset text · mean · n1000_s00 | euclidean |


The full per-level ranking these winners are drawn from is in `crosslevel_agreement.md`.
This variant picks different rows out of that ranking; it does not change how any row scores.
