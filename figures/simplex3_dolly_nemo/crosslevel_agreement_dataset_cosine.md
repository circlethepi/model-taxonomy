| level | dCor vs ground truth | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.8703 | 0.1571 | R=16 · per generation | cosine |
| functional | 0.8908 | 0.0834 | all 41 layers (reference) | cosine |
| structural | 0.9193 | **0.0431** | middle third | cosine |
| dataset_embedding | 0.8527 | 0.2248 | dataset text · mean · n1000_s00 | cosine |


The full per-level ranking these winners are drawn from is in `crosslevel_agreement.md`.
This variant picks different rows out of that ranking; it does not change how any row scores.
