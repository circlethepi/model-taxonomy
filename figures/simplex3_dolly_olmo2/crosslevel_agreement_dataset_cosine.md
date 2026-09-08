| level | dCor vs ground truth | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.8292 | 0.2102 | greedy · per generation | bw |
| functional | 0.9403 | 0.0562 | all 17 layers (reference) | frobenius |
| structural | 0.9414 | **0.0375** | output projections | cosine |
| dataset_embedding | 0.8527 | 0.2248 | dataset text · mean · n1000_s00 | cosine |


The full per-level ranking these winners are drawn from is in `crosslevel_agreement.md`.
This variant picks different rows out of that ranking; it does not change how any row scores.
