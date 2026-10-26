| level | dCor vs ground truth | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.8022 | 0.2496 | greedy · per generation | cosine |
| functional | 0.9288 | 0.0620 | all 17 layers (reference) | bw |
| structural | 0.9656 | 0.0136 | middle third | cosine |
| dataset_embedding | 0.9900 | **0.0103** | dataset text · mean · n500_s00 | frobenius |


The full per-level ranking these winners are drawn from is in `crosslevel_agreement.md`.
This variant picks different rows out of that ranking; it does not change how any row scores.
