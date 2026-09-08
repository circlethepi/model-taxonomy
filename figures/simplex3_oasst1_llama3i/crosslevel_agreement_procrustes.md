| level | dCor vs ground truth | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.7894 | 0.1859 | R=16 · per query | cosine |
| functional | 0.9206 | 0.0628 | all 33 layers (reference) | bw |
| structural | 0.9639 | 0.0185 | middle third | cosine |
| dataset_embedding | 0.9900 | **0.0103** | dataset text · mean · n500_s00 | frobenius |


The full per-level ranking these winners are drawn from is in `crosslevel_agreement.md`.
This variant picks different rows out of that ranking; it does not change how any row scores.
