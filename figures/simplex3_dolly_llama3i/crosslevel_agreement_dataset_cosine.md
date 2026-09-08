| level | dCor vs ground truth | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.8401 | 0.2353 | R=16 · per generation | bw |
| functional | 0.9011 | 0.0868 | all 33 layers (reference) | frobenius |
| structural | 0.9245 | **0.0418** | middle third | cosine |
| dataset_embedding | 0.8527 | 0.2248 | dataset text · mean · n1000_s00 | cosine |


The full per-level ranking these winners are drawn from is in `crosslevel_agreement.md`.
This variant picks different rows out of that ranking; it does not change how any row scores.
