| level | dCor vs ground truth | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.7978 | 0.3559 | R=16 · per query | bw |
| functional | 0.9206 | 0.0628 | all 33 layers (reference) | bw |
| structural | 0.9664 | **0.0207** | q_proj (whole) | cosine |
| dataset_embedding | 0.9628 | 0.0543 | dataset text · mean · n500_s00 | cosine |


The full per-level ranking these winners are drawn from is in `crosslevel_agreement.md`.
This variant picks different rows out of that ranking; it does not change how any row scores.
