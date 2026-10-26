| level | dCor vs ground truth | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.8370 | 0.3016 | R=16 · per query | bw |
| functional | 0.9187 | 0.0680 | all 41 layers (reference) | bw |
| structural | 0.9633 | **0.0199** | v_proj | cosine |
| dataset_embedding | 0.9628 | 0.0543 | dataset text · mean · n500_s00 | cosine |


The full per-level ranking these winners are drawn from is in `crosslevel_agreement.md`.
This variant picks different rows out of that ranking; it does not change how any row scores.
