| level | dCor vs ground truth | Procrustes residual (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.9051 | 0.7217 | R=16 · per query | bw |
| functional | 0.9752 | **0.0088** | all 41 layers (reference) | cosine |
| structural | 0.9552 | 0.0139 | output projections | cosine |
| dataset_embedding | 0.9371 | 0.0760 | dataset text · mean · n1000_s00 | cosine |


The full per-level ranking these winners are drawn from is in `crosslevel_agreement.md`.
This variant picks different rows out of that ranking; it does not change how any row scores.
