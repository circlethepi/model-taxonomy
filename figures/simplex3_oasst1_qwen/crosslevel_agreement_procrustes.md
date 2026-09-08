| level | dCor vs ground truth | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.7933 | 0.1093 | R=16 · per query | cosine |
| functional | 0.9437 | 0.0317 | all 33 layers (reference) | cosine |
| structural | 0.9640 | 0.0141 | full-attn · late third | cosine |
| dataset_embedding | 0.9900 | **0.0103** | dataset text · mean · n500_s00 | frobenius |


The full per-level ranking these winners are drawn from is in `crosslevel_agreement.md`.
This variant picks different rows out of that ranking; it does not change how any row scores.
