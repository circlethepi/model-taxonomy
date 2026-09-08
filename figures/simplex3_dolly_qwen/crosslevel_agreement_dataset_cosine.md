| level | dCor vs ground truth | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.7974 | 0.2315 | R=16 · per generation | bw |
| functional | 0.9086 | 0.0678 | all 33 layers (reference) | cosine |
| structural | 0.9340 | **0.0316** | full-attn · late third | cosine |
| dataset_embedding | 0.8527 | 0.2248 | dataset text · mean · n1000_s00 | cosine |


The full per-level ranking these winners are drawn from is in `crosslevel_agreement.md`.
This variant picks different rows out of that ranking; it does not change how any row scores.
