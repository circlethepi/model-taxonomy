| level | dCor vs ground truth | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.8167 | 0.5582 | R=16 · per generation | cosine |
| functional | 0.9437 | 0.0317 | all 33 layers (reference) | cosine |
| structural | 0.9640 | **0.0141** | full-attn · late third | cosine |
| dataset_embedding | 0.9628 | 0.0543 | dataset text · mean · n500_s00 | cosine |


The full per-level ranking these winners are drawn from is in `crosslevel_agreement.md`.
This variant picks different rows out of that ranking; it does not change how any row scores.
