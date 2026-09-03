| level | dCor vs ground truth | Procrustes residual (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.9398 | 0.0869 | R=16 · per generation | cosine |
| functional | 0.9790 | **0.0063** | late third | cosine |
| structural | 0.9706 | 0.0087 | v_proj | cosine |
| dataset_embedding | 0.9371 | 0.0760 | dataset text · mean · n1000_s00 | cosine |


The full per-level ranking these winners are drawn from is in `crosslevel_agreement.md`.
This variant picks different rows out of that ranking; it does not change how any row scores.
