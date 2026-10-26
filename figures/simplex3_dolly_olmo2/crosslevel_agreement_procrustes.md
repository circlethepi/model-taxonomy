| level | dCor vs ground truth | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.7792 | 0.1192 | R=16 · per query | bw |
| functional | 0.9403 | 0.0562 | all 17 layers (reference) | bw |
| structural | 0.9403 | **0.0277** | late third | cosine |
| dataset_embedding | 0.9119 | 0.0612 | dataset text · mean · n1000_s00 | euclidean |


The full per-level ranking these winners are drawn from is in `crosslevel_agreement.md`.
This variant picks different rows out of that ranking; it does not change how any row scores.
