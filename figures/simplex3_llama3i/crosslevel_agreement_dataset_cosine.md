| level | dCor vs ground truth | Procrustes residual (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.9343 | 0.4623 | R=16 · per generation | bw |
| functional | 0.9763 | 0.0112 | h32 · final hidden state | cosine |
| structural | 0.9629 | **0.0107** | output projections | cosine |
| dataset_embedding | 0.9371 | 0.0760 | dataset text · mean · n1000_s00 | cosine |


The full per-level ranking these winners are drawn from is in `crosslevel_agreement.md`.
The override picks a different row out of that ranking; it does not change how any row scores.
