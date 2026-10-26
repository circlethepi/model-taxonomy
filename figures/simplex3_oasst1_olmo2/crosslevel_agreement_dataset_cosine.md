| level | dCor vs ground truth | Procrustes residual at d=3 (lower=better) | surrogate | metric |
|---|---|---|---|---|
| behavioral | 0.8118 | 0.4520 | greedy · per generation | bw |
| functional | 0.9366 | 0.0661 | h16 · final hidden state | bw |
| structural | 0.9685 | **0.0151** | late third | cosine |
| dataset_embedding | 0.9628 | 0.0543 | dataset text · mean · n500_s00 | cosine |


The full per-level ranking these winners are drawn from is in `crosslevel_agreement.md`.
This variant picks different rows out of that ranking; it does not change how any row scores.
