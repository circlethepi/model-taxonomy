| level | dCor vs ground truth | rung | metric |
|---|---|---|---|
| behavioral | 0.8164 | R=16 · per query | cka |
| functional | 0.9746 | h32 · final hidden state | cosine |
| structural | 0.9684 | linear-attn · late third | cosine |
| dataset_embedding | 0.9371 | dataset text · mean · n1000_s00 | cosine |


The full per-level ranking these winners are drawn from is in `crosslevel_agreement.md`.
This variant picks different rows out of that ranking; it does not change how any row scores.
