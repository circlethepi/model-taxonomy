| level | dCor vs ground truth | rung | metric |
|---|---|---|---|
| behavioral | 0.8164 | R=16 · per query | cka |
| functional | 0.9746 | h32 · final hidden state | cosine |
| structural | 0.9684 | linear-attn · late third | cosine |
| dataset_embedding | 0.9800 | dataset text · mean · n1000_s00 | frobenius |


## Every rung, per level

### behavioral

| dCor | rung | metric |
|---|---|---|
| 0.8164 | R=16 · per query | cka |
| 0.7942 | R=16 · per query | frobenius |
| 0.7942 | R=16 · per query | euclidean |
| 0.7895 | R=16 · per query | cosine |
| 0.7462 | greedy · per generation | cosine |
| 0.7437 | greedy · per generation | frobenius |
| 0.7437 | greedy · per generation | euclidean |
| 0.7392 | R=16 · per query | bw |
| 0.7357 | R=16 · per generation · whitened | cosine |
| 0.7292 | R=16 · per generation · whitened | frobenius |
| 0.7234 | R=16 · per generation · centered | cosine |
| 0.7218 | R=16 · per generation · centered | frobenius |
| 0.7200 | R=16 · per generation · whitened | euclidean |
| 0.7070 | R=16 · per generation | frobenius |
| 0.7070 | R=16 · per generation | euclidean |
| 0.7039 | greedy · per generation | bw |
| 0.7017 | R=16 · per generation | cosine |
| 0.6729 | R=16 · per generation | cka |
| 0.6456 | R=16 · per generation · whitened | bw |
| 0.6450 | greedy · per generation | cka |
| 0.6077 | R=16 · per generation | bw |
| 0.4751 | R=16 · per generation · whitened | energy |
| 0.4674 | R=16 · per generation · whitened | mmd |
| 0.4425 | R=16 · per generation · centered | bw |
| 0.2915 | greedy · model mean | frobenius |
| 0.2915 | greedy · model mean | euclidean |
| 0.2718 | greedy · model mean | cosine |
| 0.2685 | R=16 · model mean · centered | cosine |
| 0.2588 | R=16 · model mean · centered | frobenius |
| 0.2206 | R=16 · per generation | energy |
| 0.2174 | R=16 · per generation · centered | energy |
| 0.2173 | R=16 · per generation · centered | mmd |
| 0.2156 | R=16 · per generation | mmd |
| 0.2080 | R=16 · model mean | cosine |
| 0.2064 | R=16 · model mean | euclidean |
| 0.2064 | R=16 · model mean | frobenius |
| 0.1961 | R=16 · per query | mmd |
| 0.1797 | greedy · per generation | mmd |
| 0.1774 | greedy · per generation | energy |
| 0.1619 | R=16 · per query | energy |
| -0.5479 | R=16 · per generation · centered | cka |
| -0.6741 | R=16 · per generation · whitened | cka |

### functional

| dCor | rung | metric |
|---|---|---|
| 0.9746 | h32 · final hidden state | cosine |
| 0.9729 | late third | cosine |
| 0.9701 | full-attn outputs | cosine |
| 0.9687 | all 33 layers (reference) | cosine |
| 0.9679 | linear-attn outputs | cosine |
| 0.9640 | h32 · final hidden state | frobenius |
| 0.9640 | h32 · final hidden state | euclidean |
| 0.9640 | h32 · final hidden state | bw |
| 0.9630 | late third | bw |
| 0.9630 | late third | frobenius |
| 0.9630 | late third | euclidean |
| 0.9599 | full-attn outputs | bw |
| 0.9598 | full-attn outputs | euclidean |
| 0.9598 | full-attn outputs | frobenius |
| 0.9594 | late third | energy |
| 0.9585 | all 33 layers (reference) | bw |
| 0.9584 | all 33 layers (reference) | euclidean |
| 0.9584 | all 33 layers (reference) | frobenius |
| 0.9582 | late third | mmd |
| 0.9577 | linear-attn outputs | bw |
| 0.9576 | linear-attn outputs | frobenius |
| 0.9576 | linear-attn outputs | euclidean |
| 0.9573 | h32 · final hidden state | energy |
| 0.9571 | middle third | cosine |
| 0.9559 | full-attn outputs | energy |
| 0.9555 | full-attn outputs | mmd |
| 0.9549 | late third · centered | energy |
| 0.9548 | all 33 layers (reference) | mmd |
| 0.9546 | all 33 layers (reference) | energy |
| 0.9542 | linear-attn outputs | mmd |
| 0.9539 | linear-attn outputs | energy |
| 0.9535 | h32 · final hidden state | mmd |
| 0.9524 | h16 · mid-stack (full-attn) | cosine |
| 0.9522 | late third · centered | frobenius |
| 0.9518 | all 33 layers · centered | energy |
| 0.9507 | h1 · first linear-attn | cosine |
| 0.9486 | all 33 layers · centered | frobenius |
| 0.9472 | middle third | bw |
| 0.9470 | middle third | frobenius |
| 0.9470 | middle third | euclidean |
| 0.9463 | late third · centered | cosine |
| 0.9463 | early third | cosine |
| 0.9456 | h4 · first full-attn | cosine |
| 0.9440 | h32 · final hidden state | cka |
| 0.9440 | all 33 layers · centered | cosine |
| 0.9427 | h16 · mid-stack (full-attn) | bw |
| 0.9425 | h16 · mid-stack (full-attn) | frobenius |
| 0.9425 | h16 · mid-stack (full-attn) | euclidean |
| 0.9414 | h1 · first linear-attn | euclidean |
| 0.9414 | h1 · first linear-attn | frobenius |
| 0.9414 | h1 · first linear-attn | bw |
| 0.9409 | middle third | mmd |
| 0.9399 | all 33 layers · centered | mmd |
| 0.9392 | middle third | energy |
| 0.9391 | late third · centered | mmd |
| 0.9364 | h1 · first linear-attn | energy |
| 0.9346 | early third | bw |
| 0.9345 | h1 · first linear-attn | mmd |
| 0.9345 | early third | frobenius |
| 0.9345 | early third | euclidean |
| 0.9341 | h16 · mid-stack (full-attn) | mmd |
| 0.9340 | h4 · first full-attn | frobenius |
| 0.9340 | h4 · first full-attn | euclidean |
| 0.9339 | h4 · first full-attn | bw |
| 0.9330 | h16 · mid-stack (full-attn) | energy |
| 0.9275 | early third | mmd |
| 0.9265 | h4 · first full-attn | energy |
| 0.9263 | h4 · first full-attn | mmd |
| 0.9244 | early third | energy |
| 0.9168 | full-attn outputs | cka |
| 0.9154 | late third | cka |
| 0.9062 | h16 · mid-stack (full-attn) | cka |
| 0.9046 | all 33 layers (reference) | cka |
| 0.8988 | linear-attn outputs | cka |
| 0.8865 | middle third | cka |
| 0.8430 | h4 · first full-attn | cka |
| 0.8248 | early third | cka |
| 0.7231 | h1 · first linear-attn | cka |
| 0.2844 | all 33 layers · centered | cka |
| 0.2224 | late third · centered | cka |
| 0.1104 | late third · centered | bw |
| 0.0883 | all 33 layers · centered | bw |

### structural

| dCor | rung | metric |
|---|---|---|
| 0.9684 | linear-attn · late third | cosine |
| 0.9683 | full-attn · late third | cosine |
| 0.9679 | linear-attn · in_proj_z | cosine |
| 0.9668 | linear-attn · middle third | cosine |
| 0.9661 | linear-attn · qkv,z (d_in 2560) | cosine |
| 0.9659 | full-attn · q_proj gate half | cosine |
| 0.9659 | linear-attn · layer 30 · out_proj | cosine |
| 0.9650 | full-attn · q_proj (whole) | cosine |
| 0.9649 | linear-attn · qkv,z,out | cosine |
| 0.9649 | linear-attn · in_proj_qkv | cosine |
| 0.9645 | all layers · all projections | cosine |
| 0.9643 | full-attn · q,k,v (d_in 2560) | cosine |
| 0.9638 | full-attn · q_proj query half | cosine |
| 0.9624 | full-attn · q,k,v,o | cosine |
| 0.9619 | full-attn · middle third | cosine |
| 0.9601 | full-attn · k_proj | cosine |
| 0.9573 | linear-attn · early third | cosine |
| 0.9563 | linear-attn · in_proj_z | frobenius |
| 0.9563 | linear-attn · late third | frobenius |
| 0.9558 | full-attn · late third | frobenius |
| 0.9554 | full-attn · v_proj | cosine |
| 0.9540 | linear-attn · middle third | frobenius |
| 0.9538 | linear-attn · qkv,z (d_in 2560) | frobenius |
| 0.9536 | linear-attn · out_proj | cosine |
| 0.9534 | full-attn · q_proj gate half | frobenius |
| 0.9529 | linear-attn · layer 30 · out_proj | frobenius |
| 0.9526 | output projections (d_in 4096) | cosine |
| 0.9525 | full-attn · early third | cosine |
| 0.9522 | full-attn · q_proj (whole) | frobenius |
| 0.9521 | linear-attn · in_proj_qkv | frobenius |
| 0.9521 | linear-attn · qkv,z,out | frobenius |
| 0.9515 | all layers · all projections | frobenius |
| 0.9510 | full-attn · q,k,v (d_in 2560) | frobenius |
| 0.9505 | full-attn · q_proj query half | frobenius |
| 0.9505 | linear-attn · layer 16 · out_proj | cosine |
| 0.9485 | full-attn · q,k,v,o | frobenius |
| 0.9485 | full-attn · o_proj | cosine |
| 0.9474 | full-attn · middle third | frobenius |
| 0.9469 | full-attn · layer 15 · o_proj | cosine |
| 0.9454 | full-attn · k_proj | frobenius |
| 0.9451 | linear-attn · layer 30 · out_proj | cka |
| 0.9438 | linear-attn · early third | frobenius |
| 0.9425 | linear-attn · late third | bw |
| 0.9422 | full-attn · layer 3 · o_proj | cosine |
| 0.9387 | full-attn · early third | frobenius |
| 0.9386 | full-attn · v_proj | frobenius |
| 0.9379 | full-attn · layer 3 · o_proj | cka |
| 0.9365 | linear-attn · out_proj | frobenius |
| 0.9364 | linear-attn · qkv,z (d_in 2560) | bw |
| 0.9363 | linear-attn · in_proj_z | bw |
| 0.9352 | output projections (d_in 4096) | frobenius |
| 0.9318 | linear-attn · layer 0 · out_proj | cosine |
| 0.9317 | linear-attn · in_proj_qkv | bw |
| 0.9311 | linear-attn · layer 16 · out_proj | frobenius |
| 0.9306 | linear-attn · middle third | bw |
| 0.9301 | full-attn · o_proj | frobenius |
| 0.9292 | full-attn · layer 31 · o_proj | cosine |
| 0.9274 | full-attn · layer 3 · o_proj | frobenius |
| 0.9269 | full-attn · layer 15 · o_proj | frobenius |
| 0.9232 | full-attn · layer 15 · o_proj | cka |
| 0.9204 | linear-attn · layer 0 · out_proj | cka |
| 0.9146 | linear-attn · layer 0 · out_proj | bw |
| 0.9146 | linear-attn · layer 0 · out_proj | frobenius |
| 0.9126 | linear-attn · early third | bw |
| 0.9090 | full-attn · layer 31 · o_proj | frobenius |
| 0.9075 | full-attn · o_proj | bw |
| 0.9059 | full-attn · v_proj | bw |
| 0.9051 | linear-attn · layer 16 · out_proj | cka |
| 0.8995 | output projections (d_in 4096) | bw |
| 0.8931 | linear-attn · layer 30 · out_proj | bw |
| 0.8923 | linear-attn · out_proj | bw |
| 0.8835 | full-attn · late third | bw |
| 0.8791 | full-attn · q,k,v (d_in 2560) | bw |
| 0.8729 | full-attn · q_proj query half | bw |
| 0.8725 | full-attn · q_proj (whole) | bw |
| 0.8722 | full-attn · k_proj | bw |
| 0.8721 | full-attn · q_proj gate half | bw |
| 0.8601 | full-attn · layer 31 · o_proj | bw |
| 0.8552 | full-attn · layer 3 · o_proj | bw |
| 0.8473 | linear-attn · layer 16 · out_proj | bw |
| 0.8399 | full-attn · layer 31 · o_proj | cka |
| 0.8358 | full-attn · layer 15 · o_proj | bw |
| 0.8354 | full-attn · middle third | bw |
| 0.8306 | full-attn · early third | bw |

### dataset_embedding

| dCor | rung | metric |
|---|---|---|
| 0.9800 | dataset text · mean · n1000_s00 | frobenius |
| 0.9797 | dataset text · mean · n1000_s00 | euclidean |
| 0.9371 | dataset text · mean · n1000_s00 | cosine |
| 0.9259 | dataset text · mean · centered | frobenius |
| 0.9127 | dataset text · mean · centered | cosine |

