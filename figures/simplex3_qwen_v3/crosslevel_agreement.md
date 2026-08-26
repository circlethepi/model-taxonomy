| level | dCor vs ground truth | Procrustes residual (lower=better) | rung | metric |
|---|---|---|---|---|
| behavioral | 0.8164 | 0.1973 | R=16 · per query | cka |
| functional | 0.9746 | **0.0075** | h32 · final hidden state | cosine |
| structural | 0.9684 | 0.0078 | linear-attn · late third | cosine |
| dataset_embedding | 0.9800 | 0.0153 | dataset text · mean · n1000_s00 | frobenius |


## Every rung, per level

### behavioral

| dCor | Procrustes residual (lower=better) | rung | metric |
|---|---|---|---|
| 0.8164 | **0.1973** | R=16 · per query | cka |
| 0.7942 | **0.3010** | R=16 · per query | frobenius |
| 0.7942 | **0.3010** | R=16 · per query | euclidean |
| 0.7895 | 0.3210 | R=16 · per query | cosine |
| 0.7462 | 0.7458 | greedy · per generation | cosine |
| 0.7437 | 0.9057 | greedy · per generation | frobenius |
| 0.7437 | 0.9057 | greedy · per generation | euclidean |
| 0.7392 | 0.6144 | R=16 · per query | bw |
| 0.7357 | 0.8465 | R=16 · per generation · whitened | cosine |
| 0.7292 | 0.8688 | R=16 · per generation · whitened | frobenius |
| 0.7234 | 0.8301 | R=16 · per generation · centered | cosine |
| 0.7218 | 0.9422 | R=16 · per generation · centered | frobenius |
| 0.7200 | 0.8979 | R=16 · per generation · whitened | euclidean |
| 0.7070 | 0.8685 | R=16 · per generation | frobenius |
| 0.7070 | 0.8685 | R=16 · per generation | euclidean |
| 0.7039 | 0.8850 | greedy · per generation | bw |
| 0.7017 | 0.7137 | R=16 · per generation | cosine |
| 0.6729 | 0.6339 | R=16 · per generation | cka |
| 0.6456 | 0.8564 | R=16 · per generation · whitened | bw |
| 0.6450 | 0.8282 | greedy · per generation | cka |
| 0.6077 | 0.6677 | R=16 · per generation | bw |
| 0.4751 | 0.5235 | R=16 · per generation · whitened | energy |
| 0.4674 | 0.5235 | R=16 · per generation · whitened | mmd |
| 0.4425 | 0.7875 | R=16 · per generation · centered | bw |
| 0.2915 | 0.6080 | greedy · model mean | frobenius |
| 0.2915 | 0.6080 | greedy · model mean | euclidean |
| 0.2718 | 0.6482 | greedy · model mean | cosine |
| 0.2685 | 0.8316 | R=16 · model mean · centered | cosine |
| 0.2588 | 0.8726 | R=16 · model mean · centered | frobenius |
| 0.2206 | 0.8119 | R=16 · per generation | energy |
| 0.2174 | 0.7849 | R=16 · per generation · centered | energy |
| 0.2173 | 0.7800 | R=16 · per generation · centered | mmd |
| 0.2156 | 0.8133 | R=16 · per generation | mmd |
| 0.2080 | 0.8429 | R=16 · model mean | cosine |
| 0.2064 | 0.8242 | R=16 · model mean | euclidean |
| 0.2064 | 0.8242 | R=16 · model mean | frobenius |
| 0.1961 | 0.8910 | R=16 · per query | mmd |
| 0.1797 | 0.8626 | greedy · per generation | mmd |
| 0.1774 | 0.8634 | greedy · per generation | energy |
| 0.1619 | 0.9039 | R=16 · per query | energy |
| -0.5479 | 0.9523 | R=16 · per generation · centered | cka |
| -0.6741 | 0.8773 | R=16 · per generation · whitened | cka |

### functional

| dCor | Procrustes residual (lower=better) | rung | metric |
|---|---|---|---|
| 0.9746 | **0.0075** | h32 · final hidden state | cosine |
| 0.9729 | **0.0098** | late third | cosine |
| 0.9701 | **0.0107** | full-attn outputs | cosine |
| 0.9687 | 0.0118 | all 33 layers (reference) | cosine |
| 0.9679 | 0.0120 | linear-attn outputs | cosine |
| 0.9640 | 0.0240 | h32 · final hidden state | frobenius |
| 0.9640 | 0.0240 | h32 · final hidden state | euclidean |
| 0.9640 | 0.0237 | h32 · final hidden state | bw |
| 0.9630 | 0.0272 | late third | bw |
| 0.9630 | 0.0274 | late third | frobenius |
| 0.9630 | 0.0274 | late third | euclidean |
| 0.9599 | 0.0296 | full-attn outputs | bw |
| 0.9598 | 0.0297 | full-attn outputs | euclidean |
| 0.9598 | 0.0297 | full-attn outputs | frobenius |
| 0.9594 | 0.0210 | late third | energy |
| 0.9585 | 0.0307 | all 33 layers (reference) | bw |
| 0.9584 | 0.0308 | all 33 layers (reference) | euclidean |
| 0.9584 | 0.0308 | all 33 layers (reference) | frobenius |
| 0.9582 | 0.0226 | late third | mmd |
| 0.9577 | 0.0312 | linear-attn outputs | bw |
| 0.9576 | 0.0313 | linear-attn outputs | frobenius |
| 0.9576 | 0.0313 | linear-attn outputs | euclidean |
| 0.9573 | 0.0200 | h32 · final hidden state | energy |
| 0.9571 | 0.0183 | middle third | cosine |
| 0.9559 | 0.0221 | full-attn outputs | energy |
| 0.9555 | 0.0244 | full-attn outputs | mmd |
| 0.9549 | 0.0384 | late third · centered | energy |
| 0.9548 | 0.0242 | all 33 layers (reference) | mmd |
| 0.9546 | 0.0222 | all 33 layers (reference) | energy |
| 0.9542 | 0.0246 | linear-attn outputs | mmd |
| 0.9539 | 0.0224 | linear-attn outputs | energy |
| 0.9535 | 0.0223 | h32 · final hidden state | mmd |
| 0.9524 | 0.0206 | h16 · mid-stack (full-attn) | cosine |
| 0.9522 | 0.0970 | late third · centered | frobenius |
| 0.9518 | 0.0417 | all 33 layers · centered | energy |
| 0.9507 | 0.0269 | h1 · first linear-attn | cosine |
| 0.9486 | 0.0959 | all 33 layers · centered | frobenius |
| 0.9472 | 0.0381 | middle third | bw |
| 0.9470 | 0.0381 | middle third | frobenius |
| 0.9470 | 0.0381 | middle third | euclidean |
| 0.9463 | 0.0810 | late third · centered | cosine |
| 0.9463 | 0.0177 | early third | cosine |
| 0.9456 | 0.0211 | h4 · first full-attn | cosine |
| 0.9440 | 0.0437 | h32 · final hidden state | cka |
| 0.9440 | 0.0753 | all 33 layers · centered | cosine |
| 0.9427 | 0.0657 | h16 · mid-stack (full-attn) | bw |
| 0.9425 | 0.0659 | h16 · mid-stack (full-attn) | frobenius |
| 0.9425 | 0.0659 | h16 · mid-stack (full-attn) | euclidean |
| 0.9414 | 0.1917 | h1 · first linear-attn | euclidean |
| 0.9414 | 0.1917 | h1 · first linear-attn | frobenius |
| 0.9414 | 0.1918 | h1 · first linear-attn | bw |
| 0.9409 | 0.0321 | middle third | mmd |
| 0.9399 | 0.1167 | all 33 layers · centered | mmd |
| 0.9392 | 0.0298 | middle third | energy |
| 0.9391 | 0.1476 | late third · centered | mmd |
| 0.9364 | 0.2106 | h1 · first linear-attn | energy |
| 0.9346 | 0.0386 | early third | bw |
| 0.9345 | 0.0486 | h1 · first linear-attn | mmd |
| 0.9345 | 0.0386 | early third | frobenius |
| 0.9345 | 0.0386 | early third | euclidean |
| 0.9341 | 0.0384 | h16 · mid-stack (full-attn) | mmd |
| 0.9340 | 0.0484 | h4 · first full-attn | frobenius |
| 0.9340 | 0.0484 | h4 · first full-attn | euclidean |
| 0.9339 | 0.0488 | h4 · first full-attn | bw |
| 0.9330 | 0.0361 | h16 · mid-stack (full-attn) | energy |
| 0.9275 | 0.0334 | early third | mmd |
| 0.9265 | 0.0938 | h4 · first full-attn | energy |
| 0.9263 | 0.1499 | h4 · first full-attn | mmd |
| 0.9244 | 0.0300 | early third | energy |
| 0.9168 | 0.0386 | full-attn outputs | cka |
| 0.9154 | 0.0411 | late third | cka |
| 0.9062 | 0.0506 | h16 · mid-stack (full-attn) | cka |
| 0.9046 | 0.0407 | all 33 layers (reference) | cka |
| 0.8988 | 0.0445 | linear-attn outputs | cka |
| 0.8865 | 0.0500 | middle third | cka |
| 0.8430 | 0.0875 | h4 · first full-attn | cka |
| 0.8248 | 0.3238 | early third | cka |
| 0.7231 | 0.2272 | h1 · first linear-attn | cka |
| 0.2844 | 0.9651 | all 33 layers · centered | cka |
| 0.2224 | 0.9396 | late third · centered | cka |
| 0.1104 | 0.7401 | late third · centered | bw |
| 0.0883 | 0.7516 | all 33 layers · centered | bw |

### structural

| dCor | Procrustes residual (lower=better) | rung | metric |
|---|---|---|---|
| 0.9684 | **0.0078** | linear-attn · late third | cosine |
| 0.9683 | **0.0082** | full-attn · late third | cosine |
| 0.9679 | 0.0093 | linear-attn · in_proj_z | cosine |
| 0.9668 | **0.0092** | linear-attn · middle third | cosine |
| 0.9661 | 0.0094 | linear-attn · qkv,z (d_in 2560) | cosine |
| 0.9659 | 0.0097 | full-attn · q_proj gate half | cosine |
| 0.9659 | 0.0093 | linear-attn · layer 30 · out_proj | cosine |
| 0.9650 | 0.0099 | full-attn · q_proj (whole) | cosine |
| 0.9649 | 0.0096 | linear-attn · qkv,z,out | cosine |
| 0.9649 | 0.0097 | linear-attn · in_proj_qkv | cosine |
| 0.9645 | 0.0097 | all layers · all projections | cosine |
| 0.9643 | 0.0100 | full-attn · q,k,v (d_in 2560) | cosine |
| 0.9638 | 0.0104 | full-attn · q_proj query half | cosine |
| 0.9624 | 0.0104 | full-attn · q,k,v,o | cosine |
| 0.9619 | 0.0105 | full-attn · middle third | cosine |
| 0.9601 | 0.0109 | full-attn · k_proj | cosine |
| 0.9573 | 0.0137 | linear-attn · early third | cosine |
| 0.9563 | 0.0278 | linear-attn · in_proj_z | frobenius |
| 0.9563 | 0.0262 | linear-attn · late third | frobenius |
| 0.9558 | 0.0281 | full-attn · late third | frobenius |
| 0.9554 | 0.0118 | full-attn · v_proj | cosine |
| 0.9540 | 0.0289 | linear-attn · middle third | frobenius |
| 0.9538 | 0.0287 | linear-attn · qkv,z (d_in 2560) | frobenius |
| 0.9536 | 0.0117 | linear-attn · out_proj | cosine |
| 0.9534 | 0.0311 | full-attn · q_proj gate half | frobenius |
| 0.9529 | 0.0289 | linear-attn · layer 30 · out_proj | frobenius |
| 0.9526 | 0.0121 | output projections (d_in 4096) | cosine |
| 0.9525 | 0.0150 | full-attn · early third | cosine |
| 0.9522 | 0.0316 | full-attn · q_proj (whole) | frobenius |
| 0.9521 | 0.0293 | linear-attn · in_proj_qkv | frobenius |
| 0.9521 | 0.0293 | linear-attn · qkv,z,out | frobenius |
| 0.9515 | 0.0299 | all layers · all projections | frobenius |
| 0.9510 | 0.0318 | full-attn · q,k,v (d_in 2560) | frobenius |
| 0.9505 | 0.0323 | full-attn · q_proj query half | frobenius |
| 0.9505 | 0.0101 | linear-attn · layer 16 · out_proj | cosine |
| 0.9485 | 0.0324 | full-attn · q,k,v,o | frobenius |
| 0.9485 | 0.0134 | full-attn · o_proj | cosine |
| 0.9474 | 0.0338 | full-attn · middle third | frobenius |
| 0.9469 | 0.0152 | full-attn · layer 15 · o_proj | cosine |
| 0.9454 | 0.0324 | full-attn · k_proj | frobenius |
| 0.9451 | 0.0168 | linear-attn · layer 30 · out_proj | cka |
| 0.9438 | 0.0351 | linear-attn · early third | frobenius |
| 0.9425 | 0.0271 | linear-attn · late third | bw |
| 0.9422 | 0.0198 | full-attn · layer 3 · o_proj | cosine |
| 0.9387 | 0.0577 | full-attn · early third | frobenius |
| 0.9386 | 0.0382 | full-attn · v_proj | frobenius |
| 0.9379 | 0.0184 | full-attn · layer 3 · o_proj | cka |
| 0.9365 | 0.0349 | linear-attn · out_proj | frobenius |
| 0.9364 | 0.0310 | linear-attn · qkv,z (d_in 2560) | bw |
| 0.9363 | 0.0294 | linear-attn · in_proj_z | bw |
| 0.9352 | 0.0341 | output projections (d_in 4096) | frobenius |
| 0.9318 | 0.0205 | linear-attn · layer 0 · out_proj | cosine |
| 0.9317 | 0.0335 | linear-attn · in_proj_qkv | bw |
| 0.9311 | 0.0570 | linear-attn · layer 16 · out_proj | frobenius |
| 0.9306 | 0.0336 | linear-attn · middle third | bw |
| 0.9301 | 0.0357 | full-attn · o_proj | frobenius |
| 0.9292 | 0.0144 | full-attn · layer 31 · o_proj | cosine |
| 0.9274 | 0.0467 | full-attn · layer 3 · o_proj | frobenius |
| 0.9269 | 0.0956 | full-attn · layer 15 · o_proj | frobenius |
| 0.9232 | 0.0210 | full-attn · layer 15 · o_proj | cka |
| 0.9204 | 0.0242 | linear-attn · layer 0 · out_proj | cka |
| 0.9146 | 0.0714 | linear-attn · layer 0 · out_proj | bw |
| 0.9146 | 0.2468 | linear-attn · layer 0 · out_proj | frobenius |
| 0.9126 | 0.2195 | linear-attn · early third | bw |
| 0.9090 | 0.1759 | full-attn · layer 31 · o_proj | frobenius |
| 0.9075 | 0.1863 | full-attn · o_proj | bw |
| 0.9059 | 0.1880 | full-attn · v_proj | bw |
| 0.9051 | 0.0193 | linear-attn · layer 16 · out_proj | cka |
| 0.8995 | 0.0751 | output projections (d_in 4096) | bw |
| 0.8931 | 0.0771 | linear-attn · layer 30 · out_proj | bw |
| 0.8923 | 0.0754 | linear-attn · out_proj | bw |
| 0.8835 | 0.1061 | full-attn · late third | bw |
| 0.8791 | 0.1360 | full-attn · q,k,v (d_in 2560) | bw |
| 0.8729 | 0.1362 | full-attn · q_proj query half | bw |
| 0.8725 | 0.1011 | full-attn · q_proj (whole) | bw |
| 0.8722 | 0.2319 | full-attn · k_proj | bw |
| 0.8721 | 0.1016 | full-attn · q_proj gate half | bw |
| 0.8601 | 0.5737 | full-attn · layer 31 · o_proj | bw |
| 0.8552 | 0.2296 | full-attn · layer 3 · o_proj | bw |
| 0.8473 | 0.5746 | linear-attn · layer 16 · out_proj | bw |
| 0.8399 | 0.1721 | full-attn · layer 31 · o_proj | cka |
| 0.8358 | 0.2521 | full-attn · layer 15 · o_proj | bw |
| 0.8354 | 0.2206 | full-attn · middle third | bw |
| 0.8306 | 0.2242 | full-attn · early third | bw |

### dataset_embedding

| dCor | Procrustes residual (lower=better) | rung | metric |
|---|---|---|---|
| 0.9800 | **0.0153** | dataset text · mean · n1000_s00 | frobenius |
| 0.9797 | **0.0152** | dataset text · mean · n1000_s00 | euclidean |
| 0.9371 | **0.0760** | dataset text · mean · n1000_s00 | cosine |
| 0.9259 | 0.1257 | dataset text · mean · centered | frobenius |
| 0.9127 | 0.1383 | dataset text · mean · centered | cosine |

