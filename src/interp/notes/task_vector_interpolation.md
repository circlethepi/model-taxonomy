# Task Vector corner-interpolation

## What we want to accomplish

We assume 3 dataset components. Let' s assume we have 3 models trained on the separate dataset components. For each model, let's fix a layer $\ell$.
Then, using the respective dataset of $n$ samples, let's run a forward pass on each sample, and extract final token position $t$ on each sample, giving us $X \in \mathbb{R}^{n \times d}$. –

From this, we can compute 3 task vectors. Let the datasets be $a, b, c$. 

$$v_a = \frac{1}{n} \sum_{i=1}^n \mathbf{h}_{X_a}[i,:]$$

Then, we can compute the shared baseline mean (global mean):

$$\hat{v} = \frac{1}{3}(v_a + v_b + v_c)$$

Our new task vectors are the following:
1. $$\hat{v}_a = v_a - \hat{v}$$
2. $$\hat{v}_b = v_b - \hat{v}$$
3. $$\hat{v}_c = v_c - \hat{v}$$


Then, we can get our mixture hidden states. We can get the same forward pass final token vectors on the mixed dataset. 


Then, for each sample, compute
$$\hat{h}_\text{mix} = h_\text{mix} - \hat{v}$$

Then we can compute least squares to get coefficients for each sample 
