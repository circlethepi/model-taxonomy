# Training-draw-size sweep (`nsweep`) — results

**nsweep** — a sweep over `n_samples`, the size of the *training* draw, holding
the mixture grid, the seeds and every optimizer setting fixed. **Rung** — one
value of `n_samples`; note this is not the `rung` that `docs/terminology.md`
retired in favour of `surrogate`.

90 collections: the 16-point 25% simplex over three yahoo topic groups, trained
at nine draw sizes × ten dataset seeds on OLMo-2-0425-1B-Instruct. Scored by
`sweep_nsweep.py` (720 rows, 876 s), plotted by `make_figures.py`.

## Median dCor* against the requested simplex

| perspective | 10 | 20 | 50 | 100 | 200 | 500 | 1000 | 2000 | 5000 |
|---|---|---|---|---|---|---|---|---|---|
| dataset embedding | 0.945 | 0.969 | 0.977 | 0.980 | 0.974 | 0.975 | 0.971 | 0.972 | 0.970 |
| structural (a) all·o | 0.785 | 0.904 | 0.942 | 0.957 | 0.963 | 0.970 | 0.971 | 0.974 | 0.970 |
| structural (b) all·qkvo | 0.790 | 0.908 | 0.947 | 0.960 | 0.966 | 0.972 | 0.970 | 0.974 | 0.967 |
| structural (c) last·o | 0.811 | 0.912 | 0.944 | 0.926 | 0.928 | 0.935 | 0.944 | 0.956 | 0.963 |
| functional (a) all states | 0.772 | 0.848 | 0.920 | 0.953 | 0.970 | 0.979 | 0.979 | 0.979 | 0.980 |
| functional (b) final state | 0.797 | 0.855 | 0.918 | 0.948 | 0.969 | 0.977 | 0.976 | 0.978 | 0.978 |
| behavioral R=16 | 0.146 | 0.273 | 0.322 | 0.388 | 0.549 | 0.464 | 0.560 | 0.556 | 0.589 |
| behavioral greedy | 0.491 | 0.586 | 0.597 | 0.815 | 0.845 | 0.859 | 0.832 | 0.846 | 0.827 |

## What the figure says

**The dataset level barely moves.** 0.945 at N=10 and 0.97–0.98 everywhere
above it — the mean embedding of a draw is already an excellent estimate of the
mixture at ten rows. It is the *only* level with no meaningful N-dependence, and
it sets the ceiling the other levels are climbing toward.

**Structural and functional saturate by N≈100–500** at dCor ≈ 0.97 and
disparity ≈ 0.01, then are flat for the remaining 1.5 decades. Buying data past
N≈500 buys nothing at either level.

**The all-layer/final-state contrast is a null result at both levels.**
functional (a) and (b) agree to 0.002 everywhere. Structural (c), the last
layer alone, is the one that differs — and it is *worse* in the middle of the
range (0.926–0.944 at N=100–1000 against 0.957–0.974 for the two all-layer
rows), converging only at N=5000. All-layer is the better default.

**Behavioral R=16 never recovers.** It reaches only 0.589 dCor and 0.326
disparity at N=5000, still far from the other levels' N=20 performance, and its
climb is non-monotonic (0.549 at N=200, dipping to 0.464 at N=500).

**Greedy decoding is the fix, and the gap is large.** The deterministic control
reaches 0.815 by N=100 and holds 0.83–0.86 thereafter — 0.24–0.39 dCor above
the sampled row at every rung. The two differ by *nothing but decoding*: same
draw, same queries, same embedder, same pooling. So the sampled behavioral row
is sampling-noise-limited, not perspective-limited.

## Q4 is settled: the discretisation never mattered

The experiment scored every slice twice, against the mixture its name requests
and against the one largest-remainder allocation actually realized, because at
small N those differ deterministically and would otherwise be charged to the
taxonomy level.

They agree. The median |gap| in dCor* is at most **0.028** at N=10, ≤0.006 from
N=100, and identically 0 from N=1000. Nowhere does it approach the size of the
N-effect being measured. **The requested/realized distinction can be dropped
from future figures** — the second scoring is nearly free, but it answers a
question whose answer is now known.

`check_realized_truth` passed at every rung; the worst weight deviation is
exactly 2/3 of the `1/N` bound at each (0.0667 at N=10 → 0.000133 at N=5000).

## Open

**Greedy disparity is non-monotonic in a way its dCor is not:** 0.549 at N=10,
down to 0.083 at N=200, back up to 0.295 at N=5000, while dCor stays flat at
~0.83 across that whole range. Two estimators over the same matrices disagreeing
about direction is worth a look before this panel is used for a claim; the MDS
fit is the likeliest suspect.

**N=10000 was declined** and is not on any curve. Every rung here is 160
adapters; that one holds 4.
