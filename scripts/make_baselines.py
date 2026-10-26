"""Tabulate the uninformed baselines that figures draw as a band.

An **uninformed baseline** is the reference level a taxonomy that learned
nothing would score; the **uninformed band** is its 5-95 interval.  This script
precomputes both, per score and per collection size, into

    results/baselines/constants.json

which every figure driver reads through
:func:`src.analysis.baselines.uninformed_band`.  That path is **gitignored**
(``.gitignore`` ignores ``results`` whole), so the file is a generated artifact
like every other one under ``results/``: a fresh clone has to run this script
before any figure can draw a band, and every reader says so by name when the
file is missing.

Two generators, both the **structure null** -- replace the taxonomy with a
structureless configuration and keep the truth:

    gaussian    iid N(0, I)          -- no geometry at all
    dirichlet   Dirichlet(1) weights -- *a* valid recipe, just not this one

Usage:
    python scripts/make_baselines.py
    python scripts/make_baselines.py --k 3 --n-grid 5,10,20,50,100,200,500
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.baselines import (  # noqa: E402
    GENERATORS,
    band_quantiles,
    dcor_band_from_c,
    dcor_null_direct,
    disparity_null_analytic,
    disparity_null_direct,
    fit_dcor_c,
    table_key,
    truth_singular_values,
)
from src.analysis.ground_truth import simplex_vertices  # noqa: E402

REPO = Path(__file__).parent.parent
MEMBERS = REPO / "figures/simplex_collection_size/group_members.csv"
OUT = REPO / "results/baselines/constants.json"

#: ``026g1`` -> 26% of vertex 1.  The mixture head of an adapter directory name
#: is the only record of a group's ground truth that survives in
#: ``group_members.csv``, so the weights are read back out of it.
_WEIGHT_RE = re.compile(r"(\d+)g(\d+)")


def parse_weights(mixture: str, k: int) -> np.ndarray | None:
    """Mixture weights out of ``yahoo_026g1_032g2_042g3``, or None."""
    pairs = _WEIGHT_RE.findall(mixture)
    if len(pairs) != k:
        return None
    w = np.zeros(k, dtype=np.float64)
    for value, index in pairs:
        j = int(index) - 1
        if not 0 <= j < k:
            return None
        w[j] = int(value) / 100.0
    if abs(w.sum() - 1.0) > 1e-9:
        return None
    return w


def lambda_from_members(path: Path, k: int) -> dict[int, np.ndarray]:
    """Pooled ``L`` per n, averaged over that n's replicate groups.

    The exact truth configuration of every ``(n, replicate)`` group is
    recoverable -- ``group_members.csv`` names its members and the names carry
    their mixture weights -- so this is the real truth shape the sweep scored
    against, not a stand-in.

    Pooling over replicates is the approximation the plan accepted: one ``L``
    per n rather than one per group.  ``L`` varies most across replicates at
    small n, so that is where the band is softest, and ``lambda_pooled`` is
    written into the table so the approximation is auditable rather than
    implicit.
    """
    per_n: dict[int, list[np.ndarray]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            n = int(row["n"])
            weights = [parse_weights(m, k) for m in row["members"].split()]
            if any(w is None for w in weights) or len(weights) != n:
                continue
            coords = np.vstack(weights) @ simplex_vertices(k)
            try:
                per_n.setdefault(n, []).append(truth_singular_values(coords))
            except ValueError:
                continue  # a degenerate group contributes nothing
    return {n: np.mean(np.vstack(v), axis=0) for n, v in per_n.items() if v}


def lambda_synthetic(n: int, k: int, replicates: int, rng) -> np.ndarray:
    """Pooled ``L`` from Dirichlet(1) truths, for an n or K with no members file."""
    out = []
    for _ in range(replicates):
        coords = rng.dirichlet(np.ones(k), size=n) @ simplex_vertices(k)
        out.append(truth_singular_values(coords))
    return np.mean(np.vstack(out), axis=0)


def truth_with_lambda(n: int, lam: np.ndarray, seed: int) -> np.ndarray:
    """An ``(n, d)`` centred, unit-norm configuration whose singular values are *lam*.

    The disparity's null depends on the truth **only** through ``L`` (the
    derivation reduces it to ``Z = U^T X``, and ``U`` washes out), so any
    configuration with the pooled ``L`` samples the same law -- and unlike a real
    group's coordinates it is available at every n.

    Built as ``U @ diag(lam)`` with ``U``'s columns orthonormal and orthogonal to
    ``1``, so the result is already centred and already unit Frobenius norm.
    Padding a diagonal with zero rows and then centring would *not* work: the
    centring shifts every row and changes the singular values it was supposed to
    preserve.
    """
    d = len(lam)
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(np.hstack([np.ones((n, 1)), rng.standard_normal((n, d))]))
    return basis[:, 1: d + 1] * lam[None, :]


def disparity_entry(ns, lam_by_n, source_by_n, generator, k, n_mc, seed):
    """Band quantiles per n for the Procrustes disparity.

    ``gaussian`` uses the exact analytic law -- all the n-dependence is one
    chi-square draw, so it is both cheap and correct at every n.  ``dirichlet``
    is not Gaussian, so it runs the full scoring path instead; the two agree to
    within Monte Carlo noise at every n tested, which is worth knowing but is
    not assumed here.
    """
    q05, q50, q95 = [], [], []
    for n in ns:
        lam = lam_by_n[n]
        if generator == "gaussian":
            sample = disparity_null_analytic(n, lam, n_mc=n_mc, seed=seed + n)
        else:
            sample = disparity_null_direct(
                n, truth_with_lambda(n, lam, seed + n), generator=generator,
                n_mc=max(500, n_mc // 20), seed=seed + n,
            )
        lo, mid, hi = band_quantiles(sample)
        q05.append(lo)
        q50.append(mid)
        q95.append(hi)
    return {"q05": q05, "q50": q50, "q95": q95,
            "lambda_source": [source_by_n[n] for n in ns]}


def _dcor_band(n, k, generator, n_mc, seed):
    """Sampled ``(lo, mid, hi)`` of the dCor* structure null at one n.

    The centre is the sampled **median**, not a hard 0.  Zero is the null's
    *mean* -- that is what the bias correction buys -- but dCor* is a ratio with
    a random denominator and so is right-skewed, putting the median below it:
    -0.027 at n=10, -0.0014 at n=200.  A centre line is a median, so it gets the
    median.
    """
    sample = dcor_null_direct(
        n, _reference_truth(n, k), k=k, generator=generator,
        n_mc=n_mc, seed=seed + n,
    )
    return band_quantiles(sample)


def dcor_entry(ns, lam_by_n, source_by_n, generator, k, n_mc, seed):
    """Band per n for dCor*, from the fitted ``c * n**-p`` law.

    Both edges *and* the centre are sampled.  The plan expected a hard-zero
    centre -- 0 is what the bias correction guarantees -- but that guarantee is
    about the **mean**, and a band's centre line is a median.  dCor* is
    right-skewed, so the two differ: see :func:`_dcor_band`.

    The width is **sampled at every tabulated n**, not extrapolated from a fit.
    An earlier revision fitted ``c * n**-p`` on the interior of the grid and
    read the ends off the law; the ends came out 23% wrong, at both ``k`` and
    under both generators.  The power law is a good description of the *shape*
    -- see :func:`~src.analysis.baselines.fit_dcor_c` -- but not accurate enough
    to extrapolate a band edge from, and the grid is seven points, so sampling
    all of them costs the same as sampling five and guessing two.

    ``(c, p)`` is still fitted and stored, from the same samples, as the
    diagnostic that records the exponent and lets a caller off the grid
    extrapolate knowingly.
    """
    draws = max(1_000, n_mc // 10)
    sampled = [n for n in ns if n >= 4]
    triples = {n: _dcor_band(n, k, generator, draws, seed) for n in sampled}
    widths = [0.5 * (hi - lo) for lo, _, hi in (triples[n] for n in sampled)]
    c, p = fit_dcor_c(sampled, widths)

    q05, q50, q95 = [], [], []
    for n in ns:
        if n in triples:
            lo, mid, hi = triples[n]
        else:
            # n < 4 cannot be U-centred at all; the law is the only thing left,
            # and off the sampled grid there is no median to report.
            lo, mid, hi = dcor_band_from_c(n, c, p)
        q05.append(lo)
        q50.append(mid)
        q95.append(hi)
    return {"q05": q05, "q50": q50, "q95": q95,
            "dcor_c": c, "dcor_p": p,
            "dcor_sampled_n": sampled, "dcor_sampled_width": widths,
            "dcor_fit_rel_error": [abs(c * n ** -p - w) / w
                                   for n, w in zip(sampled, widths)],
            "lambda_source": [source_by_n[n] for n in ns]}


def _reference_truth(n: int, k: int, seed: int = 0) -> np.ndarray:
    """A Dirichlet(1) ground-truth distance matrix at *n*, for the dCor null.

    dCor is matrix-level, so unlike the disparity its null does not reduce to
    the truth's singular values -- it depends on the truth's whole distance
    profile.  One fixed reference truth per (n, k) is used, drawn from the same
    Dirichlet(1) the real recipes are spread over.
    """
    from scipy.spatial.distance import pdist, squareform

    rng = np.random.default_rng(seed * 1000 + n)
    coords = rng.dirichlet(np.ones(k), size=n) @ simplex_vertices(k)
    return squareform(pdist(coords))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, action="append", dest="ks", default=None,
                    help="simplex vertex counts to tabulate (default: 3 and 4)")
    ap.add_argument("--n-grid", default="5,10,20,50,100,200,500",
                    help="collection sizes to tabulate")
    ap.add_argument("--members", default=str(MEMBERS),
                    help="group_members.csv, the source of the pooled truth shape")
    ap.add_argument("--n-mc", type=int, default=20_000,
                    help="Monte Carlo draws for the analytic disparity law")
    ap.add_argument("--lambda-replicates", type=int, default=100,
                    help="synthetic truths pooled when the members file has no n")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    ks = args.ks or [3, 4]
    ns = [int(x) for x in args.n_grid.split(",")]
    rng = np.random.default_rng(args.seed)

    members = Path(args.members)
    entries: dict[str, dict] = {}
    t0 = time.time()

    for k in ks:
        d = k - 1
        measured = lambda_from_members(members, k) if members.exists() else {}
        lam_by_n, source_by_n = {}, {}
        for n in ns:
            if n in measured:
                lam_by_n[n], source_by_n[n] = measured[n], "members"
            else:
                lam_by_n[n] = lambda_synthetic(n, k, args.lambda_replicates, rng)
                source_by_n[n] = "dirichlet"

        for generator in sorted(GENERATORS):
            for score, build in (("disparity", disparity_entry),
                                 ("dcor", dcor_entry)):
                # The disparity's null is undefined where the configuration
                # spans the truth exactly: (n-1-d)*d degrees of freedom, which
                # is <= 0 at n <= 1+d.  Drop those rungs rather than fake them.
                keep = [n for n in ns
                        if score != "disparity" or (n - 1 - d) * d > 0]
                if not keep:
                    continue
                entry = build(keep, lam_by_n, source_by_n, generator, k,
                              args.n_mc, args.seed)
                entry["n"] = keep
                entry["lambda_pooled"] = [
                    [float(x) for x in lam_by_n[n]] for n in keep
                ]
                entries[table_key(k, d, generator, score)] = entry
                extra = ""
                if "dcor_p" in entry:
                    extra = (f"  c={entry['dcor_c']:.3f} p={entry['dcor_p']:.3f}"
                             f" (law would miss by up to "
                             f"{max(entry['dcor_fit_rel_error']):.0%})")
                print(f"  {table_key(k, d, generator, score):24s} "
                      f"n={keep[0]}..{keep[-1]}  "
                      f"[{entry['q05'][0]:+.3f},{entry['q95'][0]:+.3f}]"
                      f" -> [{entry['q05'][-1]:+.3f},{entry['q95'][-1]:+.3f}]"
                      f"{extra}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_by": "scripts/make_baselines.py",
        "seed": args.seed,
        "n_mc": args.n_mc,
        "band_quantiles": [5.0, 95.0],
        "members": str(members) if members.exists() else None,
        "entries": entries,
    }
    with out.open("w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"\nwrote {out}  ({len(entries)} entries, {time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
