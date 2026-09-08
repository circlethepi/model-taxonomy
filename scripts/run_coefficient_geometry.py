"""Score fitted interpolation coefficients against the ground-truth simplex.

`scripts/run_lora_interpolation.py` asks how far each adapter's fitted
coefficients are from its own true recipe.  This asks whether the fitted
coefficients, taken together, reproduce the *shape* the data recipes define —
Procrustes disparity and distance correlation against the ideal simplex, the
same two statistics `src/analysis/comparison.py` reports for every other
taxonomy level.

Reads the arrays an interpolation run already wrote to `fits/fits.safetensors`,
so it needs no adapters, no Gram and no GPU: a few seconds on a login node, and
re-runnable against past results.

Usage:
    python scripts/run_coefficient_geometry.py --run-dir results/lora_interpolation

    # only the genuine mixtures, dropping the corners that fit themselves
    python scripts/run_coefficient_geometry.py --run-dir results/lora_interpolation \\
        --mixtures-only --output-dir results/lora_interpolation/mixtures_only

Note: the default MDS path needs a scikit-learn new enough for MDSGeometry's
`metric_mds=` keyword.  Use the project environment (`conda activate taxonomy`).
`--geometry pca` is classical MDS in pure numpy and has no such requirement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.interp.geometry import COEFF_KINDS, compare_from_fits
from src.interp.report import load_fits


def main(cfg: dict, make_figures: bool = True):
    """Run the comparison and write the report. Returns the report object."""
    run_dir = Path(cfg["run_dir"])
    output_dir = Path(cfg.get("output_dir") or run_dir)

    print(f"Reading fits from {run_dir}")
    fits = load_fits(run_dir)
    n, k = len(fits["targets"]), len(fits["vertices"])
    print(f"  {n} adapter(s), {sum(fits['is_corner'])} corner(s), vertices {fits['vertices']}")

    report = compare_from_fits(
        fits,
        mixtures_only=cfg.get("mixtures_only", False),
        kinds=cfg.get("kinds") or COEFF_KINDS,
        n_components=cfg.get("n_components"),
        method=cfg.get("geometry", "mds"),
        n_permutations=cfg.get("n_permutations", 9999),
        random_state=cfg.get("random_state", 0),
        run_protest=not cfg.get("no_protest", False),
    )
    print(f"  scored {len(report.model_ids)} adapter(s) in "
          f"{report.config['n_components']}d via {report.config['method']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report.save(output_dir, figures=make_figures)
    print(f"\n{report.to_markdown(figures=make_figures)}")

    if make_figures:
        from src.interp.plots import make_geometry_figures

        paths = make_geometry_figures(report, output_dir / "figures")
        print(f"  {len(paths)} figure(s) -> {output_dir / 'figures'}")

    print(f"Wrote {output_dir}")
    return report


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-dir", required=True,
                   help="an interpolation run directory containing fits/fits.safetensors")
    p.add_argument("--output-dir", default=None, help="default: --run-dir")
    p.add_argument("--kinds", nargs="+", default=None,
                   help=f"coefficient estimates to score (default: {' '.join(COEFF_KINDS)})")
    p.add_argument("--mixtures-only", action="store_true",
                   help="drop the corner adapters, which fit themselves exactly")
    p.add_argument("--n-components", type=int, default=None,
                   help="embedding dimension (default: k-1, where the distances are exact)")
    p.add_argument("--geometry", default="mds", choices=["mds", "pca"],
                   help="mds (SMACOF) or pca (classical MDS, deterministic)")
    p.add_argument("--n-permutations", type=int, default=9999)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--no-protest", action="store_true",
                   help="skip the Procrustes permutation test")
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    main(
        {
            "run_dir": args.run_dir,
            "output_dir": args.output_dir,
            "kinds": args.kinds,
            "mixtures_only": args.mixtures_only,
            "n_components": args.n_components,
            "geometry": args.geometry,
            "n_permutations": args.n_permutations,
            "random_state": args.random_state,
            "no_protest": args.no_protest,
        },
        make_figures=not args.no_figures,
    )
