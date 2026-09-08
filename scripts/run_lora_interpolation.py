"""Fit trained mixture LoRAs as interpolations of the simplex-corner LoRAs.

For a family of adapters trained on mixtures of k dataset components, asks how
well the adapter trained on mixture w is reproduced by combining the k adapters
trained on the pure components, and what coefficients fit best.

Everything is closed form in the Gram matrix of effective updates, so the whole
run is a few seconds of single-threaded CPU on a login node — there is no SLURM
job for this and none is needed.

Usage:
    python scripts/run_lora_interpolation.py \\
        --adapter-root /exp/nverma/model_taxonomy/shared_cache/03_adapters \\
        --base-model Qwen/Qwen3.5-4B \\
        --output-dir results/lora_interpolation

    # only the full-attention modules
    python scripts/run_lora_interpolation.py ... --modules q_proj k_proj v_proj o_proj

The run also scores the fitted coefficients against the ground-truth simplex
(Procrustes and dCor); see scripts/run_coefficient_geometry.py, which does the
same thing standalone from a finished run's fits/fits.safetensors.  That step
needs scikit-learn and can be skipped with --no-geometry.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.interp.gram import load_or_compute_gram
from src.interp.loading import discover_adapters, mixture_proportions
from src.interp.report import build_report

DEFAULT_OUTPUT = "results/lora_interpolation"


def collect_truth(refs) -> tuple[list[str], dict[str, np.ndarray]]:
    """Mixture proportions for every adapter that encodes one.

    Adapters whose names carry no mixture are dropped with a note rather than
    guessed at — a target with unknown proportions cannot be scored.
    """
    vertices: list[str] | None = None
    truth: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    for ref in refs:
        parsed = mixture_proportions(ref.path)
        if parsed is None:
            skipped.append(ref.name)
            continue
        labels, weights = parsed
        if vertices is None:
            vertices = labels
        elif labels != vertices:
            raise ValueError(
                f"{ref.name} declares components {labels}, but earlier adapters "
                f"declare {vertices}. All adapters must share one simplex."
            )
        truth[ref.name] = weights
    if vertices is None:
        raise ValueError(
            "No adapter name encoded a mixture (expected segments like "
            "'025g1_050g2_025g3')."
        )
    if skipped:
        print(f"  note: {len(skipped)} adapter(s) carry no mixture, skipped: {skipped}")
    return vertices, truth


def main(
    cfg: dict,
    recompute: bool = False,
    make_figures: bool = True,
    make_geometry: bool = True,
) -> "object":
    """Run the analysis and write the report. Returns the InterpolationReport."""
    adapter_root = cfg["adapter_root"]
    base_model = cfg.get("base_model")
    output_dir = Path(cfg.get("output_dir", DEFAULT_OUTPUT))

    print(f"Scanning {adapter_root}")
    refs = discover_adapters(adapter_root, base_model_slug=base_model)
    if cfg.get("adapters"):
        wanted = set(cfg["adapters"])
        refs = [r for r in refs if r.name in wanted]
        missing = wanted - {r.name for r in refs}
        if missing:
            raise ValueError(f"Adapters not found: {sorted(missing)}")
    print(f"  {len(refs)} adapter(s)")

    vertices, truth = collect_truth(refs)
    print(f"  simplex vertices: {vertices}")

    gram_cache = cfg.get("gram_cache") or (output_dir / "gram_cache")
    gram, provenance = load_or_compute_gram(
        refs,
        cache_dir=gram_cache,
        recompute=recompute,
        threads=cfg.get("threads", 1),
    )
    print(
        f"  gram {gram.per_block.shape} "
        f"({'cached' if provenance['cache_hit'] else 'computed'}, "
        f"hash {provenance['gram_hash']})"
    )

    layers = cfg.get("layers")
    modules = cfg.get("modules")
    if layers or modules:
        gram = gram.restrict(layers=layers, modules=modules)
        print(f"  restricted to {len(gram.blocks)} block(s)")

    corners = cfg.get("corners")
    report = build_report(gram, truth, vertices, corners=corners, provenance=provenance)
    print(f"  corners: {report.corners}")

    output_dir.mkdir(parents=True, exist_ok=True)
    gram.save(output_dir / "gram")
    report.save(output_dir, figures=make_figures)
    print(f"\n{report.to_markdown()}")

    if make_figures:
        from src.interp.plots import make_all_figures

        paths = make_all_figures(report, gram, output_dir / "figures")
        print(f"  {len(paths)} figure(s) -> {output_dir / 'figures'}")

    if make_geometry:
        # Configuration-level scoring against the ideal simplex. Kept optional
        # because it is the only part of this script that needs scikit-learn.
        from src.interp.geometry import compare_from_report

        geo = compare_from_report(report)
        geo.save(output_dir, figures=make_figures)
        print(f"  coefficient geometry: {geo.headline()}")
        if make_figures:
            from src.interp.plots import make_geometry_figures

            make_geometry_figures(geo, output_dir / "figures")

    print(f"Wrote {output_dir}")
    return report


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter-root", required=True, help="directory of PEFT adapter dirs")
    p.add_argument("--base-model", default=None, help="e.g. Qwen/Qwen3.5-4B")
    p.add_argument("--adapters", nargs="+", default=None, help="restrict to these names")
    p.add_argument("--corners", nargs="+", default=None,
                   help="corner adapter names in vertex order (default: auto-detect)")
    p.add_argument("--layers", nargs="+", type=int, default=None, help="restrict to these layers")
    p.add_argument("--modules", nargs="+", default=None, help="restrict to these module names")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    p.add_argument("--gram-cache", default=None, help="default: {output_dir}/gram_cache")
    p.add_argument("--recompute", action="store_true", help="ignore a cached Gram")
    p.add_argument("--threads", type=int, default=1,
                   help="BLAS threads; 1 is ~20x faster here (small GEMMs)")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-geometry", action="store_true",
                   help="skip the simplex comparison (the only step needing scikit-learn)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    main(
        {
            "adapter_root": args.adapter_root,
            "base_model": args.base_model,
            "adapters": args.adapters,
            "corners": args.corners,
            "layers": args.layers,
            "modules": args.modules,
            "output_dir": args.output_dir,
            "gram_cache": args.gram_cache,
            "threads": args.threads,
        },
        recompute=args.recompute,
        make_figures=not args.no_figures,
        make_geometry=not args.no_geometry,
    )
