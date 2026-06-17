"""Print distance matrices from a saved taxonomy run.

Usage:
    python scripts/show_distances.py results/yahoo_topics
    python scripts/show_distances.py results/yahoo_topics --taxonomy behavioral
    python scripts/show_distances.py results/yahoo_topics --taxonomy functional --precision 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.distance import DistanceMatrix


def _short(model_id: str, max_len: int = 30) -> str:
    return model_id if len(model_id) <= max_len else "..." + model_id[-(max_len - 3):]


def print_matrix(dm: DistanceMatrix, precision: int = 3) -> None:
    labels = [_short(m) for m in dm.model_ids]
    col_w = max(len(l) for l in labels)
    row_w = max(len(l) for l in labels)

    fmt = f"{{:.{precision}f}}"
    num_w = precision + 4

    header = " " * (row_w + 2) + "  ".join(l.rjust(num_w) for l in labels)
    print(header)
    print(" " * (row_w + 2) + "-" * (len(header) - row_w - 2))

    for i, row_label in enumerate(labels):
        row = "  ".join(fmt.format(dm.matrix[i, j]).rjust(num_w) for j in range(len(labels)))
        print(f"{row_label.rjust(row_w)}  {row}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Display taxonomy distance matrices.")
    parser.add_argument("results_dir", help="Experiment output directory (e.g. results/yahoo_topics)")
    parser.add_argument("--taxonomy", nargs="+", metavar="NAME",
                        help="Only show these taxonomies (default: all found)")
    parser.add_argument("--precision", type=int, default=3, help="Decimal places (default: 3)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    tax_root = results_dir / "taxonomy"

    if not tax_root.exists():
        print(f"No taxonomy results found at {tax_root}")
        sys.exit(1)

    available = sorted(
        d.name for d in tax_root.iterdir()
        if d.is_dir() and (d / "distance_matrix" / "distance_matrix.safetensors").exists()
    )

    if not available:
        print(f"No distance matrices found under {tax_root}")
        sys.exit(1)

    names = [t for t in available if t in args.taxonomy] if args.taxonomy else available

    for name in names:
        dm = DistanceMatrix.load(tax_root / name / "distance_matrix")
        print(f"\n{'=' * 60}")
        print(f"  {name.upper()}  |  metric: {dm.metric}  |  n={len(dm.model_ids)}")
        print(f"{'=' * 60}")
        print_matrix(dm, precision=args.precision)

    print()


if __name__ == "__main__":
    main()
