"""Re-embed cached behavioral generations with a second embedder — no decoding.

``05_generated`` splits an entry in two on purpose::

    …/{recipe_hash}/n{n}_s{seed}[_f{fmt}]/
        generations/{variant_token}.json                     ← the text
        embeddings/{variant_token}_{embedder_hash}.safetensors ← one per embedder

so a second embedder is meant to cost one forward pass over text that already
exists.  :class:`~src.cache.generated_text_cache.GeneratedTextCache` says as
much: "the text is generated once, and re-embedding it adds one file beside the
first with no GPU generation pass."

**Nothing implemented that.**  ``BehavioralTaxonomy.extract`` tests
:meth:`GeneratedTextCache.exists`, which requires *both* halves, so a draw whose
text is present but whose embedding is missing for the requested embedder falls
through to ``_extract_fresh`` and re-runs the whole decode.  For the qbig tree
that is ~5 GPU-hours to recompute 1,024,000 continuations that are already
sitting on disk, and — because sampling is seeded but batch-shape dependent —
the recomputed text would not even be guaranteed identical to the stored text.
This script is the missing path: read ``generations/*.json``, embed, write the
new ``embeddings/*`` beside the old.

What it does *not* do is change how anything is keyed.  The embedder config is
built through :class:`~src.embedders.SentenceTransformerEmbedder`, so the
``embedder_hash`` is byte-identical to what a normal extraction would have
written, and the entry it produces is indistinguishable from one a full re-run
would have left.  The run record is cloned from the existing record for the same
variant with only ``embedder``/``embedder_hash`` swapped, so ``runs/`` keeps
reading as a log of everything that touched the draw.

Example — the qbig draw, both variants, on nomic v2-moe::

    python scripts/reembed_behavioral.py \\
        --base-model allenai/OLMo-2-0425-1B-Instruct \\
        --draw n1000_s02_fea27ccee \\
        --embedder-model nomic-ai/nomic-embed-text-v2-moe

Sharding is by *draw directory* (one adapter × one draw), so ``--shard i
--num-shards k`` splits the fleet across k jobs that share nothing and can run
in any order.  Writes are atomic and idempotent: a shard that dies mid-fleet is
re-runnable, and an entry that already exists is skipped unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from src.cache.generated_text_cache import GeneratedTextCache  # noqa: E402
from src.core.representation import ModelRepresentation  # noqa: E402
from src.embedders import SentenceTransformerEmbedder  # noqa: E402


def _default_shared_cache() -> Path:
    """Where to look for the cache when ``--cache-root`` is silent.

    Same worktree handling as ``scripts/check_analysis.py``: run from
    ``.claude/worktrees/<name>`` the in-repo path resolves inside the worktree,
    where no cache has ever been written, and the script would report an empty
    fleet rather than an unresolved path.
    """
    parts = REPO.parts
    if len(parts) >= 3 and parts[-2] == "worktrees" and parts[-3] == ".claude":
        main_checkout = REPO.parents[2] / "results/shared_cache"
        if main_checkout.exists():
            return main_checkout
    return REPO / "results/shared_cache"


def find_draws(
    cache: GeneratedTextCache, base_model: str, adapter_glob: str, draw: str
) -> list[Path]:
    """Every ``…/{recipe_hash}/{draw}`` directory for one base model.

    Matched on the *directory* name rather than reconstructed from a query key,
    because the query key is what we are trying to discover: ``queries.json``
    inside each hit is canonical and is what the writes are keyed on.
    """
    base_dir = cache._base / base_model.replace("/", "--")
    if not base_dir.is_dir():
        raise SystemExit(f"no cached generations for base model {base_model!r} "
                         f"under {base_dir}")
    hits = sorted(
        d for d in base_dir.glob(f"{adapter_glob}/*/{draw}")
        if d.is_dir() and (d / "generations").is_dir()
    )
    return hits


def _clone_run_config(draw_dir: Path, sampling_hash: str, replicates: int) -> dict | None:
    """The config of the existing run for this variant, if one is recorded.

    Cloning beats synthesising: the config carries fields this script has no way
    to know and no business inventing — ``torch_dtype`` above all, which names
    the precision the *generations* were drawn at and stays true of them however
    they are later embedded.
    """
    runs = draw_dir / "runs"
    if not runs.is_dir():
        return None
    for path in sorted(runs.glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if (rec.get("sampling_hash") == sampling_hash
                and int(rec.get("replicates", -1)) == int(replicates)):
            cfg = rec.get("config")
            if isinstance(cfg, dict):
                return json.loads(json.dumps(cfg))  # deep copy
    return None


def reembed_draw(
    cache: GeneratedTextCache,
    draw_dir: Path,
    base_model: str,
    embedder: SentenceTransformerEmbedder,
    embedder_hash: str,
    *,
    batch_size: int,
    variant_glob: str,
    force: bool,
    dry_run: bool,
) -> list[str]:
    """Re-embed every generation variant in one draw directory.

    Returns one human-readable line per variant, for the log.
    """
    adapter_name = draw_dir.parent.parent.name
    queries = json.loads((draw_dir / "queries.json").read_text())
    query_key = queries["query_key"]
    source_indices = queries.get("source_indices") or []

    lines: list[str] = []
    for gen_path in sorted((draw_dir / "generations").glob(f"{variant_glob}.json")):
        # Not _GEN_RE: that regex describes an *embedding* filename and requires
        # the embedder component, which a generations stem does not carry.
        parts = gen_path.stem.split("_")
        if len(parts) != 3 or not parts[0].startswith("generation"):
            lines.append(f"    ?? {gen_path.name}: unparseable variant token, skipped")
            continue
        max_new_tokens = int(parts[0][len("generation"):])

        gen = json.loads(gen_path.read_text())
        texts = gen.get("generated_texts") or []
        replicates = int(gen.get("replicates", 1))
        sampling = gen["sampling"]
        model_id = gen["model_id"]
        sampling_hash = GeneratedTextCache.sampling_hash(sampling)

        target = cache.embeddings_path(
            base_model, adapter_name, query_key, max_new_tokens,
            replicates, sampling_hash, embedder_hash,
        )
        if target.exists() and not force:
            lines.append(f"    -- {gen_path.stem}: already embedded, skipped")
            continue

        nested = [t if isinstance(t, list) else [t] for t in texts]
        flat = [t for per_query in nested for t in per_query]
        if not flat:
            lines.append(f"    !! {gen_path.stem}: no generated text, skipped")
            continue

        if dry_run:
            lines.append(f"    -> {gen_path.stem}: would embed {len(flat)} texts "
                         f"-> {target.name}")
            continue

        t0 = time.time()
        matrix = embedder.encode_texts(flat, batch_size=batch_size)
        dt = time.time() - t0

        if matrix.shape[0] != len(flat):
            raise RuntimeError(
                f"{gen_path}: embedder returned {matrix.shape[0]} rows for "
                f"{len(flat)} texts"
            )

        config = _clone_run_config(draw_dir, sampling_hash, replicates) or {
            "taxonomy": "behavioral",
            "query_key": {k: query_key[k] for k in
                          ("recipe_hash", "n_samples", "seed", "prompt_format_id")
                          if k in query_key},
            "n_queries": len(nested),
            "max_new_tokens": max_new_tokens,
            "replicates": replicates,
            "sampling": dict(sampling),
        }
        config["embedder"] = embedder.config_dict()

        rep = ModelRepresentation(
            model_id=model_id,
            taxonomy="behavioral",
            matrix=matrix,
            metadata={
                "generated_texts": nested,
                "think_closure": gen.get("think_closure"),
                # Provenance for this pass specifically.  The generations were
                # produced by some earlier job on some other card; what this
                # records is the machine that turned them into these vectors.
                "device_name": embedder.device,
                "batch_size": batch_size,
                "reembedded_from_generations": gen_path.name,
            },
        )

        cache.save(
            base_model,
            adapter_name,
            query_key,
            rep,
            max_new_tokens=max_new_tokens,
            replicates=replicates,
            sampling=sampling,
            embedder_hash=embedder_hash,
            config=config,
            source_indices=source_indices,
        )
        rate = len(flat) / dt if dt else float("inf")
        lines.append(
            f"    ok {gen_path.stem}: {matrix.shape[0]}x{matrix.shape[1]} in "
            f"{dt:.1f}s ({rate:.0f} texts/s) -> {target.name}"
        )
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-root", type=Path, default=None)
    p.add_argument("--base-model", required=True,
                   help="e.g. allenai/OLMo-2-0425-1B-Instruct")
    p.add_argument("--draw", required=True,
                   help="draw directory name, e.g. n1000_s02_fea27ccee")
    p.add_argument("--adapter-glob", default="*",
                   help="restrict which adapters are re-embedded (default: all)")
    p.add_argument("--variant-glob", default="*",
                   help="restrict which generation variants are re-embedded, "
                        "e.g. 'generation128_64r_*' for the sampled one only")
    p.add_argument("--embedder-model", required=True)
    p.add_argument("--prompt-name", default="search_document",
                   help="task prefix to embed under; pass '' for none. Every "
                        "behavioral config in this repo uses search_document, "
                        "and it is part of the embedder_hash -- changing it "
                        "makes a different cache entry, not a corrected one.")
    p.add_argument("--no-trust-remote-code", action="store_true")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--force", action="store_true",
                   help="rewrite entries that already exist")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cache_root = args.cache_root or _default_shared_cache()
    cache = GeneratedTextCache(cache_root)

    embedder = SentenceTransformerEmbedder(
        model_name=args.embedder_model,
        device=args.device,
        use_generated_text=True,
        normalize_embeddings=not args.no_normalize,
        trust_remote_code=not args.no_trust_remote_code,
        prompt_name=args.prompt_name or None,
    )
    embedder_hash = GeneratedTextCache.embedder_hash(embedder.config_dict())

    draws = find_draws(cache, args.base_model, args.adapter_glob, args.draw)
    if not draws:
        raise SystemExit(
            f"no draw directories matched {args.adapter_glob}/*/{args.draw} under "
            f"{cache_root / '05_generated' / args.base_model.replace('/', '--')}"
        )
    mine = draws[args.shard::args.num_shards]

    print(f"cache root   : {cache_root}")
    print(f"base model   : {args.base_model}")
    print(f"draw         : {args.draw}")
    print(f"embedder     : {args.embedder_model} "
          f"(prompt_name={args.prompt_name!r}, hash={embedder_hash})")
    print(f"prefix       : {embedder.prompt_prefix!r}")
    print(f"fleet        : {len(draws)} draw dirs; shard {args.shard}/"
          f"{args.num_shards} takes {len(mine)}")
    print(flush=True)

    t_all = time.time()
    for i, draw_dir in enumerate(mine, 1):
        adapter = draw_dir.parent.parent.name
        print(f"[{i}/{len(mine)}] {adapter}", flush=True)
        for line in reembed_draw(
            cache, draw_dir, args.base_model, embedder, embedder_hash,
            batch_size=args.batch_size, variant_glob=args.variant_glob,
            force=args.force, dry_run=args.dry_run,
        ):
            print(line, flush=True)

    print(f"\ncomplete ({time.time() - t_all:.1f}s)")


if __name__ == "__main__":
    main()
