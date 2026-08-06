#!/usr/bin/env python
"""Re-key ``05_generated`` from run-wise to model-wise, matching ``04_activations``.

**What was wrong.**  Behavioral entries lived at
``05_generated/{config_hash}/embeddings/{model_slug}.safetensors``, where
``model_slug`` hashed the adapter's **full path**.  The stored paths were
*relative* (``results/shared_cache/03_adapters/…``), so the key depended on the
working directory the extraction ran in — and any reader scanning the cache with
an absolute root computed a different slug and found nothing.  This was not a
latent hazard.  Measured before this migration: ``behavioral_repr`` resolved to
**0 hits across all 25 adapters × 2 configs** while 10 representations sat
readable on disk.  Every write had succeeded; the cache simply read as empty.

It stayed invisible because the two checks touching behavioral both bypass the
broken path — one reads by config hash directly, the other excludes behavioral
unless exactly one config exists, and there were two.

**What it becomes.**  The same coordinates ``04_activations`` uses::

    05_generated/{base_slug}/{adapter_slug}/{recipe_hash}/n{n}_s{seed}/
        queries.json                       ← query_key + source_indices, no text
        runs/{config_hash}.json            ← the original run, preserved
        generations/{mode_token}.json
        embeddings/{mode_token}_{embedder_hash}.safetensors

Nothing is hashed from a path.  ``base_slug`` and ``adapter_slug`` come from
:mod:`src.cache._draw_keyed`, shared with the functional level, so one model
under one draw sits at one place in both stages.

**This drops the stored query text.**  The old layout copied all 64 query strings
into a ``queries.json`` beside every run.  ``(recipe_hash, n_samples, seed)``
determines them completely — ``text_field`` is part of the recipe and therefore
part of ``recipe_hash`` — so ``01_datasets`` was always canonical and the copy was
redundant.  The **generated** text, which is model output and genuinely
irreplaceable, is preserved in full.  ``--dry-run`` reports the count it will
drop rather than dropping it silently.

Safety model:

- **Self-verifying.**  Every migrated entry is read back and required to match
  the original bytes exactly (``np.array_equal``, not ``allclose``) before the
  old tree is touched.
- **``--revert`` exists**, because this migration *is* reversible: the old
  ``config_hash`` is preserved in ``runs/`` and the old ``model_slug`` is
  recomputable from the stored ``model_id``.
- **Refuses rather than merges.**  Two entries resolving to one destination is an
  error, not a last-writer-wins.

Usage::

    python scripts/migrate_behavioral_layout.py --dry-run
    python scripts/migrate_behavioral_layout.py --apply
    python scripts/check_analysis.py
    python scripts/migrate_behavioral_layout.py --revert   # if needed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_ROOT = REPO / "results" / "shared_cache"

#: Old-layout config directories are 16 hex characters.
_OLD_DIR_RE = __import__("re").compile(r"^[0-9a-f]{16}$")


def _old_model_slug(model_id: str) -> str:
    """The retired ``generated_text_cache.model_slug``.

    Kept here, and only here, so ``--revert`` can rebuild the old filenames and
    so the inventory can cross-check that a file it found is the file it thinks
    it found.  It must not come back into ``src/``.
    """
    digest = hashlib.sha256(str(model_id).encode()).hexdigest()[:8]
    return f"{Path(str(model_id)).name}__{digest}"


# ── Phase 1: inventory ────────────────────────────────────────────────────────

def inventory(root: Path) -> list[dict]:
    """Every (config, model) entry in the old layout, with its stored metadata."""
    from safetensors.numpy import load_file

    base = root / "05_generated"
    if not base.exists():
        raise SystemExit(f"{base} does not exist; nothing to migrate")

    entries: list[dict] = []
    for cfg_dir in sorted(base.iterdir()):
        if not cfg_dir.is_dir() or not _OLD_DIR_RE.match(cfg_dir.name):
            continue
        cfg_path = cfg_dir / "config.json"
        if not cfg_path.exists():
            raise SystemExit(
                f"{cfg_dir} looks like an old config directory but has no config.json; "
                "refusing to guess what it is"
            )
        config = json.loads(cfg_path.read_text())
        query_key = config.get("query_key") or {}
        missing = {"recipe_hash", "n_samples", "seed"} - set(query_key)
        if missing:
            raise SystemExit(f"{cfg_path} has an incomplete query_key, missing {sorted(missing)}")

        queries_path = cfg_dir / "queries.json"
        n_queries_stored = 0
        if queries_path.exists():
            n_queries_stored = len(json.loads(queries_path.read_text()).get("queries", []))

        emb_dir, gen_dir = cfg_dir / "embeddings", cfg_dir / "generations"
        for st in sorted(emb_dir.glob("*.safetensors")):
            gen_path = gen_dir / f"{st.stem}.json"
            if not gen_path.exists():
                raise SystemExit(f"{st} has no matching generations/{st.stem}.json")

            tensors = load_file(str(st))
            meta = json.loads(tensors["_meta_json"].tobytes().decode("utf-8"))
            gen = json.loads(gen_path.read_text())

            model_id = meta["model_id"]
            if gen.get("model_id") != model_id:
                raise SystemExit(
                    f"{st.stem}: embedding says model_id={model_id!r} but its generations "
                    f"peer says {gen.get('model_id')!r}; refusing to migrate a mismatched pair"
                )
            if _old_model_slug(model_id) != st.stem:
                raise SystemExit(
                    f"{st.name} does not match the slug of its own stored model_id "
                    f"({_old_model_slug(model_id)}); the file was renamed or the id rewritten"
                )

            entries.append({
                "config_hash": cfg_dir.name,
                "config": config,
                "query_key": query_key,
                "model_id": model_id,
                "old_stem": st.stem,
                "matrix": tensors["matrix"],
                "meta": meta,
                "generated_texts": gen.get("generated_texts", []),
                "n_queries_stored": n_queries_stored,
            })

    if not entries:
        raise SystemExit(
            f"found no entries under {base}. A walk that finds nothing must not read as "
            "'nothing to do' — check --root."
        )
    return entries


# ── Phase 2: resolve destinations ─────────────────────────────────────────────

def resolve(entries: list[dict], root: Path) -> list[dict]:
    """Attach ``base_model_id`` and ``embedder_hash`` to every entry, or refuse."""
    from src.cache.generated_text_cache import GeneratedTextCache
    from src.taxonomy._hf_inference import HFInferenceTaxonomy

    adapters_root = root / "03_adapters"
    seen: dict[tuple, str] = {}

    for e in entries:
        # Stored IDs are relative, so never open Path(model_id) directly — that
        # is the very cwd-dependence being removed.  Re-root on the 03_adapters
        # component against --root instead.
        parts = Path(e["model_id"]).parts
        if "03_adapters" not in parts:
            raise SystemExit(f"{e['model_id']!r} has no 03_adapters component to re-root on")
        tail = parts[parts.index("03_adapters") + 1:]
        if len(tail) < 2:
            raise SystemExit(f"{e['model_id']!r} does not look like {{base_slug}}/{{adapter}}")
        adapter_dir = adapters_root.joinpath(*tail)
        if not adapter_dir.exists():
            raise SystemExit(f"{adapter_dir} does not exist; cannot recover its base model")

        # Two independent sources must agree.
        from_config = HFInferenceTaxonomy._resolve_base_model_id(adapter_dir)
        from_path = tail[0].replace("--", "/")
        if from_config is None:
            raise SystemExit(f"{adapter_dir} has no recoverable base_model_id")
        if from_config != from_path:
            raise SystemExit(
                f"{adapter_dir}: adapter config says base {from_config!r} but it is stored "
                f"under {from_path!r}. Refusing to guess which is right."
            )

        e["base_model_id"] = from_config
        e["adapter_id"] = str(adapter_dir)
        e["max_new_tokens"] = int(e["config"]["max_new_tokens"])

        # The highest-risk guard here: recompute the embedder hash the way a
        # future extraction will, and require it to match what the stored config
        # describes.  If these disagree, migrated files land under a name the
        # next run never looks at — a silent GPU re-run rather than an error.
        stored_embedder = e["config"]["embedder"]
        rebuilt = _rebuild_embedder_config(stored_embedder)
        if rebuilt != stored_embedder:
            raise SystemExit(
                "the embedder config a fresh SentenceTransformerEmbedder reports does not "
                "match the one stored in config.json.\n"
                f"  stored:  {json.dumps(stored_embedder, sort_keys=True)}\n"
                f"  rebuilt: {json.dumps(rebuilt, sort_keys=True)}\n"
                "Migrating would file these under a name the next extraction never reads."
            )
        e["embedder_hash"] = GeneratedTextCache.embedder_hash(stored_embedder)

        key = (
            e["base_model_id"], adapter_dir.name,
            tuple(sorted(e["query_key"].items())),
            e["max_new_tokens"], e["embedder_hash"],
        )
        if key in seen:
            raise SystemExit(
                f"{e['old_stem']} and {seen[key]} both resolve to the same destination. "
                "Refusing to merge two entries into one."
            )
        seen[key] = e["old_stem"]

    return entries


def _rebuild_embedder_config(stored: dict) -> dict:
    """``config_dict()`` of an embedder constructed from *stored*.

    Scoped to ``05_generated`` deliberately.  The ``02_dataset_embeddings``
    entries predate the nomic prompt-prefix fix and have no ``prompt_prefix``
    key, so their configs are not directly comparable to these and a shared
    guard would fire spuriously.  See ``docs/notes/embedder_task_prefixes.md``.
    """
    from src.embedders.sentence_transformer import SentenceTransformerEmbedder

    emb = SentenceTransformerEmbedder(
        model_name=stored["model_name"],
        device="cpu",                       # outside config_dict by design
        use_generated_text=stored.get("use_generated_text", True),
        normalize_embeddings=stored.get("normalize_embeddings", True),
        trust_remote_code=stored.get("trust_remote_code", False),
        prompt_name=stored.get("prompt_name"),
    )
    return emb.config_dict()


# ── Phase 3: write ────────────────────────────────────────────────────────────

def write(entries: list[dict], root: Path, dry_run: bool) -> dict:
    from src.cache.generated_text_cache import GeneratedTextCache
    from src.core.representation import ModelRepresentation

    cache = GeneratedTextCache(root)
    verb = "would write" if dry_run else "wrote"
    stats = {"written": 0, "queries_dropped": 0}

    for e in entries:
        dest = cache.embeddings_path(
            e["base_model_id"], e["adapter_id"], e["query_key"],
            e["max_new_tokens"], e["embedder_hash"],
        )
        print(f"    {verb} {dest.relative_to(root)}")
        stats["queries_dropped"] += e["n_queries_stored"]

        if dry_run:
            continue

        metadata = dict(e["meta"].get("metadata", {}))
        metadata["generated_texts"] = e["generated_texts"]
        rep = ModelRepresentation(
            model_id=e["model_id"],
            taxonomy=e["meta"].get("taxonomy", "behavioral"),
            matrix=e["matrix"],
            metadata=metadata,
            cache_key="",
        )
        cache.save(
            e["base_model_id"], e["adapter_id"], e["query_key"], rep,
            max_new_tokens=e["max_new_tokens"],
            embedder_hash=e["embedder_hash"],
            # The original config, so runs/ keeps its hash — the only link back
            # to the run these numbers actually came from.
            config=e["config"],
            source_indices=None,
        )
        stats["written"] += 1

    # queries_dropped is counted per entry above but the file is per config.
    stats["queries_dropped"] = sum(
        {e["config_hash"]: e["n_queries_stored"] for e in entries}.values()
    )
    return stats


# ── Phase 4: self-verify ──────────────────────────────────────────────────────

def verify(entries: list[dict], root: Path) -> dict:
    from src.cache.activation_cache import ActivationCache
    from src.cache.generated_text_cache import GeneratedTextCache

    cache = GeneratedTextCache(root)
    act = ActivationCache(root)
    stats = {"verified": 0, "shared_with_functional": 0, "per_config": {}}

    for e in entries:
        rep = cache.load(
            e["base_model_id"], e["adapter_id"], e["query_key"],
            e["max_new_tokens"], e["embedder_hash"],
        )
        if not np.array_equal(rep.matrix, e["matrix"]):
            raise SystemExit(f"{e['old_stem']}: matrix changed through the migration")
        if rep.model_id != e["model_id"]:
            raise SystemExit(f"{e['old_stem']}: model_id changed through the migration")
        if rep.metadata.get("generated_texts") != e["generated_texts"]:
            raise SystemExit(f"{e['old_stem']}: generated text changed through the migration")
        if rep.matrix.shape[0] != int(e["query_key"]["n_samples"]):
            raise SystemExit(
                f"{e['old_stem']}: {rep.matrix.shape[0]} rows but the draw says "
                f"n_samples={e['query_key']['n_samples']}"
            )
        stats["verified"] += 1

        # The payoff: does this model-draw sit at the same coordinates in both
        # stages?  Compared relative to each stage dir, so the check is about the
        # shared suffix rather than the stage name.
        beh = cache.draw_dir(e["base_model_id"], e["adapter_id"], e["query_key"])
        fun = act.draw_dir(e["base_model_id"], e["adapter_id"], e["query_key"])
        cfg = e["config_hash"]
        stats["per_config"].setdefault(cfg, 0)
        if fun.exists():
            if beh.relative_to(cache._base) != fun.relative_to(act._base):
                raise SystemExit(
                    f"{e['old_stem']}: behavioral and functional disagree on coordinates\n"
                    f"  behavioral: {beh.relative_to(cache._base)}\n"
                    f"  functional: {fun.relative_to(act._base)}"
                )
            stats["shared_with_functional"] += 1
            stats["per_config"][cfg] += 1

    return stats


# ── Phase 5: remove the old tree ──────────────────────────────────────────────

def remove_old(entries: list[dict], root: Path, dry_run: bool) -> dict:
    verb = "would remove" if dry_run else "removed"
    old = sorted({e["config_hash"] for e in entries})
    for h in old:
        print(f"    {verb} 05_generated/{h}/")
        if not dry_run:
            shutil.rmtree(root / "05_generated" / h)
    return {"removed": len(old)}


# ── revert ────────────────────────────────────────────────────────────────────

def revert(root: Path, dry_run: bool) -> int:
    """Rebuild the old run-wise tree from the new model-wise one.

    Possible because ``runs/`` preserved the original ``config_hash`` and the
    stored ``model_id`` regenerates the old slug.  The query text cannot be
    restored — it was dropped deliberately — so ``queries.json`` comes back
    without it, which is enough for the old reader.
    """
    from src.cache._draw_keyed import _DRAW_RE
    from src.cache.generated_text_cache import _GEN_RE, GeneratedTextCache

    base = root / "05_generated"
    cache = GeneratedTextCache(root)
    verb = "would restore" if dry_run else "restored"
    n = 0

    for emb in sorted(base.glob("*/*/*/*/embeddings/*.safetensors")):
        draw_dir = emb.parent.parent
        if not _DRAW_RE.match(draw_dir.name):
            continue
        m = _GEN_RE.match(emb.stem)
        if not m:
            continue
        runs = sorted((draw_dir / "runs").glob("*.json"))
        if not runs:
            print(f"    !! {emb} has no runs/ record; cannot recover its config_hash")
            return 1
        run = json.loads(runs[0].read_text())
        config_hash, config = runs[0].stem, run["config"]
        model_id = run.get("model_id")
        if not model_id:
            print(f"    !! {runs[0]} has no model_id; cannot recover the old slug")
            return 1

        old_dir = base / config_hash
        stem = _old_model_slug(model_id)
        print(f"    {verb} {config_hash}/embeddings/{stem}.safetensors")
        n += 1
        if dry_run:
            continue

        (old_dir / "embeddings").mkdir(parents=True, exist_ok=True)
        (old_dir / "generations").mkdir(parents=True, exist_ok=True)
        query_key = config["query_key"]
        max_new_tokens = int(config["max_new_tokens"])
        emb_hash = m["embedder"]

        rep = cache.load(
            run.get("base_model_id") or _base_from_path(emb, base),
            model_id, query_key, max_new_tokens, emb_hash,
        )
        shutil.copy2(emb, old_dir / "embeddings" / f"{stem}.safetensors")
        (old_dir / "generations" / f"{stem}.json").write_text(
            json.dumps(
                {"model_id": model_id,
                 "generated_texts": rep.metadata.get("generated_texts", [])},
                indent=2,
            )
        )
        if not (old_dir / "config.json").exists():
            (old_dir / "config.json").write_text(
                json.dumps({"schema_version": "1", "config_hash": config_hash, **config},
                           indent=2)
            )
            (old_dir / "queries.json").write_text(
                json.dumps({"query_key": query_key, "queries": []}, indent=2)
            )

    print(f"\n  {verb} {n} entry(ies) into the old run-wise layout.")
    if not dry_run:
        print("  The new model-wise tree was left in place; remove it by hand once happy.")
    return 0


def _base_from_path(emb: Path, base: Path) -> str:
    return emb.relative_to(base).parts[0].replace("--", "/")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="plan only, write nothing")
    mode.add_argument("--apply", action="store_true", help="write the new layout")
    mode.add_argument("--revert", action="store_true",
                      help="rebuild the old run-wise layout from the new one")
    args = parser.parse_args()

    root = args.root.resolve()
    print(f"cache root: {root}\n")

    if args.revert:
        print("Phase R — rebuilding the old layout")
        return revert(root, dry_run=False)

    dry = args.dry_run

    print("Phase 1 — inventory")
    entries = inventory(root)
    configs = sorted({e["config_hash"] for e in entries})
    print(f"  {len(entries)} entry(ies) across {len(configs)} config(s): {configs}\n")

    print("Phase 2 — resolve destinations")
    entries = resolve(entries, root)
    print(f"  all {len(entries)} resolved; base models: "
          f"{sorted({e['base_model_id'] for e in entries})}\n")

    print("Phase 3 — write")
    wstats = write(entries, root, dry_run=dry)
    print(f"  {wstats['written']} written; "
          f"{wstats['queries_dropped']} stored query string(s) dropped "
          f"(recoverable from 01_datasets)\n")

    if dry:
        print("Phase 4 — self-verify   (skipped: nothing was written)")
        print("Phase 5 — remove old tree")
        remove_old(entries, root, dry_run=True)
        print("\nDry run only. Re-run with --apply to make these changes.")
        return 0

    print("Phase 4 — self-verify")
    vstats = verify(entries, root)
    print(f"  {vstats['verified']} entry(ies) round-tripped byte-identical")
    print(f"  {vstats['shared_with_functional']} share coordinates with 04_activations")
    for cfg in configs:
        print(f"      {cfg}: {vstats['per_config'].get(cfg, 0)}")
    if vstats["shared_with_functional"] == 0:
        print("  !! no model-draw is present at both levels. That is not fatal, but the")
        print("     payoff of this migration is unobserved — check the draw actually matches.")

    print("\nPhase 5 — remove old tree")
    remove_old(entries, root, dry_run=False)

    print("\nDone. Next:")
    print("    python scripts/check_analysis.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
