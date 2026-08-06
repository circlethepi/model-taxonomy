#!/usr/bin/env python
"""Rename ``05_generated`` entries into the replicate-and-sampling spelling.

**What changed and why.**  Behavioral generation used to be greedy, one
continuation per query, so a stored entry was fully identified by its token
budget and its embedder::

    generations/generation128.json
    embeddings/generation128_{embedder}.safetensors

Generation now samples, and draws ``replicates`` continuations per query.  Both
facts change the numbers, and neither was in the name.  Since
:meth:`GeneratedTextCache.save` is idempotent *on the filename*, a sampled run
over a draw that already held a greedy entry would have been a silent no-op that
returned the greedy numbers — the hazard the class documents for ``torch_dtype``,
on an axis that changes the result far more.  So both are in the name now::

    generations/generation128_1r_6f000f01.json
    embeddings/generation128_1r_6f000f01_{embedder}.safetensors

**``6f000f01`` is greedy.**  It is
``GeneratedTextCache.sampling_hash(GeneratedTextCache.GREEDY_SAMPLING)``, frozen
here as a literal rather than recomputed.  Freezing is the point: if the sampling
canon ever gains an axis, every *future* hash moves and these already-renamed
files must not.  The same reasoning froze the digest in
``scripts/migrate_recipe_identity.py``.

**Nothing is recomputed and nothing is discarded.**  Tensor files are renamed, so
they stay byte-identical; only the generations JSON is rewritten, to nest its
flat ``list[str]`` into the ``list[list[str]]`` that a replicate-aware reader
expects (one replicate per query, so each string becomes a one-element list) and
to record the greedy sampling settings alongside it.

The greedy entries are **kept, not superseded**.  They are what the 2026-08-05
measurement in ``docs/notes/TODO.md`` was computed from, and keeping them costs
2.4 MB.  They remain addressable by their sampling hash — and are *not*
comparable with sampled runs, which is exactly why the hash is in the name.

Safety model, following ``scripts/migrate_behavioral_layout.py``:

- **Self-verifying.**  Every renamed entry is read back through
  :class:`GeneratedTextCache` and required to match the original bytes exactly
  (``np.array_equal``, not ``allclose``), and its text to survive round-trip.
- **``--revert`` exists** and is exact: the old name is the new one with the two
  added components removed, and the nesting is undone.
- **Refuses rather than merges.**  A destination that already exists is an error.

Usage::

    python scripts/migrate_behavioral_replicates.py --dry-run
    python scripts/migrate_behavioral_replicates.py --apply
    python scripts/check_analysis.py
    python scripts/migrate_behavioral_replicates.py --revert   # if needed
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

#: The pre-replicate stem: mode and embedder, nothing else.  Frozen here rather
#: than imported, because ``generated_text_cache._GEN_RE`` now describes the
#: *new* names and this migration is the only thing that still has to recognise
#: the old ones.
_OLD_RE = re.compile(r"^(?P<mode>generation\d+)_(?P<embedder>[0-9a-f]{16})$")

#: What greedy decoding hashes to.  See the module docstring on why this is a
#: literal and not a call.
GREEDY_HASH = "6f000f01"

#: One replicate: greedy produced exactly one continuation per query.
GREEDY_REPLICATES = 1


def _greedy_sampling() -> dict:
    from src.cache.generated_text_cache import GeneratedTextCache

    return dict(GeneratedTextCache.GREEDY_SAMPLING)


def _check_frozen_hash() -> None:
    """Fail loudly if the frozen digest and the live canon have diverged.

    They agree today.  If they ever stop agreeing, the honest outcome is a
    refusal to run: silently renaming to a *different* hash would put entries
    somewhere no reader looks for them, which is the class of bug item 13 was.
    """
    from src.cache.generated_text_cache import GeneratedTextCache

    live = GeneratedTextCache.sampling_hash(GeneratedTextCache.GREEDY_SAMPLING)
    if live != GREEDY_HASH:
        raise SystemExit(
            f"frozen greedy hash {GREEDY_HASH} != live {live}. The sampling canon "
            "changed after entries were renamed under the old digest. Decide "
            "deliberately: either re-freeze and re-migrate, or leave the old "
            "entries where they are."
        )


# ── discovery ─────────────────────────────────────────────────────────────────

def find_entries(root: Path) -> list[dict]:
    """Every embedding still carrying the pre-replicate stem, with its peers."""
    from src.cache._draw_keyed import _DRAW_RE

    base = root / "05_generated"
    if not base.exists():
        return []

    entries = []
    for emb in sorted(base.glob("*/*/*/*/embeddings/*.safetensors")):
        draw_dir = emb.parent.parent
        if not _DRAW_RE.match(draw_dir.name):
            continue
        m = _OLD_RE.match(emb.stem)
        if not m:
            continue
        mode = m["mode"]
        entries.append({
            "embedding": emb,
            "draw_dir": draw_dir,
            "mode": mode,
            "embedder": m["embedder"],
            "generations": draw_dir / "generations" / f"{mode}.json",
            "new_embedding": emb.with_name(
                f"{mode}_{GREEDY_REPLICATES}r_{GREEDY_HASH}_{m['embedder']}.safetensors"
            ),
            "new_generations": draw_dir / "generations"
            / f"{mode}_{GREEDY_REPLICATES}r_{GREEDY_HASH}.json",
        })
    return entries


def _read_matrix(path: Path):
    from safetensors.numpy import load_file

    return load_file(str(path))["matrix"]


# ── apply ─────────────────────────────────────────────────────────────────────

def migrate(root: Path, entries: list[dict], dry_run: bool) -> int:
    import numpy as np

    from src.cache.generated_text_cache import GeneratedTextCache

    cache = GeneratedTextCache(root)
    verb = "would rename" if dry_run else "renamed"
    sampling = _greedy_sampling()
    n_texts = 0

    for e in entries:
        if e["new_embedding"].exists():
            raise SystemExit(
                f"refusing to overwrite {e['new_embedding']}: a replicate-named "
                "entry already exists beside the old one. Inspect both before "
                "deciding which is authoritative."
            )
        if not e["generations"].exists():
            raise SystemExit(
                f"{e['embedding']} has no generations peer at {e['generations']}. "
                "Half an entry is not safe to rename; the text is the part that "
                "cannot be recomputed."
            )

        print(f"    {verb} {e['embedding'].relative_to(root)}")
        print(f"           -> {e['new_embedding'].name}")

        payload = json.loads(e["generations"].read_text())
        texts = payload.get("generated_texts", [])
        n_texts += len(texts)
        if dry_run:
            continue

        before = _read_matrix(e["embedding"])

        e["embedding"].rename(e["new_embedding"])
        e["new_generations"].write_text(
            json.dumps(
                {
                    "schema_version": "3",
                    "model_id": payload.get("model_id"),
                    "replicates": GREEDY_REPLICATES,
                    "sampling": sampling,
                    # One greedy continuation per query becomes a one-element
                    # list, so texts[q][r] indexes the same way at every R.
                    "generated_texts": [[t] for t in texts],
                },
                indent=2,
            )
        )
        e["generations"].unlink()

        # Verify through the reader, not by re-globbing: the point is that the
        # entry is reachable at its new coordinates by the code that will read it.
        base_model_id, adapter_id, query_key = _coordinates(e, root)
        rep = cache.load(
            base_model_id, adapter_id, query_key,
            max_new_tokens=int(e["mode"][len("generation"):]),
            replicates=GREEDY_REPLICATES,
            sampling_hash=GREEDY_HASH,
            embedder_hash=e["embedder"],
        )
        if not np.array_equal(rep.matrix, before):
            raise SystemExit(
                f"verification failed for {e['new_embedding']}: the matrix read "
                "back differs from the one written. Nothing further was touched."
            )
        got_texts = rep.metadata.get("generated_texts", [])
        if [t[0] for t in got_texts] != texts:
            raise SystemExit(
                f"verification failed for {e['new_embedding']}: generated text "
                "did not survive nesting."
            )

    print(
        f"\n  {verb} {len(entries)} entry(ies); "
        f"{n_texts} generated text(s) nested, none dropped."
    )
    return 0


def _coordinates(entry: dict, root: Path) -> tuple[str, str, dict]:
    """``(base_model_id, adapter_id, query_key)`` recovered from the path.

    The path *is* the key at this stage — that is what item 13 bought — so this
    is a parse, not a lookup, and it needs no file open.
    """
    from src.cache._draw_keyed import _DRAW_RE

    draw_dir = entry["draw_dir"]
    recipe_hash = draw_dir.parent.name
    m = _DRAW_RE.match(draw_dir.name)
    query_key = {
        "recipe_hash": recipe_hash,
        "n_samples": int(m.group(1)),
        "seed": int(m.group(2)),
    }
    adapter_slug = draw_dir.parent.parent.name
    base_slug = draw_dir.parent.parent.parent.name
    return base_slug.replace("--", "/"), adapter_slug, query_key


# ── revert ────────────────────────────────────────────────────────────────────

def revert(root: Path, dry_run: bool) -> int:
    """Undo the rename for entries written as ``1r`` under the greedy hash.

    Deliberately narrow: only ``{GREEDY_REPLICATES}r_{GREEDY_HASH}`` entries are
    reverted, because those are the ones this script created.  A sampled entry
    has no pre-replicate spelling to go back to — its name is the only record of
    how it was produced — so reverting one would destroy information.
    """
    base = root / "05_generated"
    verb = "would revert" if dry_run else "reverted"
    n = 0

    suffix = f"_{GREEDY_REPLICATES}r_{GREEDY_HASH}"
    for emb in sorted(base.glob("*/*/*/*/embeddings/*.safetensors")):
        stem = emb.stem
        parts = stem.split("_")
        if len(parts) != 4 or f"_{parts[1]}_{parts[2]}" != suffix:
            continue
        mode, embedder = parts[0], parts[3]
        old_emb = emb.with_name(f"{mode}_{embedder}.safetensors")
        gen = emb.parent.parent / "generations" / f"{mode}{suffix}.json"
        old_gen = gen.with_name(f"{mode}.json")

        print(f"    {verb} {emb.name} -> {old_emb.name}")
        n += 1
        if dry_run:
            continue

        emb.rename(old_emb)
        if gen.exists():
            payload = json.loads(gen.read_text())
            texts = payload.get("generated_texts", [])
            old_gen.write_text(
                json.dumps(
                    {
                        "model_id": payload.get("model_id"),
                        # Flatten back: every entry this script made has R=1.
                        "generated_texts": [t[0] if isinstance(t, list) else t
                                            for t in texts],
                    },
                    indent=2,
                )
            )
            gen.unlink()

    print(f"\n  {verb} {n} entry(ies) to the pre-replicate spelling.")
    return 0


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cache-root", default="results/shared_cache", type=Path)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="report and change nothing (default)")
    g.add_argument("--apply", action="store_true", help="perform the rename")
    g.add_argument("--revert", action="store_true", help="undo it")
    args = p.parse_args()

    root = args.cache_root
    _check_frozen_hash()

    if args.revert:
        print(f"\n  Reverting {root}/05_generated to the pre-replicate spelling\n")
        return revert(root, dry_run=False)

    dry_run = not args.apply
    entries = find_entries(root)
    print(f"\n  {root}/05_generated: {len(entries)} pre-replicate entry(ies)\n")
    if not entries:
        print("  Nothing to do — every entry already carries a replicate count.")
        return 0

    rc = migrate(root, entries, dry_run=dry_run)
    if dry_run:
        print("\n  Dry run. Re-run with --apply to perform it.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
