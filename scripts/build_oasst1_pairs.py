#!/usr/bin/env python
"""Derive a prompt/response table from OpenAssistant/oasst1, once, to local parquet.

oasst1 is not a table of prompt/response rows.  It ships 84,437 individual
*messages* -- 31,525 prompter turns and 52,912 assistant turns -- arranged into
conversation trees by ``parent_id``.  Turning that into the (prompt, response,
lang) shape the simplex experiment draws from is a join with real choices in it,
so it lives in this script rather than being implied by a config nobody reads.

**The pairing rule.**  For every assistant message that is not deleted and whose
``rank`` is 0 -- the best-ranked reply to its parent -- look up its parent.  Keep
the pair if that parent is a non-deleted prompter message.  ``prompt`` is the
parent's text, ``response`` is the message's own, and ``lang`` is the
*assistant* message's language, since that is the text a language vertex is
supposed to be about.

**Why ``rank == 0`` and not "any reply".**  Admitting unranked single replies
adds 3,215 rows, 264 of them ``zh``.  That would put ``zh`` at 1,002 rows, and a
1000-row draw from a 1,002-row pool is the whole pool on every seed -- strictly
worse than the 68% overlap the n=500 draw already accepts.  The rule is frozen
into the ``v1`` directory name and is expensive to revisit, so it is stated here
rather than left to be re-derived from the row count.

**Why the output is sorted, and why that is correctness rather than tidiness.**
``SampledDatasetCache`` records draws as *row indices* into the source split and
``source_registry.validate`` only checks ``num_rows``, so a non-deterministic row
order would silently repoint every index recorded against this dataset.  Sorting
by ``message_id`` -- a UUID, so the order is arbitrary but total and stable --
makes the mapping from index to row a property of the data instead of a property
of whichever Arrow shard happened to be scanned first.

**Why ``v1`` is in the path.**  A dataset built from local files has no Hub
revision, so ``source_registry.revision_of`` returns None and ``validate`` skips
that check entirely; ``num_rows`` is the only guard left.  Versioning the
directory means a change to the pairing rule lands somewhere new rather than
quietly invalidating draws already on disk.

Run once::

    python scripts/build_oasst1_pairs.py

Writes ``results/shared_cache/00_sources/oasst1_pairs_v1/``, which
``load_dataset`` resolves as a bare directory of parquet.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: The four vertices of the oasst1 simplex.  Chosen in the design document
#: because they are the only languages whose best-reply pools clear n=500.
LANGS = ("en", "es", "ru", "zh")

SOURCE_ID = "OpenAssistant/oasst1"
SPLIT = "train"
VERSION = "v1"

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "results" / "shared_cache" / "00_sources" / f"oasst1_pairs_{VERSION}"

#: Columns in the emitted table.  ``prompt``/``response``/``lang`` are what the
#: experiment reads; the rest is provenance, so a surprising row can be traced
#: back to the message it came from without re-running this script.
COLUMNS = ["prompt", "response", "lang", "message_id", "parent_id",
           "message_tree_id", "is_root"]


def build(rows_iter) -> list[dict]:
    """The pairing, as a pure function of the source rows.

    Two passes rather than one: the parent of an assistant message may appear
    after it in the split, so the index has to be complete before any lookup.
    """
    by_id: dict[str, dict] = {}
    for row in rows_iter:
        by_id[row["message_id"]] = row

    out: list[dict] = []
    for row in by_id.values():
        if row["role"] != "assistant" or row["deleted"]:
            continue
        if row["rank"] != 0:
            continue
        if row["lang"] not in LANGS:
            continue
        parent = by_id.get(row["parent_id"])
        if parent is None or parent["role"] != "prompter" or parent["deleted"]:
            continue
        out.append({
            "prompt": parent["text"],
            "response": row["text"],
            "lang": row["lang"],
            "message_id": row["message_id"],
            "parent_id": row["parent_id"],
            "message_tree_id": row["message_tree_id"],
            # Carried, and deliberately unused: it marks the pairs whose prompt
            # opened its tree, so a single-turn ablation is a filter rather than
            # a rebuild.  Not dead code -- unspent.
            "is_root": parent["parent_id"] is None,
            # Not emitted; used only for the cross-language count reported below.
            "_parent_lang": parent["lang"],
        })
    # Total and stable: message_id is a UUID, so this order is arbitrary but is
    # the same on every machine and every datasets version.  Recorded draw
    # indices point at rows, not at scan order.
    out.sort(key=lambda r: r["message_id"])
    return out


def fingerprint(rows: list[dict]) -> str:
    """A digest of the sorted message_id list, pasted into the emitted configs.

    Cheaper than a row hash and sufficient for the thing that actually goes
    wrong: a change to the pairing rule or the sort changes which row each
    recorded index names, and both change this.
    """
    h = hashlib.sha256()
    for row in rows:
        h.update(row["message_id"].encode())
    return h.hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(OUT_DIR),
                        help="Directory to write the parquet into.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the counts and the fingerprint, write nothing.")
    args = parser.parse_args()

    from datasets import Dataset, load_dataset

    source = load_dataset(SOURCE_ID, split=SPLIT)
    rows = build(source)

    counts = Counter(r["lang"] for r in rows)
    print(f"{SOURCE_ID} [{SPLIT}]: {len(source)} messages -> {len(rows)} pairs")
    for lang in LANGS:
        print(f"  {lang}  {counts[lang]}")
    print(f"  message_id fingerprint: {fingerprint(rows)}")

    # A handful of pairs answer a prompt written in another language.  Too few to
    # matter for a language vertex, but counted rather than assumed: the vertex is
    # the *assistant* message's language, so these rows carry a prompt that is not
    # in the vertex's language at all.
    cross = sum(1 for r in rows if r["_parent_lang"] != r["lang"])
    print(f"  pairs whose prompt is in another language: {cross}")

    if args.dry_run:
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = Dataset.from_dict({c: [r[c] for r in rows] for c in COLUMNS})
    table.to_parquet(str(out / "train-00000-of-00001.parquet"))
    print(f"\nWrote {out}")
    print("Load it with: load_dataset(<that directory>, split='train')")


if __name__ == "__main__":
    main()
