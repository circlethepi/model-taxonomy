"""Persistent cache of sampled dataset draws, keyed by (recipe_hash, n_samples, seed).

A *draw* is the post-shuffle/filter row list that MixedDataset / ClassMixedDataset
produce.  Downstream steps (embedding, fine-tuning) read draws rather than re-running
the sampler, which matters most at large n where materialising rows dominates.

Directory layout::

    cache_root/01_datasets/{recipe_hash}/
        recipe.json           ← the mixing spec this hash identifies (immutable)
        names.json            ← human labels that resolved to this hash (accumulates)
        n{n}_s{seed}.json     ← one draw manifest

``recipe_hash`` is content-addressed (see ``DatasetRecipe._canonical``), so one mixture
has one directory no matter how many ``_n{n}_s{seed}`` config-block names refer to it.
That is why n and seed live in the *filename* rather than the hash.

Draws store **source indices, not row text**.  A manifest records which rows of which
upstream split were drawn, in order, and rehydration selects them back out::

    {"schema_version": "2",
     "n_samples": 1000, "seed": 0,
     "sources": [{"dataset_id": ..., "split": ..., "subset": ...,
                  "revision": ..., "num_rows": ..., "hf_fingerprint": ...}],
     "indices": [[source_index, row_index], ...],
     "rows_sha256": "..."}

This costs ~85× less disk than storing rows (2.07 GiB → ~25 MB across the 564 draws
that existed when it was introduced) and rehydrates faster than the per-row ``dict()``
loop it replaces, because ``Dataset.select`` works off memory-mapped Arrow.

The trade is a dependency on the upstream dataset, guarded in three layers:
``revision`` pins the Hub commit so a cold machine downloads the right data;
``num_rows`` catches a reshaped split; and ``rows_sha256`` is authoritative — a
rehydrated draw that does not reproduce it raises rather than returning other rows.
:mod:`src.datasets.source_registry` owns all three.

Schema version 1 — a bare ``list[dict]`` of rows — remains readable.  It is what every
draw looked like before indices, and it is still what a source that cannot be indexed
would need, though writing it is currently refused rather than done silently (see
``put``).  This counter is independent of ``recipe.json``'s own ``schema_version``.

Writes are atomic (temp-file rename).  ``names.json`` is read-merge-write under a
FileLock, since one directory is now shared by many draws written concurrently.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

#: Draw-manifest schema.  Independent of recipe.json's schema_version.
DRAW_SCHEMA_VERSION = "2"


class SampledDatasetCache:
    """File-backed cache for sampled draws, their recipes, and their labels.

    Draw keys are ``(recipe_hash, n_samples, seed)`` triples; values are lists of row
    dicts (all original columns).  Calling code extracts the text field it wants.

    Takes the cache root and appends its own directory name, the same contract as
    :class:`~src.cache.lora_cache.LoRACache` and the other cache classes.
    """

    def __init__(self, cache_root: Path | str, hf_token: str | None = None) -> None:
        self.root = Path(cache_root) / "01_datasets"
        self.root.mkdir(parents=True, exist_ok=True)
        self.hf_token = hf_token

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def _dir(self, recipe_hash: str) -> Path:
        return self.root / recipe_hash

    def _path(self, recipe_hash: str, n_samples: int, seed: int) -> Path:
        return self._dir(recipe_hash) / f"n{n_samples}_s{seed}.json"

    def _recipe_path(self, recipe_hash: str) -> Path:
        return self._dir(recipe_hash) / "recipe.json"

    def _names_path(self, recipe_hash: str) -> Path:
        return self._dir(recipe_hash) / "names.json"

    # ------------------------------------------------------------------
    # Draws
    # ------------------------------------------------------------------

    def exists(self, recipe_hash: str, n_samples: int, seed: int) -> bool:
        return self._path(recipe_hash, n_samples, seed).exists()

    def get(
        self, recipe_hash: str, n_samples: int, seed: int, hf_token: str | None = None
    ) -> list[dict] | None:
        """The rows of a draw, or None if it was never cached.

        Raises rather than returning different rows if the upstream data moved: see
        :func:`src.datasets.source_registry.validate` and the ``rows_sha256`` check.
        """
        path = self._path(recipe_hash, n_samples, seed)
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            return self._read_v1(payload)
        return self._read_v2(payload, path, hf_token or self.hf_token)

    def get_manifest(self, recipe_hash: str, n_samples: int, seed: int) -> dict | None:
        """A draw's manifest without rehydrating it, or None for a miss or a v1 draw.

        Cheap — it only parses the index file.  Lets a cache hit carry its provenance
        forward, so a draw read from one cache can be written to another.
        """
        path = self._path(recipe_hash, n_samples, seed)
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        return None if isinstance(payload, list) else payload

    def put(
        self,
        recipe_hash: str,
        n_samples: int,
        seed: int,
        *,
        rows: list[dict],
        indices: list[tuple[int, int]] | None,
        sources: list[dict] | None,
    ) -> None:
        """Record a draw as source indices.

        *rows* is used only to compute the checksum — it is not stored.  *indices* and
        *sources* come off the sampler (``ds.source_indices`` / ``ds.sources``).
        """
        if indices is None or sources is None:
            raise ValueError(
                f"Cannot cache draw {recipe_hash} n={n_samples} seed={seed}: the sampler "
                f"did not report source indices. Every source must be a map-style "
                f"Dataset that can be indexed; a streaming source cannot be cached this "
                f"way. Storing full rows instead would silently cost ~85x the disk, so "
                f"it is refused rather than done quietly."
            )
        if len(indices) != len(rows):
            raise ValueError(
                f"Draw {recipe_hash} n={n_samples} seed={seed}: {len(indices)} indices "
                f"for {len(rows)} rows. These must correspond one-to-one and in order."
            )
        manifest = {
            "schema_version": DRAW_SCHEMA_VERSION,
            "n_samples": n_samples,
            "seed": seed,
            "sources": sources,
            "indices": [list(pair) for pair in indices],
            "rows_sha256": rows_checksum(rows),
        }
        self._write(self._path(recipe_hash, n_samples, seed), json.dumps(manifest))

    # ------------------------------------------------------------------
    # Draw readers, one per schema version
    # ------------------------------------------------------------------

    @staticmethod
    def _read_v1(payload: list[dict]) -> list[dict]:
        """Pre-index format: the rows themselves, stored verbatim."""
        return payload

    @staticmethod
    def _read_v2(payload: dict, path: Path, hf_token: str | None) -> list[dict]:
        """Index format: select the recorded rows back out of the upstream splits."""
        from src.datasets import source_registry

        sources = payload["sources"]
        indices = [tuple(pair) for pair in payload["indices"]]

        # One select() per source rather than per row; then reassemble in draw order.
        wanted: dict[int, list[int]] = {}
        for source_index, row_index in indices:
            wanted.setdefault(source_index, []).append(row_index)

        materialised: dict[int, dict[int, dict]] = {}
        for source_index, row_indices in wanted.items():
            desc = sources[source_index]
            ds = source_registry.get(
                desc["dataset_id"], desc.get("subset"), desc.get("split", "train"),
                revision=desc.get("revision"), token=hf_token,
            )
            source_registry.validate(
                desc,
                source_registry.describe(
                    ds, desc["dataset_id"], desc.get("subset"), desc.get("split", "train")
                ),
            )
            unique = sorted(set(row_indices))
            materialised[source_index] = {
                i: dict(row) for i, row in zip(unique, ds.select(unique))
            }

        rows = [materialised[s][i] for s, i in indices]

        expected = payload.get("rows_sha256")
        if expected and rows_checksum(rows) != expected:
            raise source_registry.SourceMismatch(
                f"{path}: rehydrated rows do not match the recorded checksum. The "
                f"upstream data changed in a way the revision and row-count checks did "
                f"not catch; this draw can no longer be reproduced."
            )
        return rows

    # ------------------------------------------------------------------
    # Recipes
    # ------------------------------------------------------------------

    def get_recipe(self, recipe_hash: str) -> dict | None:
        """The recipe dict this hash identifies, or None if it was never written."""
        path = self._recipe_path(recipe_hash)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def put_recipe(self, recipe_hash: str, recipe: dict) -> None:
        """Mirror a recipe next to its draws.

        Idempotent — an existing recipe.json is left alone.  The hash pins the content,
        so rewriting it would only churn the file.  This stays correct because labels
        live in ``names.json``: nothing mutable is stored here.
        """
        path = self._recipe_path(recipe_hash)
        if path.exists():
            return
        self._write(path, json.dumps(recipe, indent=2))

    # ------------------------------------------------------------------
    # Names
    # ------------------------------------------------------------------

    def get_names(self, recipe_hash: str) -> list[str]:
        """Every label that has resolved to this hash, sorted."""
        path = self._names_path(recipe_hash)
        if not path.exists():
            return []
        return json.loads(path.read_text()).get("names", [])

    def add_name(self, recipe_hash: str, name: str | None) -> None:
        """Record a label for this hash, merging with any already present.

        Because the hash is content-addressed, several config-block names legitimately
        resolve to one recipe — ``yahoo_x_n100_s00`` and ``yahoo_x_n1000_s03`` and a
        differently-spelled twin all land here.  Read-merge-write under a lock, since
        concurrent jobs sampling different draws of one mixture write this same file.
        """
        if not name:
            return
        # Fast path: this runs once per draw, and after the first write the answer is
        # almost always "already there".  A stale read only costs a redundant lock —
        # the check inside the lock is the one that decides.
        if name in self.get_names(recipe_hash):
            return

        from filelock import FileLock

        directory = self._dir(recipe_hash)
        directory.mkdir(parents=True, exist_ok=True)
        with FileLock(str(directory / ".lock")):
            names = set(self.get_names(recipe_hash))
            if name in names:
                return
            names.add(name)
            self._write(self._names_path(recipe_hash), json.dumps(
                {"names": sorted(names)}, indent=2
            ))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _write(path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(payload)
        os.replace(tmp, path)


def rows_checksum(rows: list[dict]) -> str:
    """SHA-256 over a draw's rows.

    The authoritative identity of a draw's *content*, independent of where the rows
    came from.  Writer and reader must serialise identically, so this is the one
    definition both use.
    """
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()
