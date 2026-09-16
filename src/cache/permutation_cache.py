"""``07A_permutation_tests`` — label-null permutation tests over stored collections.

A **label null** keeps both real geometries and permutes which model is which,
destroying only the model-to-model correspondence.  It is the counterpart to the
**structure null** of :mod:`src.analysis.baselines`, which instead replaces the
taxonomy with a structureless configuration and keeps the truth.  The figure
artifact that displays it is the **permutation band**; see
``docs/terminology.md`` and ``docs/notes/chance_baselines.md``, which spell
``label null`` where the figures spell ``permutation null``.  They are one
concept under two names, and nothing here is a third.

**The letter suffix is load-bearing.**  ``docs/terminology.md`` fixes the rule
that directories sharing a number sit at the same stage and a letter suffix
means analysis *of* the objects at that stage, as ``03A_adapter_alignments``
is analysis of ``03_adapters``.  A permutation test is analysis of the distance
matrices and geometries in ``07_collections``, so it is ``07A_``, not ``08_``.

**Why a derived quantity may be stored here at all.**
:meth:`src.plots.simplex_suite.SuiteCache.register` sets the policy: store a
derived quantity only when it cannot be cheaply reconstructed from something
more primitive, because a second copy can drift from its source with no way to
tell which is stale.  An assembled distance matrix fails that test — it is 120
dict lookups away from the pairs — and is deliberately not stored.  A
permutation null passes it on exactly the grounds an MDS fit does: 9,999
re-superimpositions are not an assembly, and the result depends on an RNG
stream.  That dependence is also why ``random_state`` and ``n_permutations``
are **in the key**, the same correction ``CollectionCache.geometry_key`` makes
one level down.

Directory layout::

    cache_root/07A_permutation_tests/
        index.json                      ← catalogue of every result
        {test}/                         ← protest | dcor
            {group_key}/                ← one model set against one truth
                group_info.json         ← members, n, source handle, truth kind
                {test_key}/             ← one parameterisation of the test
                    config.json         ← params + the scalar results
                    null.safetensors    ← the null distribution, float64

**Two levels, as both sibling tiers use.**  ``group_info.json`` describes the
model set and is shared by every parameterisation beneath it; the leaf's
``config.json`` describes one test.  Re-running the same cells at a different
``n_permutations`` therefore adds leaves rather than rewriting the group.

**The group key is what makes a subgroup addressable.**  A five-model subgroup
of a 1003-model pool has no ``07_collections`` entry of its own — the sweeps
build it with ``dm.reindex(...)`` off the pool matrix and throw it away.
Hashing the *sorted member list* alongside the pool's handle gives that
transient object a stable name, which is the same rule
:meth:`src.cache.collection_cache.CollectionCache.collection_key` uses for a
whole collection.

**The null goes in safetensors, not JSON.**  A 9,999-draw ``float64`` null is
80 KB; as JSON it would be several times that and would round-trip through
decimal.  ``float64`` rather than ``float32`` for the reason
``CollectionCache.save_distance_matrix`` gives: a cached value must be
bit-identical to a freshly computed one, or the only test that the reuse is
correct stops testing anything.

**Accepted exposure: a stored null does not know which code produced it.**  A
key carries the test's *parameters*, never the implementation's version, so a
behavioural change to ``protest``, to ``dcor_test``, or to any metric feeding
them leaves every stored null keyed exactly as before and the next run reads
back numbers computed by the old code.  This tier inherits the exposure
**doubly** — from the metric, as ``06_pairwise`` does, and again from the test.
It is accepted rather than solved for the same reason: a hand-bumped version
constant is only as good as the discipline of remembering to bump it, so it
fails open on forgetfulness while reading as protection.  **The remedy:** after
changing either, delete the affected results
(``rm -rf 07A_permutation_tests/{test}``) and let them recompute.  Nothing else
needs touching, because ``index.json`` is a catalogue no read depends on.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.utils.atomic import atomic_path, atomic_write_json

from ._draw_keyed import DrawKeyedCache

#: The tests this tier stores.  Named rather than free-form so a typo lands as
#: an error at write time instead of as a directory nobody ever reads again.
TESTS = ("protest", "dcor")

#: How the truth side of a comparison was built.  ``simplex`` is the mixture
#: grid a suite was trained on; ``requested`` and ``realized`` are the nsweep's
#: two readings of it, which differ because largest-remainder allocation makes a
#: requested 25/75 realize as 3/7 at N=10.
TRUTH_KINDS = ("simplex", "requested", "realized")


class PermutationCache:
    """Cache for label-null permutation tests, keyed by model set and parameters.

    *cache_root* is the shared cache root, not the stage directory — matching
    :class:`~src.cache.collection_cache.CollectionCache` and
    :class:`~src.cache.pairwise_cache.PairwiseCache`.
    """

    _STAGE_DIR = "07A_permutation_tests"

    def __init__(self, cache_root: Path | str) -> None:
        self.root = Path(cache_root)
        self._tests_dir = self.root / self._STAGE_DIR
        #: Results served from disk and results written, over this object's
        #: lifetime.  The unit is one test on one cell, which is the unit the
        #: caller decides to skip or pay for.
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------------

    @staticmethod
    def group_key(*, members, source_handle: str | None, truth_kind: str) -> str:
        """Name the model set a test was run on.

        Members are hashed **sorted**, so the key does not depend on the order
        the caller happened to assemble them in — the rule
        ``CollectionCache.collection_key`` uses, and the reason a subgroup
        drawn twice by different code paths lands on one entry.

        *source_handle* is the ``07_collections`` handle the distances came
        from.  It is part of the key rather than decoration: the same models
        scored under a different surrogate or metric are a different
        measurement, and two such results must not collide.
        """
        if truth_kind not in TRUTH_KINDS:
            raise ValueError(
                f"unknown truth_kind {truth_kind!r}; have {list(TRUTH_KINDS)}")
        return DrawKeyedCache.config_hash({
            "members": sorted(str(m) for m in members),
            "source": source_handle,
            "truth": truth_kind,
        })

    @staticmethod
    def test_key(*, members, source_handle: str | None, truth_kind: str,
                 test: str, params: dict) -> str:
        """Name one parameterisation of one test on one model set.

        *params* must carry **everything that changes the numbers** —
        ``n_permutations`` and ``random_state`` at minimum, plus the
        test-specific flags (``scaling`` / ``reflection`` for PROTEST,
        ``bias_corrected`` for dCor) and the ``n_components`` the taxonomy
        geometry was fitted at.  Omitting one does not make the key smaller; it
        makes two different measurements share a name.
        """
        if test not in TESTS:
            raise ValueError(f"unknown test {test!r}; have {list(TESTS)}")
        group = PermutationCache.group_key(
            members=members, source_handle=source_handle, truth_kind=truth_kind)
        return DrawKeyedCache.config_hash({
            "group": group, "test": test, "params": params})

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self._tests_dir / "index.json"

    def _group_dir(self, test: str, group_key: str) -> Path:
        return self._tests_dir / test / group_key

    def _leaf_dir(self, test: str, group_key: str, test_key: str) -> Path:
        return self._group_dir(test, group_key) / test_key

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def has_result(self, test: str, group_key: str, test_key: str) -> bool:
        """Whether a complete result is stored.  Cheap: one ``stat``."""
        return (self._leaf_dir(test, group_key, test_key)
                / "null.safetensors").exists()

    def load_result(self, test: str, group_key: str, test_key: str,
                    *, with_null: bool = True) -> dict | None:
        """The stored result, or ``None`` on a miss.

        With ``with_null=False`` only the scalars are read.  That is the common
        case for a figure — a band needs quantiles, which are stored — and it
        skips opening a 80 KB tensor per cell across thousands of cells.
        """
        leaf = self._leaf_dir(test, group_key, test_key)
        config_path = leaf / "config.json"
        if not config_path.exists():
            self.misses += 1
            return None
        try:
            out = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError):
            self.misses += 1
            return None
        if with_null:
            from safetensors.numpy import load_file

            null_path = leaf / "null.safetensors"
            if not null_path.exists():
                self.misses += 1
                return None
            out["null"] = load_file(str(null_path))["null"].astype(np.float64)
        self.hits += 1
        return out

    def load_group_info(self, test: str, group_key: str) -> dict:
        path = self._group_dir(test, group_key) / "group_info.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def load_index(self) -> dict[str, dict]:
        """The catalogue.  Recomputable from the tree; no read depends on it."""
        if not self.index_path.exists():
            return {}
        try:
            return json.loads(self.index_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def list_results(self) -> list[tuple[str, str, str]]:
        """``(test, group_key, test_key)`` for every stored result, by walking.

        Walks rather than reading ``index.json``, so a stale or deleted
        catalogue cannot hide stored work.
        """
        if not self._tests_dir.exists():
            return []
        out = []
        for path in sorted(self._tests_dir.rglob("null.safetensors")):
            rel = path.parent.relative_to(self._tests_dir)
            if len(rel.parts) == 3:
                out.append(tuple(rel.parts))
        return out

    def find(self, **criteria) -> list[dict]:
        """Index records matching every given field.

        A convenience over ``load_index``, not an authority — a result absent
        from the catalogue is still readable by its keys.
        """
        return [rec for rec in self.load_index().values()
                if all(rec.get(k) == v for k, v in criteria.items())]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_result(self, *, test: str, members, source_handle: str | None,
                    truth_kind: str, params: dict, statistic: float,
                    p_value: float, null: np.ndarray, n_models: int,
                    n_permutations: int, exact: bool = False,
                    label: str | None = None,
                    provenance: dict | None = None) -> tuple[str, str]:
        """Store one test result.  Returns ``(group_key, test_key)``.

        **Two locks, never nested**, following ``PairwiseCache.save_pairs``:
        ``group_info.json`` and the leaf are written under the *group* lock,
        which is released before the *root* lock is taken to merge
        ``index.json``.  The catalogue is shared by every group, so holding a
        group lock while reaching for the root one is the single ordering that
        could deadlock against a writer doing the reverse — and there is no
        reason to hold both.
        """
        from filelock import FileLock
        from safetensors.numpy import save_file

        if test not in TESTS:
            raise ValueError(f"unknown test {test!r}; have {list(TESTS)}")
        members = [str(m) for m in members]
        group_key = self.group_key(members=members, source_handle=source_handle,
                                   truth_kind=truth_kind)
        test_key = self.test_key(members=members, source_handle=source_handle,
                                 truth_kind=truth_kind, test=test, params=params)

        group_dir = self._group_dir(test, group_key)
        leaf_dir = self._leaf_dir(test, group_key, test_key)
        now = datetime.now(timezone.utc).isoformat()

        info = {
            "schema_version": "1",
            "group_key": group_key,
            "test": test,
            "truth_kind": truth_kind,
            "source_handle": source_handle,
            "label": label,
            "n_models": n_models,
            "members": sorted(members),
            "updated": now,
        }
        config = {
            "schema_version": "1",
            "test": test,
            "test_key": test_key,
            "group_key": group_key,
            "params": dict(params),
            "statistic": float(statistic),
            "p_value": float(p_value),
            "n_models": int(n_models),
            "n_permutations": int(n_permutations),
            "exact": bool(exact),
            "provenance": dict(provenance or {}),
            "created_at": now,
        }
        # The quantiles the figures actually read, stored beside the scalars so
        # a band never has to open the null tensor.  BAND_Q rather than a local
        # literal, so the band this tier serves and the band
        # `src.analysis.baselines` serves cannot drift apart.
        from src.analysis.baselines import BAND_Q

        arr = np.asarray(null, dtype=np.float64).ravel()
        lo, hi = (float(np.percentile(arr, q)) for q in BAND_Q)
        config["band"] = {"q_lo": BAND_Q[0], "q_hi": BAND_Q[1],
                          "lo": lo, "mid": float(np.median(arr)), "hi": hi}

        group_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(group_dir / "group.lock")):
            atomic_write_json(group_dir / "group_info.json", info, sort_keys=True)
            leaf_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(leaf_dir / "config.json", config, sort_keys=True)
            meta_bytes = np.frombuffer(
                json.dumps({"test": test, "group_key": group_key,
                            "test_key": test_key}).encode("utf-8"),
                dtype=np.uint8,
            )
            with atomic_path(leaf_dir / "null.safetensors") as tmp:
                save_file({"null": np.ascontiguousarray(arr),
                           "_meta_json": meta_bytes}, str(tmp))

        self._update_index(test, group_key, test_key, info, config)
        return group_key, test_key

    def _update_index(self, test: str, group_key: str, test_key: str,
                      info: dict, config: dict) -> None:
        """Merge one result's summary into ``index.json``, atomically."""
        from filelock import FileLock

        self._tests_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "test": test,
            "group_key": group_key,
            "test_key": test_key,
            "truth_kind": info.get("truth_kind"),
            "source_handle": info.get("source_handle"),
            "label": info.get("label"),
            "n_models": info.get("n_models"),
            "n_permutations": config.get("n_permutations"),
            "statistic": config.get("statistic"),
            "p_value": config.get("p_value"),
            "exact": config.get("exact"),
            "updated": info.get("updated"),
        }
        with FileLock(str(self._tests_dir / "index.lock")):
            index = self.load_index()
            index[f"{test}/{group_key}/{test_key}"] = record
            atomic_write_json(self.index_path, index, sort_keys=True)

    def rebuild_index(self) -> int:
        """Rebuild ``index.json`` by walking the tree.  Returns the entry count.

        The catalogue is never load-bearing, so this is a repair for humans
        reading it rather than a step any cache hit needs.
        """
        from filelock import FileLock

        index = {}
        for test, group_key, test_key in self.list_results():
            config = self.load_result(test, group_key, test_key, with_null=False)
            if config is None:
                continue
            info = self.load_group_info(test, group_key)
            index[f"{test}/{group_key}/{test_key}"] = {
                "test": test, "group_key": group_key, "test_key": test_key,
                "truth_kind": info.get("truth_kind"),
                "source_handle": info.get("source_handle"),
                "label": info.get("label"),
                "n_models": config.get("n_models"),
                "n_permutations": config.get("n_permutations"),
                "statistic": config.get("statistic"),
                "p_value": config.get("p_value"),
                "exact": config.get("exact"),
                "updated": info.get("updated"),
            }
        self._tests_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self._tests_dir / "index.lock")):
            atomic_write_json(self.index_path, index, sort_keys=True)
        return len(index)
