"""``06_pairwise`` — individual pairwise distances, reusable across collections.

A distance matrix is not the unit of reuse; a *pair* is.  ``07_collections``
keys a whole matrix on the set of models it covers, so adding one model to a
16-model collection is a total miss and recomputes all 120 distances to obtain
16 new ones, and asking for a 5-model subgroup of a warm 16-model collection
recomputes all 10.  Keying on the pair instead makes both free: a matrix for any
model set, in any order, is *assembled by lookup*.

Row order stops existing as a concept below the assembly step — a pair is keyed
on the unordered pair of models, and the matrix is filled in whatever order the
caller asked for.  Subsets and supersets are then correct by construction rather
than by a second, hand-written notion of "the same collection".

Directory layout::

    cache_root/06_pairwise/
        index.json                                   ← catalogue of every handle
        {taxonomy}/
            {selector_slug}_{selector_key}/          ← one surrogate
                meta.json                            ← selector + per-model identity
                {metric}/                            ← one perspective
                    pairs.json                       ← {pair_id: distance}

**The metric sits below the selector**, because several metrics are computed
over one surrogate: the figure suite resolves the representations once and then
fills every metric column from them.  Putting the metric on top would scatter
one surrogate's results across sibling directories.  The consequence is that
``meta.json`` describes the selector and the models rather than the metric, so
it lives once at the surrogate level and is shared by every metric beneath it —
the same split ``CollectionCache`` uses for ``collection_info.json`` against a
leaf's ``config.json``.

A **handle** is ``{taxonomy}/{selector_slug}_{selector_key}/{metric}`` and
addresses exactly one **perspective**: a surrogate together with a similarity
metric.  See ``docs/terminology.md``.

**One ``pairs.json`` per handle, not one file per pair.**  16 models is 120
pairs, and a per-pair inode layout is hostile to Weka.  Appending is a
read-modify-write under ``filelock``, matching how ``CollectionCache``
guards concurrent SLURM writers.

The four guards this store enforces are documented on the methods that enforce
them: G1 (fleet transforms bypass the store) belongs to the caller and lives in
:func:`src.analysis.comparison._distances_via_pairs`; G2 (artifact identity must
agree) and G4 (one query draw per handle) are enforced in :meth:`save_pairs`;
G3 (assembly by identity, never by position) belongs to assembly.

**Accepted exposure: a pair does not know which metric implementation made it.**
A handle carries the metric's *name*, never its version, so a behavioural change
to a metric — a fix to ``cka_linear``, different regularization in
``bures_wasserstein`` — leaves every stored pair keyed exactly as before, and
the next run reads back distances computed by the old code.  Nothing here can
detect that.  It is accepted rather than solved: ``07_collections`` already
behaves this way, and the alternative — a hand-bumped version constant per
metric — is only as reliable as the discipline of remembering to bump it, so it
fails open on forgetfulness while reading as protection.  **The remedy:** after
changing a metric's behaviour, delete the affected handles
(``rm -rf 06_pairwise/{taxonomy}/*/{metric}``) and rebuild ``index.json`` by
walking the tree.  Nothing else needs touching, because ``index.json`` is a
catalogue no cache hit depends on.  See ``docs/notes/pairwise_store.md``.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from ._draw_keyed import DrawKeyedCache

#: The selector fields that go into the readable slug, in this fixed order.
#: Fixed because dict iteration order is a property of how a dict was built, and
#: two call paths that build the same selector differently would otherwise slug
#: it differently.
_SLUG_FIELDS = ("mode", "pooling", "view", "normalize", "replicate_reduction",
                "is_kernel", "layers")

#: A handle component stays well inside any filesystem limit: 48 + 1 + 16 = 65.
_SLUG_MAX = 48


class PairwiseCache:
    """Cache for individual pairwise distances, keyed by perspective.

    *cache_root* is the shared cache root, not the stage directory — matching
    :class:`~src.cache.collection_cache.CollectionCache`, and required because
    ``artifact_path`` is stored relative to it.
    """

    _STAGE_DIR = "06_pairwise"

    def __init__(self, cache_root: Path | str) -> None:
        self.root = Path(cache_root)
        self._pairs_dir = self.root / self._STAGE_DIR
        #: Pairs served from disk and pairs written, counted across this
        #: object's lifetime. Reporting per *pair* rather than per matrix is the
        #: only honest unit here: one matrix is now a mix of both.
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------------

    @staticmethod
    def pair_id(a, b, *, qualifier=None) -> str:
        """Order-free key for one pair.

        *qualifier* distinguishes two entries that share model ids but not the
        tensors behind them — two query draws of one model, say.  With
        ``qualifier=None``, which is the uniform-draw case and everything run
        today, the id is the bare readable ``a__b`` form.

        The parameter exists from the outset so that enabling a cross-draw mode
        later *adds* ids rather than rewriting the ones already on disk.  That
        is the same forward-compatibility rule the transform key follows in
        ``src.analysis.comparison.collection_handle``.
        """
        if qualifier is None:
            return "__".join(sorted([str(a), str(b)]))
        qa, qb = qualifier(str(a)), qualifier(str(b))
        return "__".join(sorted([f"{a}@{qa}", f"{b}@{qb}"]))

    @staticmethod
    def selector_key(selector: dict) -> str:
        """Digest of the resolved selector — the identity of one surrogate."""
        return DrawKeyedCache.config_hash(dict(selector or {}))

    @staticmethod
    def selector_slug(selector: dict, *, fallback: str = "") -> str:
        """A readable, **non-identifying** prefix for a surrogate directory.

        Presentation only: nothing may read identity from it, which is what lets
        it be improved later without orphaning a single entry.  It is still
        specified exactly, so two implementations produce the same directory
        names and a ``diff`` of two cache roots stays meaningful.
        """
        selector = dict(selector or {})
        parts: list[str] = []
        for field in _SLUG_FIELDS:
            value = selector.get(field)
            if value is None:
                continue
            if field == "is_kernel":
                if value:
                    parts.append("kernel")
                continue
            if field == "layers":
                parts.append(_layers_token(value))
                continue
            parts.append(str(value))

        slug = "_".join(p for p in parts if p).lower()
        slug = re.sub(r"[^a-z0-9_-]+", "-", slug)
        if len(slug) > _SLUG_MAX:
            slug = slug[:_SLUG_MAX]
            cut = slug.rfind("_")
            if cut >= _SLUG_MAX - 8:
                slug = slug[:cut]
        slug = slug.strip("_-")
        return slug or fallback

    def handle(self, taxonomy: str, metric_name: str, selector: dict) -> str:
        """``{taxonomy}/{selector_slug}_{selector_key}/{metric_name}``."""
        if "/" in metric_name:
            raise ValueError(
                f"metric {metric_name!r} cannot contain '/': it names a directory"
            )
        if "/" in taxonomy:
            raise ValueError(f"taxonomy {taxonomy!r} cannot contain '/'")
        slug = self.selector_slug(selector, fallback=taxonomy)
        return f"{taxonomy}/{slug}_{self.selector_key(selector)}/{metric_name}"

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def _leaf_dir(self, handle: str) -> Path:
        return self._pairs_dir / handle

    def _surrogate_dir(self, handle: str) -> Path:
        """Where ``meta.json`` lives — one level above the metric leaf."""
        return self._leaf_dir(handle).parent

    @property
    def index_path(self) -> Path:
        return self._pairs_dir / "index.json"

    def exists(self, handle: str) -> bool:
        return (self._leaf_dir(handle) / "pairs.json").exists()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_meta(self, handle: str) -> dict:
        """The surrogate's ``meta.json``, or ``{}`` when it has never been written.

        ``{}`` is what makes "verify identity on every read and write"
        satisfiable on the *first* write: :meth:`save_pairs` treats an empty meta
        as "first write" and records the models block rather than verifying
        against it.  Only a present block is compared.
        """
        path = self._surrogate_dir(handle) / "meta.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def load_pairs(self, handle: str, pair_ids=None) -> dict[str, float]:
        """Stored distances for *pair_ids*; **misses are omitted**, not raised.

        A missing pair is the normal case — a pair never computed, or a model
        new to this handle — and assembly treats it as work to do.  Passing
        ``None`` returns everything stored under the handle.
        """
        path = self._leaf_dir(handle) / "pairs.json"
        if not path.exists():
            return {}
        try:
            stored = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        if pair_ids is None:
            found = {k: float(v) for k, v in stored.items()}
        else:
            wanted = set(pair_ids)
            found = {k: float(v) for k, v in stored.items() if k in wanted}
        self.hits += len(found)
        return found

    def load_index(self) -> dict[str, dict]:
        """The catalogue.  Recomputable from the tree; no cache hit depends on it."""
        if not self.index_path.exists():
            return {}
        try:
            return json.loads(self.index_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def list_handles(self) -> list[str]:
        """Every handle with a ``pairs.json``, by walking the tree.

        Walks rather than reading ``index.json``, so a stale or deleted
        catalogue cannot hide stored work.
        """
        if not self._pairs_dir.exists():
            return []
        out = []
        for path in sorted(self._pairs_dir.rglob("pairs.json")):
            rel = path.parent.relative_to(self._pairs_dir)
            if len(rel.parts) == 3:
                out.append("/".join(rel.parts))
        return out

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_pairs(self, handle: str, distances: dict, models, selector: dict,
                   label: str | None = None) -> None:
        """Merge *distances* into the handle, enforcing G2 and G4.

        Whatever is handed in is written.  **Partial batches are expected and
        legitimate:** pairs are independent, so a partially filled ``pairs.json``
        is not an inconsistent artifact but a smaller correct one,
        indistinguishable from a handle that has not seen those models yet.

        *models* is the per-model identity list — ``model_id``,
        ``artifact_path``, ``surrogate_hash`` and ``draw`` — as
        ``src.analysis.comparison._model_identity`` builds it.

        **Two locks, never nested.**  The pairs and ``meta.json`` are written
        under the *surrogate* lock, which is then released before the *root*
        lock is taken to merge ``index.json``.  ``index.json`` sits at the root
        of the stage and is shared by every surrogate, so two writers under
        different surrogates hold different surrogate locks and would both
        read-modify-write it — exactly the corruption
        ``CollectionCache._update_index`` exists to prevent.  Acquiring the root
        lock while holding a surrogate lock is the one ordering that could
        deadlock against a future writer doing the reverse, and there is no
        reason to hold both.
        """
        from filelock import FileLock

        models = [dict(m) for m in models]
        leaf_dir = self._leaf_dir(handle)
        surrogate_dir = self._surrogate_dir(handle)
        surrogate_dir.mkdir(parents=True, exist_ok=True)

        with FileLock(str(surrogate_dir / "meta.lock")):
            meta = self.load_meta(handle)
            block = dict(meta.get("models") or {})
            merged = _merge_models(block, models)      # G2
            _check_one_draw(merged)                    # G4

            meta = {
                "schema_version": "1",
                "taxonomy": handle.split("/")[0],
                "selector_key": self.selector_key(selector),
                "selector": dict(selector or {}),
                "label": label if label is not None else meta.get("label"),
                # A growing superset of every model ever seen at this handle, not
                # a description of any one run. A reader must not infer a
                # collection from it; what a run covered is recorded by the
                # caller, not by the store.
                "models": merged,
                "updated": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_json(surrogate_dir / "meta.json", meta)

            leaf_dir.mkdir(parents=True, exist_ok=True)
            hits_before = self.hits
            stored = self.load_pairs(handle)
            self.hits = hits_before          # a read-modify-write is not reuse
            stored.update({k: float(v) for k, v in distances.items()})
            self.misses += len(distances)
            _atomic_json(leaf_dir / "pairs.json", stored)
            n_pairs = len(stored)

        # Surrogate lock released. Now the root lock, for the shared catalogue.
        self._update_index(handle, meta, n_pairs)

    def _update_index(self, handle: str, meta: dict, n_pairs: int) -> None:
        """Merge one perspective's summary into ``index.json``, atomically."""
        from filelock import FileLock

        self._pairs_dir.mkdir(parents=True, exist_ok=True)
        taxonomy, surrogate, metric = handle.split("/")
        slug, _, key = surrogate.rpartition("_")
        record = {
            "taxonomy": taxonomy,
            "metric": metric,
            "selector_key": key,
            "selector_slug": slug,
            "label": meta.get("label"),
            "n_models": len(meta.get("models") or {}),
            "n_pairs": n_pairs,
            "updated": meta.get("updated"),
        }
        with FileLock(str(self._pairs_dir / "index.lock")):
            index = self.load_index()
            index[handle] = record
            _atomic_json(self.index_path, index)


# ── helpers ───────────────────────────────────────────────────────────────────


def _atomic_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _layers_token(value) -> str:
    """``L{lo}-{hi}`` for a contiguous ascending list, else ``L{n}x``."""
    if isinstance(value, dict):                 # {"input": [0, ..., 28]}
        value = next(iter(value.values()), None)
    if isinstance(value, int):
        return f"L{value}"
    if not isinstance(value, (list, tuple)) or not value:
        return f"L{value}" if value is not None else ""
    nums = [v for v in value if isinstance(v, int)]
    if len(nums) != len(value):
        return f"L{len(value)}x"
    if all(b - a == 1 for a, b in zip(nums, nums[1:])):
        return f"L{nums[0]}-{nums[-1]}"
    return f"L{len(nums)}x"


def _merge_models(block: dict, models: list[dict]) -> dict:
    """G2 — artifact identity must agree, with all three cases defined.

    ============  ==========================================  ==========
    case          meaning                                     action
    ============  ==========================================  ==========
    identical     already computed against these tensors      proceed
    differing     stored pairs were built from other tensors  **raise**
    absent        a model new to this handle                  **append**
    ============  ==========================================  ==========

    Appending is what makes the store incremental — a 17th model against a warm
    16-model handle costs 16 new distances, not 136 — and it is why the model
    set is deliberately not part of the handle.  The safety that
    ``collection_key`` bought by hashing ``artifact_path`` is recovered here, by
    recording the models and refusing on mismatch.

    A stored model *absent from the request* is likewise not an error: that is
    the normal case for any subset, and it is what makes subsetting free.
    """
    merged = dict(block)
    for model in models:
        mid = model.get("model_id")
        if mid is None:
            raise ValueError(f"a model identity has no model_id: {model!r}")
        incoming = {
            "artifact_path": model.get("artifact_path"),
            "surrogate_hash": model.get("surrogate_hash"),
            "draw": model.get("draw"),
        }
        stored = merged.get(mid)
        if stored is None:
            merged[mid] = incoming
            continue
        for field in ("artifact_path", "surrogate_hash"):
            if stored.get(field) != incoming.get(field):
                raise ValueError(
                    f"model {mid!r} presents {field}={incoming.get(field)!r} but "
                    f"the stored pairs at this handle were built from "
                    f"{field}={stored.get(field)!r}. The distances on disk "
                    "describe different tensors than the caller holds; delete "
                    "the handle or key it apart rather than mixing them."
                )
        merged[mid] = {**stored, **{k: v for k, v in incoming.items()
                                    if v is not None or k not in stored}}
    return merged


def _check_one_draw(models: dict) -> None:
    """G4 — models under one handle share one query draw, by default.

    The hazard is narrower than "different draws are incomparable": ``pair_id``
    is built from ``model_id`` alone, so one model under two draws yields one
    pair id and two genuinely different distances collide on it.  The guard
    exists to stop that collision, not to forbid the comparison — which is why
    ``pair_id`` takes a conditional *qualifier* from the outset.

    Levels that write no draw token record ``draw: null`` and are **exempt**,
    not vacuously passing: structural has no query draw at all, and
    ``02_dataset_embeddings`` folds the draw into a hash.  For those the draw is
    still inside ``artifact_path``, which G2 compares byte for byte.
    """
    seen: dict[str, list[str]] = {}
    for mid, entry in models.items():
        draw = entry.get("draw")
        if draw is None:
            continue
        token = json.dumps(draw, sort_keys=True)
        seen.setdefault(token, []).append(mid)
    if len(seen) > 1:
        detail = "; ".join(f"{sorted(v)} under {k}" for k, v in sorted(seen.items()))
        raise ValueError(
            "models under one handle were extracted against different query "
            f"draws: {detail}. pair_id is built from model_id alone, so two "
            "such distances would collide on one key. Key them apart with a "
            "pair_id qualifier, or use separate handles."
        )
