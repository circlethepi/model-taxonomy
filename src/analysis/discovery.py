"""What is actually in the shared cache, and how the pieces line up.

The individual caches can each list their own contents —
:meth:`~src.cache.lora_cache.LoRACache.list_adapters`,
:meth:`~src.cache.dataset_embedding_cache.DatasetEmbeddingCache.list_recipes`,
and so on — but a comparison needs the *join*: which models have structural data
**and** dataset-embedding data, what recipe each was trained on, and which
``(n_samples, seed)`` group each belongs to.  Nothing assembled that, so every
analysis re-derived it by hand from directory names.

The join key is ``recipe_hash``.  ``scripts/finetune_lora.py`` writes an
``experiment_meta.json`` beside every adapter recording the exact hash of the
recipe it trained on, and :class:`DatasetEmbeddingCache` stores each recipe under
that same hash — so the correspondence is recorded, not inferred.  Only when the
file is missing do we fall back to
:func:`src.analysis.identity.recipe_id_for`, which parses the directory name.

Everything here is read-only and reads **only small JSON** — never a tensor, never
a model.  Recipes are resolved just for the hashes the discovered adapters
actually reference, so scanning a cache holding a four-figure number of recipes
costs a handful of reads rather than one per recipe.  Pass
``scan_all_recipes=True`` to enumerate the rest.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Sequence

from src.cache._draw import draw_name, parse_draw_name
from src.core.protocols import ModelID

__all__ = ["CacheEntry", "CacheIndex", "scan_cache"]

# scripts/_utils.expand_dataset_n_samples then expand_dataset_seeds build dataset
# names as "{base}_n{n_samples}_s{seed:02d}".  This is that convention read back,
# and it is deliberately not yahoo-specific.
_DATASET_NAME_RE = re.compile(r"^(?P<mixture>.+?)_n(?P<n>\d+)_s(?P<seed>\d+)$")

# scripts/_utils.adapter_dir: "{dataset_name}_r{rank}_i{init_seed:02d}[_b{samples_seen}]",
# with the _i suffix absent on adapters trained before it was introduced and the _b
# suffix present only on adapters trained under a sample budget.
_ADAPTER_DIR_RE = re.compile(
    r"^(?P<name>.+?)_r(?P<rank>\d+)(?:_i(?P<init>\d+))?(?:_b(?P<budget>\d+))?$"
)

# Order is load-bearing: summary() builds its per-model flag string by walking this
# tuple and emitting each name's first letter, so position i here is position i in
# the flag string and letter i of the "WRDSB" header.  Append, never insert.
_TAXONOMY_AVAILABILITY = (
    "structural_weights",
    "structural_repr",
    "dataset_embedding",
    "sampled_rows",
    "behavioral_repr",
    "functional_repr",
)

#: Flag letter per token, for the compact availability column in summary().
#: Explicit rather than ``name[0].upper()``: that rule gives three of these five
#: tokens the letter "S" (structural_weights, structural_repr, sampled_rows), so a
#: fully-available model printed "SSDS" under a header advertising "WRDS" — the
#: header could not be used to read the column it labelled.  Keep this in sync with
#: the tuple above; _AVAILABILITY_HEADER is derived from it so they cannot drift.
_AVAILABILITY_LETTERS = {
    "structural_weights": "W",
    "structural_repr": "R",
    "dataset_embedding": "D",
    "sampled_rows": "S",
    "behavioral_repr": "B",
    "functional_repr": "F",
}
_AVAILABILITY_HEADER = "".join(_AVAILABILITY_LETTERS[n] for n in _TAXONOMY_AVAILABILITY)


@dataclass
class CacheEntry:
    """One model in the cache, with everything needed to place it in a comparison."""

    model_id: ModelID                 # path to the adapter directory, as a string
    adapter_name: str                 # directory basename
    base_model_id: str | None = None
    adapter_dir: Path | None = None

    recipe_id: str | None = None      # experiment_meta.json "dataset_name"
    recipe_hash: str | None = None
    recipe: dict | None = None        # parsed dataset_embeddings/{hash}/recipe.json

    mixture: str | None = None        # recipe_id minus the _n{n}_s{seed} suffix
    n_samples: int | None = None
    seed: int | None = None
    # How much training the adapter got, as distinct from how large its dataset was.
    # Under a sample budget the two come apart: n_samples is the draw, samples_seen is
    # the total the model was shown, and n_epochs is the ratio (fractional when the
    # budget does not land on a whole pass).
    samples_seen: int | None = None
    n_epochs: float | None = None
    lora_rank: int | None = None
    lora_init_seed: int | None = None

    available: dict[str, bool] = field(default_factory=dict)
    embedder_hashes: list[str] = field(default_factory=list)

    def has(self, *names: str) -> bool:
        """True when every named artifact is present for this entry."""
        return all(self.available.get(n, False) for n in names)


class CacheIndex:
    """A set of :class:`CacheEntry` with the filtering and grouping a comparison needs."""

    def __init__(self, entries: Sequence[CacheEntry], cache_root: Path | None = None) -> None:
        self.entries = list(entries)
        self.cache_root = cache_root

    # ── container protocol ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[CacheEntry]:
        return iter(self.entries)

    def __repr__(self) -> str:  # pragma: no cover - display only
        mixtures = sorted({e.mixture for e in self.entries if e.mixture})
        seen = sorted({e.samples_seen for e in self.entries if e.samples_seen})
        return (
            f"CacheIndex(n_entries={len(self.entries)}, "
            f"n_mixtures={len(mixtures)}, "
            f"n_values={sorted({e.n_samples for e in self.entries if e.n_samples})}, "
            f"seeds={sorted({e.seed for e in self.entries if e.seed is not None})}, "
            f"samples_seen={seen})"
        )

    # ── views ─────────────────────────────────────────────────────────────────

    @property
    def model_ids(self) -> list[ModelID]:
        return [e.model_id for e in self.entries]

    def recipes(self, key: str = "recipe_id") -> dict[str, dict]:
        """``{identifier: recipe dict}`` for entries whose recipe was resolved.

        *key* selects the identifier: ``"recipe_id"`` (the default) matches what
        :func:`src.analysis.identity.recipe_id_for` produces from an adapter path,
        which is the namespace a cross-taxonomy comparison relabels into.
        ``"model_id"`` keys by adapter path instead.
        """
        if key not in ("recipe_id", "model_id"):
            raise ValueError(f"key must be 'recipe_id' or 'model_id', got {key!r}")
        out: dict[str, dict] = {}
        for e in self.entries:
            ident = getattr(e, key)
            if ident is not None and e.recipe is not None:
                out[ident] = e.recipe
        return out

    def entry_for(self, model_id: ModelID) -> CacheEntry:
        for e in self.entries:
            if e.model_id == model_id or e.recipe_id == model_id:
                return e
        raise KeyError(f"{model_id!r} not in this index")

    # ── filtering and grouping ────────────────────────────────────────────────

    def filter(self, **criteria: Any) -> "CacheIndex":
        """Keep entries matching every criterion.

        Each value is either a scalar or a collection of accepted values, so
        ``index.filter(n_samples=1000)`` and ``index.filter(n_samples=[10, 100])``
        both work.  Unknown field names raise rather than silently matching
        nothing, which is otherwise a very quiet way to get an empty result.
        """
        valid = set(CacheEntry.__dataclass_fields__)
        unknown = set(criteria) - valid
        if unknown:
            raise ValueError(
                f"unknown CacheEntry field(s): {sorted(unknown)}. "
                f"Available: {sorted(valid)}"
            )

        def matches(entry: CacheEntry) -> bool:
            for name, wanted in criteria.items():
                value = getattr(entry, name)
                if isinstance(wanted, (list, tuple, set, frozenset)):
                    if value not in wanted:
                        return False
                elif value != wanted:
                    return False
            return True

        return CacheIndex([e for e in self.entries if matches(e)], self.cache_root)

    def with_available(self, *names: str) -> "CacheIndex":
        """Keep entries that have every named artifact."""
        return CacheIndex([e for e in self.entries if e.has(*names)], self.cache_root)

    def slices(self, by: Sequence[str] = ("n_samples", "seed")) -> dict[tuple, "CacheIndex"]:
        """Group entries into sub-collections keyed by the named fields.

        The groupings a taxonomy comparison wants, all from this one method:

        * ``("n_samples", "seed")`` — one comparison per experimental cell, where
          the anchors are unambiguous.
        * ``("n_samples",)`` — one per sample size, comparing seeds within it.
        * ``("seed",)`` — one per seed, comparing sample sizes within it.
        * ``()`` — a single pooled group over everything.

        Entries missing any grouping field are dropped, since they cannot be
        placed; groups are returned in sorted key order.
        """
        by = tuple(by)
        valid = set(CacheEntry.__dataclass_fields__)
        unknown = set(by) - valid
        if unknown:
            raise ValueError(f"unknown CacheEntry field(s) to slice by: {sorted(unknown)}")

        if not by:
            return {(): CacheIndex(self.entries, self.cache_root)}

        groups: dict[tuple, list[CacheEntry]] = defaultdict(list)
        for e in self.entries:
            key = tuple(getattr(e, name) for name in by)
            if any(v is None for v in key):
                continue
            groups[key].append(e)

        return {
            key: CacheIndex(groups[key], self.cache_root) for key in sorted(groups)
        }

    def slice_label(self, key: tuple, by: Sequence[str] = ("n_samples", "seed")) -> str:
        """Filesystem-safe label for a slice key, e.g. ``n1000_s00`` or ``pooled``."""
        if not key:
            return "pooled"
        abbrev = {"n_samples": "n", "seed": "s", "samples_seen": "b", "n_epochs": "e"}
        parts = []
        for name, value in zip(by, key):
            prefix = abbrev.get(name, f"{name}-")
            parts.append(
                f"{prefix}{value:02d}" if name == "seed" else f"{prefix}{value}"
            )
        return "_".join(parts)

    # ── reporting ─────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Coverage table: which mixtures exist at which sizes and seeds, and
        which artifacts each has.

        The quickest way to find out why a comparison came back with fewer models
        than expected.
        """
        if not self.entries:
            return "CacheIndex is empty — no adapters with experiment_meta.json found."

        rows = []
        for e in sorted(
            self.entries,
            key=lambda x: (x.mixture or "", x.n_samples or 0, x.seed or 0),
        ):
            flags = "".join(
                _AVAILABILITY_LETTERS[name] if e.available.get(name) else "-"
                for name in _TAXONOMY_AVAILABILITY
            )
            rows.append(
                (
                    e.mixture or e.adapter_name,
                    "-" if e.n_samples is None else str(e.n_samples),
                    "-" if e.seed is None else f"{e.seed:02d}",
                    # Two adapters differing only in training length are otherwise
                    # indistinguishable here, and look like duplicates.
                    "-" if e.samples_seen is None else str(e.samples_seen),
                    "-" if e.lora_rank is None else str(e.lora_rank),
                    flags,
                    "yes" if e.recipe is not None else "no",
                )
            )

        headers = ("mixture", "n", "seed", "seen", "rank", _AVAILABILITY_HEADER, "recipe")
        widths = [
            max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)
        ]
        line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
        out = [
            f"{len(self.entries)} adapter(s) under {self.cache_root}",
            "",
            line,
            "-" * len(line),
        ]
        out += ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
        out += [
            "",
            f"{_AVAILABILITY_HEADER} = " + " / ".join(
                f"{_AVAILABILITY_LETTERS[n]}={n}" for n in _TAXONOMY_AVAILABILITY
            ),
        ]
        return "\n".join(out)


# ── the scan ───────────────────────────────────────────────────────────────────

def scan_cache(
    cache_root: Path | str,
    base_model_id: str | None = None,
    resolve_recipes: bool = True,
    scan_all_recipes: bool = False,
    behavioral_draw: dict | None = None,
    functional_draw: dict | None = None,
) -> CacheIndex:
    """Walk the shared cache and join adapters to the recipes they were trained on.

    Parameters
    ----------
    cache_root:
        The shared cache root, e.g. ``results/shared_cache``.
    base_model_id:
        Restrict to adapters of one base model.  Default: every base model present.
    resolve_recipes:
        Read ``recipe.json`` for the hashes the discovered adapters reference.
        Cheap — one small JSON per distinct recipe, and the ground-truth simplex
        cannot be built without it.
    scan_all_recipes:
        Additionally enumerate *every* recipe in the dataset-embedding cache, not
        just the referenced ones.  Off by default because that is one read per
        recipe and the cache can hold thousands.
    behavioral_draw:
        Which query draw the ``behavioral_repr`` token should report on, as a
        ``{recipe_hash, n_samples, seed}`` dict.

        **The meaning of the token depends on this argument, so read it before
        relying on it.**  Given a draw the token is exact: True means *this model
        has a representation over this exact query draw*.  Omitted, it degrades
        to "some behavioral representation exists for this model, under some
        draw" — weaker than it looks, because a representation generated from a
        different query set is not interchangeable with the one a caller probably
        wants.  Pass it whenever the answer will select models for a comparison.

        This replaced ``behavioral_config_hash``, and the old keyword is **gone
        rather than deprecated**.  The tree it addressed no longer exists, so a
        ``TypeError`` is the correct immediate failure; an alias that quietly
        did nothing would be the same class of silent bug this layout change
        exists to remove.
    functional_draw:
        The same, for ``functional_repr``, with the same caveat.
    """
    root = Path(cache_root)
    adapters_root = root / "03_adapters"
    embeddings_root = root / "02_dataset_embeddings"
    sampled_root = root / "01_datasets"

    if not adapters_root.exists():
        return CacheIndex([], root)

    from src.cache.activation_cache import ActivationCache
    from src.cache.dataset_embedding_cache import DatasetEmbeddingCache
    from src.cache.generated_text_cache import GeneratedTextCache
    from src.cache.lora_cache import LoRACache

    lora_cache = LoRACache(root)
    de_cache = DatasetEmbeddingCache(root)
    gen_cache = GeneratedTextCache(root)
    act_cache = ActivationCache(root)

    slugs = (
        [base_model_id.replace("/", "--")]
        if base_model_id
        else [d.name for d in sorted(adapters_root.iterdir()) if d.is_dir()]
    )

    entries: list[CacheEntry] = []
    recipe_cache: dict[str, dict | None] = {}

    for slug in slugs:
        base_dir = adapters_root / slug
        if not base_dir.is_dir():
            continue
        base_id = slug.replace("--", "/")

        for adapter_dir in sorted(base_dir.iterdir()):
            if not adapter_dir.is_dir():
                continue
            meta = _read_json(adapter_dir / "experiment_meta.json")
            if meta is None:
                # No provenance record: this is a raw or hand-placed adapter, and
                # guessing its recipe from the name alone would put an unverified
                # ground truth into the comparison.  Skipped deliberately.
                continue

            entry = _build_entry(
                adapter_dir=adapter_dir,
                base_id=meta.get("base_model_id") or base_id,
                meta=meta,
            )

            if entry.recipe_hash and resolve_recipes:
                if entry.recipe_hash not in recipe_cache:
                    recipe_cache[entry.recipe_hash] = _resolve_recipe(
                        entry.recipe_hash, sampled_root, embeddings_root
                    )
                entry.recipe = recipe_cache[entry.recipe_hash]

            # Per draw, not per recipe.  A recipe hash is content-addressed, so
            # asking it alone answers "was this mixture ever embedded, at any n
            # and seed?" — which is not what the availability flag below means.
            if entry.recipe_hash and entry.n_samples is not None and entry.seed is not None:
                entry.embedder_hashes = de_cache.list_embedder_hashes(
                    entry.recipe_hash, entry.n_samples, entry.seed
                )

            entry.available = {
                "structural_weights": (adapter_dir / "adapter_model.safetensors").exists(),
                "structural_repr": bool(
                    lora_cache.list_representations(entry.base_model_id, adapter_dir.name)
                ),
                "dataset_embedding": bool(entry.embedder_hashes),
                "sampled_rows": _sampled_rows_exist(
                    sampled_root, entry.recipe_hash, entry.n_samples, entry.seed
                ),
                # Ask the cache where its own files live rather than rebuilding the
                # path here — see _sampled_rows_exist below for why that matters.
                "behavioral_repr": _draw_keyed_repr_exists(
                    gen_cache, entry.base_model_id, entry.model_id, behavioral_draw
                ),
                "functional_repr": _draw_keyed_repr_exists(
                    act_cache, entry.base_model_id, entry.model_id, functional_draw
                ),
            }
            entries.append(entry)

    if scan_all_recipes:
        _attach_unreferenced_recipes(entries, de_cache, sampled_root, embeddings_root)

    return CacheIndex(entries, root)


def _build_entry(adapter_dir: Path, base_id: str, meta: dict) -> CacheEntry:
    """Turn one adapter directory plus its experiment_meta.json into a CacheEntry."""
    lora_cfg = meta.get("lora_config", {}) or {}
    training = meta.get("training", {}) or {}

    recipe_id = meta.get("dataset_name")
    if not recipe_id:
        # Fall back to the directory name, the same route identity.recipe_id_for takes.
        m = _ADAPTER_DIR_RE.match(adapter_dir.name)
        recipe_id = m.group("name") if m else None

    mixture, n_samples, seed = _parse_dataset_name(recipe_id)
    if n_samples is None:
        # The name did not carry a size; the training record still does.
        n_samples = training.get("n_samples")
    if seed is None:
        # Likewise for the sampling seed.  finetune_lora.py records it because the
        # recipe no longer can: the hash is content-addressed and its name carries
        # neither n nor seed.
        seed = training.get("seed")
    if n_samples is None or seed is None:
        # Older adapters predate both conventions, but their directory name is
        # {expanded_block_name}_r{rank}_i{init} and the block name did carry them.
        m = _ADAPTER_DIR_RE.match(adapter_dir.name)
        if m:
            _, dir_n, dir_seed = _parse_dataset_name(m.group("name"))
            n_samples = n_samples if n_samples is not None else dir_n
            seed = seed if seed is not None else dir_seed

    # Training length.  samples_seen is written directly by adapters trained after the
    # sample budget existed; for every adapter predating it, n_samples * n_epochs is
    # exactly the same quantity, so the axis is populated for the whole cache without a
    # migration or a retrain.  The directory suffix is the last resort.
    n_epochs = training.get("n_epochs")
    samples_seen = training.get("samples_seen")
    if samples_seen is None and n_samples is not None and n_epochs is not None:
        samples_seen = int(n_samples * n_epochs)
    if samples_seen is None:
        m = _ADAPTER_DIR_RE.match(adapter_dir.name)
        if m and m.group("budget") is not None:
            samples_seen = int(m.group("budget"))
    if n_epochs is None and samples_seen is not None and n_samples:
        n_epochs = samples_seen / n_samples

    rank = lora_cfg.get("lora_rank")
    init_seed = lora_cfg.get("lora_init_seed")
    if rank is None:
        m = _ADAPTER_DIR_RE.match(adapter_dir.name)
        if m:
            rank = int(m.group("rank"))
            init_seed = int(m.group("init")) if m.group("init") is not None else None

    return CacheEntry(
        model_id=str(adapter_dir),
        adapter_name=adapter_dir.name,
        base_model_id=base_id,
        adapter_dir=adapter_dir,
        recipe_id=recipe_id,
        recipe_hash=meta.get("recipe_hash"),
        mixture=mixture,
        n_samples=n_samples,
        seed=seed,
        samples_seen=samples_seen,
        n_epochs=n_epochs,
        lora_rank=rank,
        lora_init_seed=init_seed,
    )


def _parse_dataset_name(name: str | None) -> tuple[str | None, int | None, int | None]:
    """Split ``{mixture}_n{n}_s{seed}`` into its parts; ``(name, None, None)`` if it does not match."""
    if not name:
        return None, None, None
    m = _DATASET_NAME_RE.match(name)
    if not m:
        return name, None, None
    return m.group("mixture"), int(m.group("n")), int(m.group("seed"))


def _sampled_rows_exist(
    sampled_root: Path, recipe_hash: str | None, n_samples: int | None, seed: int | None
) -> bool:
    if not recipe_hash or n_samples is None or seed is None:
        return False
    # Still rebuilds SampledDatasetCache's layout rather than asking it — but the
    # part that actually drifted, the draw token, now comes from the one function
    # that owns it.  Constructing the cache instead would be better still, except
    # that its __init__ mkdirs the root, and a predicate used by a read-only scan
    # must not create directories as a side effect.
    return (sampled_root / recipe_hash / f"{draw_name(n_samples, seed)}.json").exists()


def _draw_keyed_repr_exists(
    cache, base_model_id: str | None, model_id: str, draw: dict | None
) -> bool:
    """Whether this model has a stored representation — exactly, or coarsely.

    One helper for both inference levels: since ``05_generated`` was re-keyed to
    match ``04_activations``, :class:`~src.cache._draw_keyed.DrawKeyedCache` gives
    them the same ``has_draw``/``has_any`` protocol, so asking "does this model
    have a functional representation?" and "…a behavioral one?" is the same
    question against a different cache.

    Both branches ask the cache where its own files live rather than rebuilding
    the path.  ``_sampled_rows_exist`` above is the counterexample already in
    this codebase: it hardcodes ``SampledDatasetCache``'s layout, so the two are
    free to drift and the failure mode is silent — every write succeeds while the
    cache reads as empty.  The behavioral cache spent its whole life in exactly
    that state; see ``docs/notes/TODO.md`` item 13.
    """
    if not base_model_id:
        return False
    if draw is None:
        return cache.has_any(base_model_id, model_id)
    return cache.has_draw(base_model_id, model_id, draw)


def _attach_unreferenced_recipes(
    entries: list[CacheEntry], de_cache, sampled_root: Path, embeddings_root: Path
) -> None:
    """Append recipe-only entries for recipes no discovered adapter was trained on.

    These are dataset-level rows — held-out probe sets, or mixtures embedded but
    never fine-tuned on.  They carry no adapter, so only the dataset-embedding
    taxonomy can use them.
    """
    seen = {e.recipe_hash for e in entries if e.recipe_hash}
    for recipe_hash in de_cache.list_recipes():
        if recipe_hash in seen:
            continue
        recipe = _resolve_recipe(recipe_hash, sampled_root, embeddings_root)
        if recipe is None:
            continue
        mixture = recipe.get("name")
        # One entry per draw, not per recipe.  The hash is content-addressed, so it
        # identifies a mixture and says nothing about n or seed — those live in the draw
        # filenames beside it, and a mixture typically has a hundred of them.  Reading
        # them off disk also makes these entries describe draws that exist rather than
        # draws a name implied.
        for n_samples, seed in sorted(_draws_for(sampled_root, recipe_hash)):
            recipe_id = f"{mixture}_n{n_samples}_s{seed:02d}" if mixture else recipe_hash
            entries.append(
                CacheEntry(
                    model_id=recipe_id,
                    adapter_name=recipe_id,
                    recipe_id=recipe_id,
                    recipe_hash=recipe_hash,
                    recipe=recipe,
                    mixture=mixture,
                    n_samples=n_samples,
                    seed=seed,
                    embedder_hashes=de_cache.list_embedder_hashes(
                        recipe_hash, n_samples, seed
                    ),
                    available={
                        "structural_weights": False,
                        "structural_repr": False,
                        "dataset_embedding": True,
                        "sampled_rows": True,
                        # These entries are recipes, not models — there is no adapter
                        # to run inference on, so neither inference-based level can
                        # ever be available for them.
                        "behavioral_repr": False,
                        "functional_repr": False,
                    },
                )
            )


def _draws_for(sampled_root: Path, recipe_hash: str) -> list[tuple[int, int]]:
    """The ``(n_samples, seed)`` draws cached under a recipe hash."""
    directory = sampled_root / recipe_hash
    if not directory.is_dir():
        return []
    draws = []
    for path in directory.glob("n*_s*.json"):
        parsed = parse_draw_name(path.stem)
        if parsed:
            draws.append(parsed)
    return draws


def _resolve_recipe(
    recipe_hash: str, sampled_root: Path, embeddings_root: Path
) -> dict | None:
    """The mixing spec a recipe hash identifies, or None if it is nowhere on disk.

    ``01_datasets`` is the authoritative home: a recipe hash names a dataset, and the
    dataset's own directory is where its definition belongs.  The dataset-embedding
    cache is consulted second only because it writes its own copy, so a recipe that
    was embedded without ever going through the sample cache still resolves.  Before
    ``01_datasets`` existed the embedding cache was the *only* source, which meant any
    recipe sampled but never embedded could not be resolved at all — that is the
    defect this ordering fixes, not a layout fallback.
    """
    recipe = _read_json(sampled_root / recipe_hash / "recipe.json")
    if recipe is not None:
        return recipe
    return _read_json(embeddings_root / recipe_hash / "recipe.json")


def _read_json(path: Path) -> dict | None:
    try:
        if path.is_file():
            return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        pass  # unreadable or malformed — treat as absent rather than aborting the scan
    return None
