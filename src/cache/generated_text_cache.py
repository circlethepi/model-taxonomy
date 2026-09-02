from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from src.utils.atomic import atomic_path
from src.cache._draw_keyed import (
    DrawKeyedCache,
    _atomic_write_json,
    _row_normalize,
    adapter_slug,
)
from src.core.representation import ModelRepresentation

__all__ = ["GeneratedTextCache", "_GEN_RE"]


#: Embedding filenames look like
#: ``generation128_8r_3f9c1a2b_5191ad734b81daff.safetensors``: the generation
#: mode (carrying its token budget), how many replicates were drawn per query,
#: the sampling settings that drew them, and the embedder that turned the text
#: into vectors.  Peer of ``activation_cache._ACT_RE`` — everything that changes
#: the stored numbers is in the name, so a directory listing is a complete
#: description of what was computed.
#:
#: ``_{r}r`` and ``_{sampling}`` were added together and for one reason: once
#: decoding samples, two runs over the same draw at different temperatures (or
#: different replicate counts) produce genuinely different text.  :meth:`save` is
#: idempotent on filename, so without both components in the name the second run
#: is a silent no-op that reuses the first entry's numbers — the hazard this
#: class documents for ``torch_dtype``, but on an axis that changes the result
#: far more than dtype does.
#:
#: If TODO item 12 gives behavioral a ``representation:`` knob, it extends this
#: name rather than becoming a side file; keep the regex and the writer in step.
_GEN_RE = re.compile(
    r"^(?P<mode>generation\d+)"
    r"_(?P<replicates>\d+)r"
    r"_(?P<sampling>[0-9a-f]{8})"
    r"_(?P<embedder>[0-9a-f]{16})$"
)


def _closure_scalars(closure: dict | None) -> dict | None:
    """The summary half of a think-closure record, without the per-row detail.

    The run record is a provenance side file that someone reads to answer "what
    happened in this run"; the full ``closed_at`` nesting is as long as the
    generations themselves and belongs beside them, not here.
    """
    if not closure:
        return None
    return {k: v for k, v in closure.items() if k != "closed_at"}


class GeneratedTextCache(DrawKeyedCache):
    """Cache for behavioral representations: generated text and its embeddings.

    Directory layout::

        cache_root/05_generated/{base_slug}/{adapter_slug}/{recipe_hash}/n{n}_s{seed}/
            queries.json                                ← query_key + source row indices
            runs/{config_hash}.json                     ← extraction provenance
            generations/{variant_token}.json            ← {model_id, generated_texts}
            embeddings/{variant_token}_{embedder_hash}.safetensors
            surrogates/{surrogate_hash}/
                config.json
                surrogate.safetensors

    where ``variant_token`` is ``generation{max_new_tokens}_{replicates}r_{sampling_hash}``.

    The prefix through ``n{n}_s{seed}`` is **byte-identical** to
    :class:`~src.cache.activation_cache.ActivationCache`'s, because both come
    from :class:`~src.cache._draw_keyed.DrawKeyedCache`.  One model under one
    draw therefore sits at the same coordinates in both stages, and the two trees
    can be read side by side.

    **This layout replaced a run-wise one keyed ``{config_hash}/`` with files
    named by a hash of the adapter's full path.**  That made an entry reachable
    only from the working directory the extraction ran in — every write
    succeeded, and the cache read as empty from anywhere else.  It was not a
    latent hazard: the stored IDs were *relative* paths, so scanning the cache
    with an absolute root missed all of them.  ``docs/notes/TODO.md`` item 13 and
    ``scripts/migrate_behavioral_layout.py`` record the migration.

    Text and tensors are split because they are read by different readers.
    Auditing what a model actually generated — the first thing worth doing after
    a run — is then a plain JSON open, with no safetensors load, no numpy and no
    GPU.  :meth:`load` reassembles the two into the single
    :class:`ModelRepresentation` the rest of the pipeline expects, so the split
    is an on-disk detail rather than an API change.

    **Generations are keyed by the generation variant alone, embeddings by that
    variant and the embedder.**  That is the asymmetry that makes a second
    embedder cheap: the text is generated once, and re-embedding it adds one file
    beside the first with no GPU generation pass.  ``ls embeddings/`` answers
    "which embedders have run over this draw?" from a directory listing.

    **A row is one replicate, not one query.**  With ``replicates=R`` the stored
    matrix is ``(n_queries * R, d)`` in query-major order — ``q0r0, q0r1, …,
    q1r0, …`` — which is the order ``model.generate(num_return_sequences=R)``
    returns, and ``generated_texts`` is correspondingly a list of ``R``-element
    lists.  Keeping every replicate rather than averaging at write time follows
    the same rule as TODO item 12's read-time pooling: the mean is recoverable
    from the rows, the spread is not recoverable from the mean.  Ask for
    ``replicate_reduction="mean"`` at read time to get the ``(n_queries, d)``
    shape back.

    **``torch_dtype`` is deliberately not in any filename**, matching
    ``04_activations``.  An fp16 and an fp32 run over the same draw write the
    same names, so the second is a no-op skip rather than a second entry.  It is
    detectable afterwards — each run leaves its own ``runs/{config_hash}.json``
    and the dtype is in there — but it is not prevented.

    **No query text is stored.**  ``(recipe_hash, n_samples, seed)`` determines it
    completely, because ``text_field`` lives in the recipe and so is inside
    ``recipe_hash``.  The old layout duplicated all 64 strings into a
    ``queries.json`` beside every run; ``01_datasets`` was always canonical.
    """

    _STAGE_DIR = "05_generated"
    _ARTIFACT_DIR = "embeddings"

    # ------------------------------------------------------------------
    # Hash helpers
    # ------------------------------------------------------------------

    @classmethod
    def embedder_hash(cls, embedder_config: dict) -> str:
        """16-hex identity of the embedder that produced a set of vectors.

        Deliberately **not**
        :meth:`src.cache.dataset_embedding_cache.DatasetEmbeddingCache.embedder_hash`,
        whose signature also folds in ``representation``, ``n_samples`` and
        ``seed``.  It has to: the dataset level has no draw directory, so the
        draw can only live in the hash.  Here the draw is already a path
        component, and folding it in again would key the entry twice and make
        ``ls embeddings/`` unreadable.

        Keeping the two independent also decouples the stages, so a change to
        the dataset level's hashing cannot move these filenames.
        """
        return cls.config_hash(embedder_config)

    #: The sampling axes that change the generated text.  Fixed and explicit so
    #: that a caller passing a partial dict still hashes to the same thing as one
    #: passing every key, and so adding an axis later is a visible edit here
    #: rather than a silent change of every digest.
    SAMPLING_KEYS = ("do_sample", "temperature", "top_p", "top_k", "generation_seed")

    #: What greedy decoding hashes to.  Every entry written before sampling
    #: existed is greedy, so this is the token
    #: ``scripts/migrate_behavioral_replicates.py`` renames them under.
    GREEDY_SAMPLING = {
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "generation_seed": None,
    }

    @classmethod
    def sampling_hash(cls, sampling: dict) -> str:
        """8-hex identity of the decoding settings that produced a generation.

        Shorter than the 16-hex hashes elsewhere in the cache because it shares a
        filename with two of them and the namespace it has to separate is tiny —
        a handful of decoding settings per project, not a content-addressed space
        of recipes.

        Unknown keys are rejected rather than ignored: a typo'd ``temp`` that
        hashed the same as the default would put two different runs in one entry,
        which is the exact failure this hash exists to prevent.
        """
        unknown = set(sampling) - set(cls.SAMPLING_KEYS)
        if unknown:
            raise ValueError(
                f"unknown sampling key(s) {sorted(unknown)}; expected a subset of "
                f"{list(cls.SAMPLING_KEYS)}"
            )
        canonical = {k: sampling.get(k) for k in cls.SAMPLING_KEYS}
        return cls.config_hash(canonical)[:8]

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def variant_token(
        self, max_new_tokens: int, replicates: int, sampling_hash: str
    ) -> str:
        """``generation{max_new_tokens}_{replicates}r_{sampling_hash}``.

        Deliberately *not* an override of
        :meth:`~src.cache._draw_keyed.DrawKeyedCache.mode_token`, which
        ``04_activations`` shares byte-for-byte — replicates are a behavioral
        concept, and widening the shared spelling would move functional filenames
        for a parameter the functional level does not have.
        ``scripts/check_analysis.py`` asserts the two caches hold the *same*
        ``mode_token`` function object, and that assertion should keep passing.
        """
        if int(replicates) < 1:
            raise ValueError(f"replicates must be >= 1, got {replicates!r}")
        base = self.mode_token("generation", max_new_tokens)
        return f"{base}_{int(replicates)}r_{sampling_hash}"

    def _variant_stem(self, variant_token: str, embedder_hash: str) -> str:
        return f"{variant_token}_{embedder_hash}"

    def generations_path(
        self,
        base_model_id: str,
        adapter_id: str,
        query_key: dict,
        max_new_tokens: int,
        replicates: int,
        sampling_hash: str,
    ) -> Path:
        token = self.variant_token(max_new_tokens, replicates, sampling_hash)
        return (
            self.draw_dir(base_model_id, adapter_id, query_key)
            / "generations"
            / f"{token}.json"
        )

    def embeddings_path(
        self,
        base_model_id: str,
        adapter_id: str,
        query_key: dict,
        max_new_tokens: int,
        replicates: int,
        sampling_hash: str,
        embedder_hash: str,
    ) -> Path:
        token = self.variant_token(max_new_tokens, replicates, sampling_hash)
        return (
            self.draw_dir(base_model_id, adapter_id, query_key)
            / "embeddings"
            / f"{self._variant_stem(token, embedder_hash)}.safetensors"
        )

    # ------------------------------------------------------------------
    # Existence
    # ------------------------------------------------------------------

    def exists(
        self,
        base_model_id: str,
        adapter_id: str,
        query_key: dict,
        max_new_tokens: int,
        replicates: int,
        sampling_hash: str,
        embedder_hash: str,
    ) -> bool:
        return (
            self.embeddings_path(
                base_model_id, adapter_id, query_key, max_new_tokens,
                replicates, sampling_hash, embedder_hash,
            ).exists()
            and self.generations_path(
                base_model_id, adapter_id, query_key, max_new_tokens,
                replicates, sampling_hash,
            ).exists()
        )

    def has_draw(self, base_model_id: str, adapter_id: str, query_key: dict) -> bool:
        """True when this draw holds both an embedding and its generated text.

        Overrides the base to require the generations peer.  An embedding whose
        text is missing is half an entry: :meth:`load` would return a
        representation with an empty ``generated_texts``, which reads as "the
        model generated nothing" rather than as a broken cache.
        """
        if not super().has_draw(base_model_id, adapter_id, query_key):
            return False
        gen_dir = self.draw_dir(base_model_id, adapter_id, query_key) / "generations"
        return gen_dir.exists() and any(gen_dir.glob("*.json"))

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(
        self,
        base_model_id: str,
        adapter_id: str,
        query_key: dict,
        rep: ModelRepresentation,
        *,
        max_new_tokens: int,
        replicates: int,
        sampling: dict,
        embedder_hash: str,
        config: dict | None = None,
        source_indices: list | None = None,
    ) -> None:
        """Atomically write queries, run record, generations and embeddings.

        Idempotent: returns immediately if this model's entry already exists.
        Note the consequence — there is no invalidation path.  Anything inside
        *config* that changes the ``embedder_hash``, the replicate count or the
        ``sampling_hash`` produces a new filename and so a new entry, which is
        the behaviour you want and is legible on disk.  Anything outside them
        (``device``, ``batch_size``, and see the class docstring on
        ``torch_dtype``) silently reuses the existing entry.

        *sampling* is the decoding settings dict, not its hash: the hash goes in
        the filename and the dict is written into the entry, so a stored
        generation says what produced it without a lookup table.
        """
        from filelock import FileLock

        sampling_hash = self.sampling_hash(sampling)
        draw = self.draw_dir(base_model_id, adapter_id, query_key)
        (draw / "generations").mkdir(parents=True, exist_ok=True)
        (draw / "embeddings").mkdir(parents=True, exist_ok=True)

        with FileLock(str(draw / ".lock")):
            if self.exists(
                base_model_id, adapter_id, query_key, max_new_tokens,
                replicates, sampling_hash, embedder_hash,
            ):
                return

            self._write_queries_once(draw, query_key, source_indices)

            metadata = dict(rep.metadata or {})
            generated_texts = metadata.pop("generated_texts", None)

            if config is not None:
                self._write_run_record(
                    draw,
                    config,
                    {
                        "mode": "generation",
                        "max_new_tokens": max_new_tokens,
                        "replicates": replicates,
                        "sampling": dict(sampling),
                        "sampling_hash": sampling_hash,
                        "embedder_hash": embedder_hash,
                        "model_id": rep.model_id,
                        # Provenance the config deliberately excludes so it does
                        # not fragment the cache; see BehavioralTaxonomy.
                        "device_name": metadata.get("device_name"),
                        "batch_size": metadata.get("batch_size"),
                        # Scalars only here; the per-replicate detail lives in
                        # the generations file next to the text it describes.
                        # None for every model without a reasoning block, which
                        # is why the key can be added without touching existing
                        # run records' meaning.
                        "think_closure": _closure_scalars(
                            metadata.get("think_closure")
                        ),
                    },
                )

            # generations/{variant_token}.json — written once per generation
            # variant, then shared by every embedder that runs over this draw.
            gen_path = self.generations_path(
                base_model_id, adapter_id, query_key, max_new_tokens,
                replicates, sampling_hash,
            )
            if not gen_path.exists():
                _atomic_write_json(
                    gen_path,
                    {
                        # 4 = think_closure may be present alongside the texts;
                        # 3 = generated_texts is nested per query; 1-2 were flat
                        # lists from before replicates existed.  load_generations
                        # reads only "generated_texts", so 3-era files still load
                        # and this bump costs no migration.
                        "schema_version": "4",
                        "model_id": rep.model_id,
                        "replicates": replicates,
                        "sampling": dict(sampling),
                        "generated_texts": _nest_texts(generated_texts, replicates),
                        # Nested exactly like generated_texts, so a reader can
                        # zip the two.  None for models with no reasoning block.
                        "think_closure": metadata.get("think_closure"),
                    },
                )

            # embeddings/{variant_token}_{embedder_hash}.safetensors
            meta_bytes = np.frombuffer(
                json.dumps(
                    {
                        "model_id": rep.model_id,
                        "base_model_id": base_model_id,
                        "taxonomy": rep.taxonomy,
                        "query_key": query_key,
                        "embedder_hash": embedder_hash,
                        "max_new_tokens": max_new_tokens,
                        "replicates": replicates,
                        "sampling": dict(sampling),
                        "sampling_hash": sampling_hash,
                        "metadata": metadata,
                    }
                ).encode("utf-8"),
                dtype=np.uint8,
            )
            st_path = self.embeddings_path(
                base_model_id, adapter_id, query_key, max_new_tokens,
                replicates, sampling_hash, embedder_hash,
            )
            _atomic_save_matrix(
                st_path,
                np.ascontiguousarray(rep.matrix.astype(np.float32)),
                meta_bytes,
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_generations(
        self,
        base_model_id: str,
        adapter_id: str,
        query_key: dict,
        max_new_tokens: int,
        replicates: int,
        sampling_hash: str,
    ) -> list[list[str]]:
        """The generated text for one model, without touching the tensors.

        Nested per query: ``texts[q][r]`` is replicate *r* of query *q*.  Entries
        written before replicates existed stored a flat list and are normalized
        to one-element lists here, so a caller sees one shape.
        """
        path = self.generations_path(
            base_model_id, adapter_id, query_key, max_new_tokens,
            replicates, sampling_hash,
        )
        if not path.exists():
            return []
        texts = json.loads(path.read_text()).get("generated_texts", [])
        return [t if isinstance(t, list) else [t] for t in texts]

    def load_matrix(
        self,
        base_model_id: str,
        adapter_id: str,
        query_key: dict,
        max_new_tokens: int,
        replicates: int,
        sampling_hash: str,
        embedder_hash: str,
    ) -> tuple[np.ndarray, dict]:
        """The stored ``(n_queries * replicates, d)`` matrix and its ``_meta_json``."""
        from safetensors.numpy import load_file

        path = self.embeddings_path(
            base_model_id, adapter_id, query_key, max_new_tokens,
            replicates, sampling_hash, embedder_hash,
        )
        if not path.exists():
            raise FileNotFoundError(
                f"no stored behavioral embeddings for {adapter_id} under draw "
                f"{self.draw_name(query_key)} "
                f"({self.variant_token(max_new_tokens, replicates, sampling_hash)}, "
                f"embedder {embedder_hash}) at {path}"
            )
        tensors = load_file(str(path))
        meta = json.loads(tensors["_meta_json"].tobytes().decode("utf-8"))
        return tensors["matrix"], meta

    def load(
        self,
        base_model_id: str,
        adapter_id: str,
        query_key: dict,
        max_new_tokens: int,
        replicates: int,
        sampling_hash: str,
        embedder_hash: str,
        view: str = "matrix",
        normalize: str | bool = "none",
        replicate_reduction: str = "all",
    ) -> ModelRepresentation:
        """Reconstruct a ModelRepresentation, generations folded back in.

        ``metadata["generated_texts"]`` is restored from ``generations/`` so the
        returned object is indistinguishable from one that was never cached.

        The stored ``("matrix", "none", "all")`` triple is returned as-is, which
        is why it is *not* routed through ``surrogates/``: writing a byte-copy of
        the stored matrix back beside it would double this stage's footprint to
        buy nothing.  Anything else goes through the same surrogate mechanism the
        functional level uses, so a view is computed at most once per
        ``(draw, variant, embedder, view, normalize, replicate_reduction)``.

        ``replicate_reduction="mean"`` averages the ``R`` replicates of each
        query, returning the ``(n_queries, d)`` shape a single-sample run
        produced.  It is applied **first**, before normalization and before the
        view: a ``gram`` of the unreduced matrix is a ``(n·R, n·R)`` kernel whose
        rows are replicate-to-replicate similarities, and averaging those is not
        the same quantity as averaging the underlying vectors.
        """
        normalize = self.canon_normalize(normalize)
        replicate_reduction = self.canon_replicate_reduction(replicate_reduction)
        matrix, meta = self.load_matrix(
            base_model_id, adapter_id, query_key, max_new_tokens,
            replicates, sampling_hash, embedder_hash,
        )

        # Computed unconditionally: it is a pure function of the request, and it
        # names the view even for the identity triple below, which is stored as
        # the base artifact rather than as a surrogate.  CollectionCache needs a
        # view identifier in both cases.
        spec = {
            "kind": "behavioral_surrogate",
            "query_key": query_key,
            "mode": self.variant_token(max_new_tokens, replicates, sampling_hash),
            "embedder_hash": embedder_hash,
            "view": view,
            "normalize": normalize,
            "replicate_reduction": replicate_reduction,
        }

        cached = None
        if view != "matrix" or normalize != "none" or replicate_reduction != "all":
            got = self.load_surrogate(base_model_id, adapter_id, query_key, spec)
            cached = got is not None
            if got is None:
                got = self._build_view(
                    matrix, view, normalize, replicates, replicate_reduction
                )
                self.save_surrogate(base_model_id, adapter_id, query_key, spec, got)
            matrix = got

        metadata = dict(meta.get("metadata", {}))
        metadata["generated_texts"] = self.load_generations(
            base_model_id, adapter_id, query_key, max_new_tokens,
            replicates, sampling_hash,
        )
        n_rows = int(matrix.shape[0])
        metadata.update(
            base_model_id=base_model_id,
            query_key=query_key,
            embedder_hash=embedder_hash,
            max_new_tokens=max_new_tokens,
            replicates=int(replicates),
            sampling=meta.get("sampling"),
            sampling_hash=sampling_hash,
            replicate_reduction=replicate_reduction,
            view=view,
            normalize=normalize,
            n_rows=n_rows,
            # After a mean the rows are queries again; before it each query owns
            # R of them.  A kernel's rows are neither, so it reports no count.
            n_queries=(
                None
                if view in self.KERNEL_VIEWS
                else (n_rows if replicate_reduction == "mean" else n_rows // int(replicates))
            ),
            is_kernel=view in self.KERNEL_VIEWS,
            # What identifies this read to CollectionCache — see the matching
            # note in ActivationCache.load.
            artifact_path=self.artifact_path(base_model_id, adapter_id, query_key),
            surrogate_hash=self.config_hash(spec),
        )
        if cached is not None:
            metadata["surrogate_cached"] = cached

        return ModelRepresentation(
            model_id=meta["model_id"],
            taxonomy=meta.get("taxonomy", "behavioral"),
            matrix=matrix,
            metadata=metadata,
            cache_key=(
                f"{adapter_slug(adapter_id)}/{self.draw_name(query_key)}/"
                f"{self._variant_stem(self.variant_token(max_new_tokens, replicates, sampling_hash), embedder_hash)}"
            ),
        )

    #: Accepted replicate reductions.  ``all`` keeps the stored rows.
    REPLICATE_REDUCTIONS = frozenset({"all", "mean"})

    @classmethod
    def canon_replicate_reduction(cls, reduction: str | None) -> str:
        """One spelling per reduction request, mirroring :meth:`canon_normalize`.

        ``None`` means "unspecified", which is the stored form.  Two spellings of
        the same request would hash to two surrogates for one matrix.
        """
        if reduction is None:
            return "all"
        if reduction in cls.REPLICATE_REDUCTIONS:
            return str(reduction)
        raise ValueError(
            f"unknown replicate_reduction {reduction!r}; expected one of "
            f"{sorted(cls.REPLICATE_REDUCTIONS)}"
        )

    @staticmethod
    def _build_view(
        matrix: np.ndarray,
        view: str,
        normalize: str,
        replicates: int = 1,
        replicate_reduction: str = "all",
    ) -> np.ndarray:
        """Apply a read-time view to the stored embedding matrix.

        Rows are queries — after a ``mean`` reduction, or exactly when
        ``replicates=1`` — as at the functional level, so ``gram`` here and
        ``gram`` there mean the same thing over the same draw.  With replicates
        kept, a ``gram`` is ``(n·R, n·R)``: still a kernel, but over replicates
        rather than queries, and the metric sees it that way.
        """
        H = matrix.astype(np.float64)
        if replicate_reduction == "mean" and int(replicates) > 1:
            R = int(replicates)
            if H.shape[0] % R:
                raise ValueError(
                    f"cannot average {R} replicates: {H.shape[0]} rows is not a "
                    "multiple of the replicate count. The matrix and the "
                    "filename disagree about how it was written."
                )
            # Query-major storage is what makes this a reshape rather than a
            # gather: rows q0r0..q0r(R-1) are already adjacent.
            H = H.reshape(H.shape[0] // R, R, H.shape[1]).mean(axis=1)
        if normalize in ("layer", "global"):
            # There is one block, so the two modes coincide; both are accepted
            # so a caller can pass the same selector to either level.
            H = _row_normalize(H)
        if view == "matrix":
            return H.astype(np.float32)
        if view == "gram":
            return (H @ H.T).astype(np.float32)
        raise ValueError(f"unknown view {view!r}; expected 'matrix' or 'gram'")

    # ------------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------------

    def list_variants(
        self, base_model_id: str, adapter_id: str, query_key: dict
    ) -> list[tuple[str, int, str, str]]:
        """``[(mode_token, replicates, sampling_hash, embedder_hash), ...]``.

        A directory listing, no file opens — the peer of
        ``ActivationCache.list_layers``.  Everything that distinguishes two
        stored generations is in the name, which is what lets a caller answer
        "what has been computed over this draw?" without a safetensors load.
        """
        emb_dir = self.draw_dir(base_model_id, adapter_id, query_key) / "embeddings"
        if not emb_dir.exists():
            return []
        out = []
        for p in sorted(emb_dir.glob("*.safetensors")):
            m = _GEN_RE.match(p.stem)
            if m:
                out.append((m["mode"], int(m["replicates"]), m["sampling"], m["embedder"]))
        return out


def _nest_texts(texts, replicates: int) -> list[list[str]]:
    """Normalize generated text to ``texts[query][replicate]``.

    Accepts what the taxonomy produces (already nested) and a flat list, so a
    caller assembling a representation by hand does not have to know which shape
    the writer wants.  A flat list is only unambiguous at ``replicates=1``;
    beyond that it is regrouped query-major, matching how the rows are stored.
    """
    texts = list(texts or [])
    if not texts:
        return []
    if all(isinstance(t, list) for t in texts):
        return [list(t) for t in texts]
    R = int(replicates)
    if len(texts) % R:
        raise ValueError(
            f"{len(texts)} generated texts is not a multiple of replicates={R}; "
            "cannot tell which replicate belongs to which query"
        )
    return [list(texts[i : i + R]) for i in range(0, len(texts), R)]


def _atomic_save_matrix(path: Path, matrix: np.ndarray, meta_bytes: np.ndarray) -> None:
    """Write ``matrix`` plus its ``_meta_json`` blob as one safetensors file."""
    from safetensors.numpy import save_file

    with atomic_path(path) as tmp:
        save_file({"matrix": matrix, "_meta_json": meta_bytes}, str(tmp))
