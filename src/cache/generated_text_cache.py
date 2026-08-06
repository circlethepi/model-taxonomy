from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from src.cache._draw_keyed import (
    DrawKeyedCache,
    _atomic_write_json,
    _row_normalize,
    adapter_slug,
)
from src.core.representation import ModelRepresentation

__all__ = ["GeneratedTextCache", "_GEN_RE"]


#: Embedding filenames look like ``generation128_5191ad734b81daff.safetensors``:
#: the generation mode (carrying its token budget) and the embedder that turned
#: the text into vectors.  Peer of ``activation_cache._ACT_RE`` — everything that
#: changes the stored numbers is in the name, so a directory listing is a
#: complete description of what was computed.
#:
#: If TODO item 12 gives behavioral a ``representation:`` knob, it extends this
#: name to ``{mode}_{embedder}_{representation}`` rather than becoming a side
#: file; keep the regex and the writer in step.
_GEN_RE = re.compile(r"^(?P<mode>[a-z]+\d*)_(?P<embedder>[0-9a-f]{16})$")


class GeneratedTextCache(DrawKeyedCache):
    """Cache for behavioral representations: generated text and its embeddings.

    Directory layout::

        cache_root/05_generated/{base_slug}/{adapter_slug}/{recipe_hash}/n{n}_s{seed}/
            queries.json                                ← query_key + source row indices
            runs/{config_hash}.json                     ← extraction provenance
            generations/{mode_token}.json               ← {model_id, generated_texts}
            embeddings/{mode_token}_{embedder_hash}.safetensors
            surrogates/{surrogate_hash}/
                config.json
                surrogate.safetensors

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

    **Generations are keyed by mode alone, embeddings by mode and embedder.**
    That is the asymmetry that makes a second embedder cheap: the text is
    generated once, and re-embedding it adds one file beside the first with no
    GPU generation pass.  ``ls embeddings/`` answers "which embedders have run
    over this draw?" from a directory listing.

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

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _variant_stem(self, mode_token: str, embedder_hash: str) -> str:
        return f"{mode_token}_{embedder_hash}"

    def generations_path(
        self,
        base_model_id: str,
        adapter_id: str,
        query_key: dict,
        max_new_tokens: int,
    ) -> Path:
        token = self.mode_token("generation", max_new_tokens)
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
        embedder_hash: str,
    ) -> Path:
        token = self.mode_token("generation", max_new_tokens)
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
        embedder_hash: str,
    ) -> bool:
        return (
            self.embeddings_path(
                base_model_id, adapter_id, query_key, max_new_tokens, embedder_hash
            ).exists()
            and self.generations_path(
                base_model_id, adapter_id, query_key, max_new_tokens
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
        embedder_hash: str,
        config: dict | None = None,
        source_indices: list | None = None,
    ) -> None:
        """Atomically write queries, run record, generations and embeddings.

        Idempotent: returns immediately if this model's entry already exists.
        Note the consequence — there is no invalidation path.  Anything inside
        *config* that changes the ``embedder_hash`` produces a new filename and
        so a new entry, which is the behaviour you want and is legible on disk.
        Anything outside it (``device``, ``batch_size``, and see the class
        docstring on ``torch_dtype``) silently reuses the existing entry.
        """
        from filelock import FileLock

        draw = self.draw_dir(base_model_id, adapter_id, query_key)
        (draw / "generations").mkdir(parents=True, exist_ok=True)
        (draw / "embeddings").mkdir(parents=True, exist_ok=True)

        with FileLock(str(draw / ".lock")):
            if self.exists(
                base_model_id, adapter_id, query_key, max_new_tokens, embedder_hash
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
                        "embedder_hash": embedder_hash,
                        "model_id": rep.model_id,
                        # Provenance the config deliberately excludes so it does
                        # not fragment the cache; see BehavioralTaxonomy.
                        "device_name": metadata.get("device_name"),
                        "batch_size": metadata.get("batch_size"),
                    },
                )

            # generations/{mode_token}.json — written once per mode, then shared
            # by every embedder that runs over this draw.
            gen_path = self.generations_path(
                base_model_id, adapter_id, query_key, max_new_tokens
            )
            if not gen_path.exists():
                _atomic_write_json(
                    gen_path,
                    {"model_id": rep.model_id, "generated_texts": generated_texts or []},
                )

            # embeddings/{mode_token}_{embedder_hash}.safetensors
            meta_bytes = np.frombuffer(
                json.dumps(
                    {
                        "model_id": rep.model_id,
                        "base_model_id": base_model_id,
                        "taxonomy": rep.taxonomy,
                        "query_key": query_key,
                        "embedder_hash": embedder_hash,
                        "max_new_tokens": max_new_tokens,
                        "metadata": metadata,
                    }
                ).encode("utf-8"),
                dtype=np.uint8,
            )
            st_path = self.embeddings_path(
                base_model_id, adapter_id, query_key, max_new_tokens, embedder_hash
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
    ) -> list[str]:
        """The generated text for one model, without touching the tensors."""
        path = self.generations_path(base_model_id, adapter_id, query_key, max_new_tokens)
        if not path.exists():
            return []
        return json.loads(path.read_text()).get("generated_texts", [])

    def load_matrix(
        self,
        base_model_id: str,
        adapter_id: str,
        query_key: dict,
        max_new_tokens: int,
        embedder_hash: str,
    ) -> tuple[np.ndarray, dict]:
        """The stored ``(n_queries, d)`` embedding matrix and its ``_meta_json``."""
        from safetensors.numpy import load_file

        path = self.embeddings_path(
            base_model_id, adapter_id, query_key, max_new_tokens, embedder_hash
        )
        if not path.exists():
            raise FileNotFoundError(
                f"no stored behavioral embeddings for {adapter_id} under draw "
                f"{self.draw_name(query_key)} "
                f"(generation{max_new_tokens}, embedder {embedder_hash}) at {path}"
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
        embedder_hash: str,
        view: str = "matrix",
        normalize: str | bool = "none",
    ) -> ModelRepresentation:
        """Reconstruct a ModelRepresentation, generations folded back in.

        ``metadata["generated_texts"]`` is restored from ``generations/`` so the
        returned object is indistinguishable from one that was never cached.

        ``view="matrix"`` is the stored embedding itself and is the default,
        which is why it is *not* routed through ``surrogates/``: writing a
        byte-copy of the stored matrix back beside it would double this stage's
        footprint to buy nothing.  Any other view goes through the same
        surrogate mechanism the functional level uses, so a ``gram`` is computed
        at most once per ``(draw, mode, embedder, view, normalize)``.
        """
        normalize = self.canon_normalize(normalize)
        matrix, meta = self.load_matrix(
            base_model_id, adapter_id, query_key, max_new_tokens, embedder_hash
        )

        cached = None
        if view != "matrix" or normalize != "none":
            spec = {
                "kind": "behavioral_surrogate",
                "query_key": query_key,
                "mode": self.mode_token("generation", max_new_tokens),
                "embedder_hash": embedder_hash,
                "view": view,
                "normalize": normalize,
            }
            got = self.load_surrogate(base_model_id, adapter_id, query_key, spec)
            cached = got is not None
            if got is None:
                got = self._build_view(matrix, view, normalize)
                self.save_surrogate(base_model_id, adapter_id, query_key, spec, got)
            matrix = got

        metadata = dict(meta.get("metadata", {}))
        metadata["generated_texts"] = self.load_generations(
            base_model_id, adapter_id, query_key, max_new_tokens
        )
        metadata.update(
            base_model_id=base_model_id,
            query_key=query_key,
            embedder_hash=embedder_hash,
            max_new_tokens=max_new_tokens,
            view=view,
            normalize=normalize,
            n_queries=int(matrix.shape[0]),
            is_kernel=view in self.KERNEL_VIEWS,
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
                f"{self._variant_stem(self.mode_token('generation', max_new_tokens), embedder_hash)}"
            ),
        )

    @staticmethod
    def _build_view(matrix: np.ndarray, view: str, normalize: str) -> np.ndarray:
        """Apply a read-time view to the stored embedding matrix.

        Rows are queries, exactly as at the functional level, so ``gram`` here
        and ``gram`` there mean the same thing: a ``(n_queries, n_queries)``
        kernel over the same draw.
        """
        H = matrix.astype(np.float64)
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
    ) -> list[tuple[str, str]]:
        """``[(mode_token, embedder_hash), ...]`` stored for one model-draw.

        A directory listing, no file opens — the peer of
        ``ActivationCache.list_layers``.
        """
        emb_dir = self.draw_dir(base_model_id, adapter_id, query_key) / "embeddings"
        if not emb_dir.exists():
            return []
        out = []
        for p in sorted(emb_dir.glob("*.safetensors")):
            m = _GEN_RE.match(p.stem)
            if m:
                out.append((m["mode"], m["embedder"]))
        return out


def _atomic_save_matrix(path: Path, matrix: np.ndarray, meta_bytes: np.ndarray) -> None:
    """Write ``matrix`` plus its ``_meta_json`` blob as one safetensors file."""
    from safetensors.numpy import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".safetensors.tmp")
    save_file({"matrix": matrix, "_meta_json": meta_bytes}, str(tmp))
    os.replace(tmp, path)
