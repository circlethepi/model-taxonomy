from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np

from src.core.protocols import Embedder

#: Models whose weights were trained with a mandatory task-instruction prefix, so
#: embedding bare text is a misuse rather than a stylistic choice.
#:
#: This exists because relying on ``SentenceTransformer.encode(prompt_name=...)``
#: silently did nothing here.  ``nomic-embed-text-v1.5``'s
#: ``config_sentence_transformers.json`` carries only a ``__version__`` block — no
#: ``prompts`` map — and sentence-transformers then synthesises
#: ``{"query": "", "document": ""}``.  Both entries are the *empty string*, so
#: ``prompt_name="document"`` resolved to a valid key that prepended nothing and
#: raised no error and no warning.
#:
#: Measured impact, embedding the same 100 rows of each yahoo mixture both ways: the
#: centroids move by cosine 0.94–0.96, and every pairwise distance shrinks by roughly
#: a constant factor (the prefix adds a component shared by all texts).  The
#: *geometry* is unaffected — distance-matrix agreement is pearson 0.9998, and MDS
#: recovery of the mixing proportion is 0.9977 vs 0.9976 pearson, spearman 1.0000
#: either way.  So embeddings written before this was fixed are internally consistent
#: and still usable; what must never happen is *mixing* the two kinds in one
#: comparison, since they live on different scales.  ``prompt_prefix`` is part of
#: :meth:`SentenceTransformerEmbedder.config_dict` to make that impossible.
#:
#: Keys are matched by prefix so the v1/v1.5/quantised variants are all covered.
_PREFIX_REQUIRED_MODELS = ("nomic-ai/nomic-embed-text",)

#: Canonical nomic task prefixes, plus the short aliases that appear in this repo's
#: experiment YAML (``prompt_name: document``).  Aliases map onto the canonical
#: literal so old configs keep working and start doing what they always claimed to.
_NOMIC_PREFIXES = {
    "search_document": "search_document: ",
    "search_query": "search_query: ",
    "clustering": "clustering: ",
    "classification": "classification: ",
    "document": "search_document: ",
    "query": "search_query: ",
}

#: Used when a prefix-required model is given no ``prompt_name``.  Corpus text is what
#: this repo embeds — dataset prose at the dataset level, generated continuations at
#: the behavioral level — so ``search_document`` is the right default for both.
#:
#: Defaulting rather than raising is deliberate, and is not the same class of silence
#: as the bug above: that bug applied *no prefix at all*, which misuses the model.
#: Choosing ``search_document`` over ``search_query`` or ``clustering`` is a far
#: smaller distinction — all three use the model as trained.  An unknown
#: ``prompt_name`` still raises, so a typo cannot quietly land here.
_DEFAULT_PROMPT_NAME = "search_document"


class SentenceTransformerEmbedder(Embedder):
    """Embeds a query using a separate sentence-transformers model.

    The sentence-transformer is loaded once at construction time on a separate
    device (usually CPU) and is kept alive across all queries and models.
    The LM under analysis generates text; this embedder then encodes that text.
    If use_generated_text=False, the raw query string is encoded instead.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        use_generated_text: bool = True,
        normalize_embeddings: bool = True,
        trust_remote_code: bool = False,
        prompt_name: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.use_generated_text = use_generated_text
        self.normalize_embeddings = normalize_embeddings
        self.trust_remote_code = trust_remote_code
        self.prompt_name = prompt_name
        self._st_model = None
        self._embedding_dim: int | None = None
        self.prompt_prefix = self._resolve_prefix(model_name, prompt_name)

    @property
    def requires_prefix(self) -> bool:
        return self.model_name.startswith(_PREFIX_REQUIRED_MODELS)

    @staticmethod
    def _resolve_prefix(model_name: str, prompt_name: str | None) -> str:
        """The literal string to prepend, resolved at construction.

        Resolving here rather than at encode time means the prefix is settled before
        any model loads, and an unusable *prompt_name* is reported immediately rather
        than after a multi-GB download.

        Returns ``""`` for models that take no prefix, in which case *prompt_name* is
        passed through to ``encode`` as before and sentence-transformers resolves it.
        """
        if not model_name.startswith(_PREFIX_REQUIRED_MODELS):
            return ""

        if prompt_name is None:
            prefix = _NOMIC_PREFIXES[_DEFAULT_PROMPT_NAME]
            # Never fires for any config in this repo — every nomic block sets
            # prompt_name explicitly — so this speaks up only for someone who omitted
            # it, which is exactly who needs to know a task prefix is being chosen.
            warnings.warn(
                f"{model_name} is trained with a task-instruction prefix and no "
                f"prompt_name was given; defaulting to {_DEFAULT_PROMPT_NAME!r} "
                f"({prefix!r}). Set prompt_name explicitly to choose another: "
                f"{sorted(_NOMIC_PREFIXES)}.",
                stacklevel=3,
            )
            return prefix

        try:
            return _NOMIC_PREFIXES[prompt_name]
        except KeyError:
            # Deliberately not falling back to the default: a typo that silently got
            # the wrong prefix would recreate the bug this module exists to prevent.
            raise ValueError(
                f"prompt_name={prompt_name!r} is not a task prefix understood by "
                f"{model_name}. Choose from {sorted(_NOMIC_PREFIXES)} — "
                f"'search_document' for corpus text, 'search_query' for questions, "
                f"'clustering' for grouping texts by similarity."
            ) from None

    def _load(self) -> None:
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(
                self.model_name, device=self.device, trust_remote_code=self.trust_remote_code
            )
            self._embedding_dim = self._st_model.get_sentence_embedding_dimension()

    @property
    def embedding_dim(self) -> int | None:
        return self._embedding_dim

    def embed(self, model_output: Any, query: str) -> np.ndarray:
        self._load()
        if self.use_generated_text:
            text = getattr(model_output, "generated_text", None)
            if text is None:
                raise ValueError(
                    "model_output.generated_text is None. "
                    "Set max_new_tokens > 0 in BehavioralTaxonomy or use use_generated_text=False."
                )
        else:
            text = query

        encode_kwargs: dict = dict(
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        if self.prompt_prefix:
            # Prepended directly rather than passed as prompt_name=: for this model
            # the prompts map sentence-transformers synthesises holds empty strings,
            # so the kwarg is accepted and silently prepends nothing.
            text = self.prompt_prefix + text
        elif self.prompt_name is not None:
            encode_kwargs["prompt_name"] = self.prompt_name
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*get_extended_attention_mask.*")
            vec = self._st_model.encode(text, **encode_kwargs)
        return vec.astype(np.float32)

    def encode_texts(
        self, texts: Sequence[str], batch_size: int = 256
    ) -> np.ndarray:
        """Embed many texts in one batched pass, as :meth:`embed` would one by one.

        The same prefix, the same ``normalize_embeddings``, the same ``encode``
        kwargs — this is :meth:`embed`'s body hoisted out for a caller that holds
        a list.  ``scripts/reembed_behavioral.py`` is why it exists: one qbig
        draw is 65,000 continuations, and a call per string is dominated by
        per-call overhead rather than by the model.

        Returns ``(len(texts), d)`` float32, in the order given.

        **Not bit-identical to :meth:`embed` in a loop.**  A batch is padded to
        its longest member, so the attention mask — and with it the kernels that
        run — depend on how the texts were grouped.  Measured drift is ~1e-6 per
        component on normalized vectors, orders of magnitude below any distance
        this repo reports, but it is not zero: treat a matrix from here and one
        from :class:`~src.taxonomy.behavioral.BehavioralTaxonomy` as equal to
        float tolerance, never as byte-equal.  The *embedder_hash* is the same
        either way, which is the point — batching is a machine detail and does
        not belong in the cache key, exactly as ``batch_size`` stays out of
        ``BehavioralTaxonomy.config_dict``.
        """
        self._load()
        texts = list(texts)
        if not texts:
            return np.empty((0, self._embedding_dim or 0), dtype=np.float32)

        encode_kwargs: dict = dict(
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            batch_size=int(batch_size),
        )
        if self.prompt_prefix:
            # Prepended here rather than passed as prompt_name=, for the reason
            # spelled out in embed(): this model's synthesised prompts map holds
            # empty strings, so the kwarg silently prepends nothing.
            texts = [self.prompt_prefix + t for t in texts]
        elif self.prompt_name is not None:
            encode_kwargs["prompt_name"] = self.prompt_name

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*get_extended_attention_mask.*")
            vecs = self._st_model.encode(texts, **encode_kwargs)
        return np.ascontiguousarray(vecs, dtype=np.float32)

    def config_dict(self) -> dict[str, Any]:
        return {
            "embedder_class": "SentenceTransformerEmbedder",
            "model_name": self.model_name,
            "use_generated_text": self.use_generated_text,
            "normalize_embeddings": self.normalize_embeddings,
            "trust_remote_code": self.trust_remote_code,
            "prompt_name": self.prompt_name,
            # The literal that was actually prepended, not just the name that was
            # asked for.  Keying on the name alone cannot distinguish embeddings
            # computed when prompt_name was a silent no-op from correctly-prefixed
            # ones — they would share a hash and be treated as interchangeable.
            "prompt_prefix": self.prompt_prefix,
        }
