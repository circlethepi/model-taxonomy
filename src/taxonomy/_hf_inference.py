"""Shared HuggingFace model loading for the two inference-based taxonomy levels.

`FunctionalTaxonomy` and `BehavioralTaxonomy` differ in what they *do* with a
loaded model — one reads hidden states, the other reads generated text — but they
load it identically: resolve the base model behind an adapter directory, hold that
base across extractions, and swap adapters onto it.  This module owns that half so
the two levels cannot drift apart on it.

Extracted from `behavioral.py` without behavioural change; the docstrings on
:meth:`_load_tokenizer` and :meth:`_get_model` record *why* each choice is the way
it is and are the load-bearing part.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Any

import torch

from src.core.protocols import ModelID


def _adapter_name(model_id: ModelID) -> str:
    """A name for one adapter inside PEFT's in-process adapter registry.

    **This is an in-memory label, not a path, and nothing persists under it.**
    It only has to be unique among the adapters loaded into a single
    ``PeftModel`` during one process, which is why hashing the full ``model_id``
    is fine here: the string's dependence on the caller's working directory
    cannot outlive the process that computed it.

    :func:`src.cache._draw_keyed.adapter_slug` is **not** a substitute. It takes
    the directory basename, and adapter basenames collide across base models —
    ``…/Llama-3.2-3B/yahoo_050t0_050t1_n1000_s00_r16_i00`` and the same leaf
    under ``Llama-3.1-8B`` share it. On disk the base model is a separate path
    component so the two stay apart; here there is no such component, so the
    name must carry the distinction itself.

    This function used to be ``generated_text_cache.model_slug`` and was used for
    **both** jobs at once — this one and naming files on disk. That is what
    keyed the behavioral cache to the writer's working directory. Keeping the
    two apart is the fix; see ``docs/notes/TODO.md`` item 13.
    """
    digest = hashlib.sha256(str(model_id).encode()).hexdigest()[:8]
    return f"{Path(str(model_id)).name}__{digest}"


class HFInferenceTaxonomy:
    """Mixin providing base-model reuse and adapter swapping.

    The base model is loaded **once** and adapters are attached to it, rather than
    reloading it per model.  Across 25 adapters that is the difference between one
    6 GB read and twenty-five of them, and the ratio only gets worse at 8B.  Call
    :meth:`close` when done to release it — or use the class as a context manager.
    """

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 8,
        torch_dtype: torch.dtype = torch.float16,
        hf_token: str | None = None,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.torch_dtype = torch_dtype
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

        # Shared across extractions, freed by close().
        self._base_model: Any = None
        self._base_model_id: str | None = None
        self._peft_model: Any = None

    # ------------------------------------------------------------------
    # Model loading — base held across calls, adapters swapped
    # ------------------------------------------------------------------

    def _model_key(self, model_id: ModelID) -> tuple[str, str]:
        """``(base_model_id, adapter_id)`` for cache addressing.

        A plain HuggingFace model has no adapter and lands under ``_base``.

        Lives here rather than on either subclass because both inference caches
        are keyed the same way, so both taxonomies need the identical answer —
        and "identical" is load-bearing: two implementations of this that
        disagreed would put the functional and behavioral entries for one model
        at different coordinates, which is the divergence item 13 exists to
        remove.
        """
        base = self._resolve_base_model_id(model_id)
        if base is None:
            return str(model_id), "_base"
        return base, str(model_id)

    @staticmethod
    def _resolve_base_model_id(model_id: ModelID) -> str | None:
        """The base model an adapter directory was trained on, or None.

        None means *model_id* is not a local adapter directory — a plain
        HuggingFace model ID, or a full checkpoint — and should be loaded whole.
        """
        path = Path(model_id)
        meta = path / "experiment_meta.json"
        if meta.exists():
            base = json.loads(meta.read_text()).get("base_model_id")
            if base:
                return base
        adapter_cfg = path / "adapter_config.json"
        if adapter_cfg.exists():
            return json.loads(adapter_cfg.read_text()).get("base_model_name_or_path")
        return None

    def _load_tokenizer(self, model_id: ModelID, base_model_id: str | None) -> Any:
        """Tokenizer from the adapter dir, falling back to the base model.

        finetune_lora.py does not call ``tokenizer.save_pretrained`` explicitly; it
        relies on ``Trainer.save_model`` persisting the processing class, so the
        files are usually but not certainly there.
        """
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, token=self.hf_token, trust_remote_code=True
            )
        except Exception:
            if base_model_id is None:
                raise
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_id, token=self.hf_token, trust_remote_code=True
            )

        # Left padding is what makes batched greedy generation agree with
        # batch_size=1: with right padding the pads land between the prompt and
        # the first generated token and every short sequence in the batch decodes
        # from the wrong position.
        #
        # It matters for the functional level too, though for a different reason:
        # with left padding the last *real* token is always at index -1, so
        # last_token pooling needs no mask arithmetic to find it.
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _get_model(self, model_id: ModelID) -> tuple[Any, bool]:
        """Return ``(model, shared)``.

        ``shared`` is False when the model was loaded standalone and the caller
        should free it; True when it is the shared base with an adapter attached
        and must outlive this extraction.
        """
        from transformers import AutoModelForCausalLM

        base_model_id = self._resolve_base_model_id(model_id)

        if base_model_id is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                token=self.hf_token,
                trust_remote_code=True,
            )
            model.eval()
            return model, False

        # Adapter swapping is only valid across adapters sharing one base.  A
        # mixed-base collection degrades to standalone loads rather than silently
        # running inference on the wrong weights.
        if self._base_model is not None and base_model_id != self._base_model_id:
            warnings.warn(
                f"{model_id} has base {base_model_id!r} but this taxonomy already holds "
                f"{self._base_model_id!r}; loading it standalone. Base-model reuse only "
                f"applies within one base model.",
                stacklevel=2,
            )
            from peft import PeftModel

            base = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                token=self.hf_token,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base, str(model_id))
            model.eval()
            return model, False

        from peft import PeftModel

        if self._base_model is None:
            # Loading the base explicitly, rather than handing transformers an
            # adapter directory and relying on it to detect the adapter and
            # resolve base_model_name_or_path, keeps us in control of which base
            # is used and separates "base failed to load" from "adapter failed to
            # attach".  It is also what makes reuse possible at all.
            self._base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                token=self.hf_token,
                trust_remote_code=True,
            )
            self._base_model_id = base_model_id

        adapter_name = _adapter_name(model_id)
        if self._peft_model is None:
            self._peft_model = PeftModel.from_pretrained(
                self._base_model, str(model_id), adapter_name=adapter_name
            )
        elif adapter_name not in self._peft_model.peft_config:
            self._peft_model.load_adapter(str(model_id), adapter_name=adapter_name)

        self._peft_model.set_adapter(adapter_name)
        self._peft_model.eval()
        return self._peft_model, True

    def close(self) -> None:
        """Release the shared base model.

        The per-extraction ``finally`` frees only standalone models; the base is
        held on purpose and would otherwise survive until interpreter exit.
        """
        self._peft_model = None
        self._base_model = None
        self._base_model_id = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
