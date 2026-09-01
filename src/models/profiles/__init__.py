"""The registry of known model families.

One module per family, one entry here.  Adding a base model to the pipeline
should be exactly this: a new file describing the model, and a line below.
"""

from __future__ import annotations

from src.models.profile import ModelProfile

from .llama3 import LLAMA3
from .llama3_instruct import LLAMA3_INSTRUCT
from .mistral_nemo import MISTRAL_NEMO
from .olmo2 import OLMO2
from .qwen3_5 import QWEN3_5

#: Order is irrelevant -- :func:`src.models.profile.resolve` picks the longest
#: matching prefix, so a more specific profile always wins regardless of where
#: it sits in this list.
PROFILES: tuple[ModelProfile, ...] = (
    LLAMA3,
    LLAMA3_INSTRUCT,
    MISTRAL_NEMO,
    OLMO2,
    QWEN3_5,
)

__all__ = [
    "PROFILES",
    "LLAMA3",
    "LLAMA3_INSTRUCT",
    "MISTRAL_NEMO",
    "OLMO2",
    "QWEN3_5",
]
