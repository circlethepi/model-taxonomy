from __future__ import annotations

from typing import Iterator

from src.core.protocols import ModelID


class ModelCollection:
    """A named collection of HuggingFace model IDs with optional hub metadata."""

    def __init__(self, model_ids: list[ModelID]) -> None:
        self._model_ids = list(model_ids)
        self._metadata: dict[ModelID, object] = {}

    @classmethod
    def from_ids(cls, model_ids: list[ModelID]) -> "ModelCollection":
        return cls(model_ids)

    @classmethod
    def from_hub_search(
        cls,
        search: str | None = None,
        author: str | None = None,
        task: str | None = None,
        library: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> "ModelCollection":
        """Search the HuggingFace Hub and return matching model IDs.

        Parameters mirror huggingface_hub.HfApi.list_models arguments.
        """
        from huggingface_hub import HfApi

        api = HfApi()
        models = api.list_models(
            search=search,
            author=author,
            task=task,
            library=library,
            tags=tags,
            limit=limit,
            sort="downloads",
            direction=-1,
        )
        model_ids = [m.modelId for m in models if m.modelId is not None]
        instance = cls(model_ids)
        return instance

    def metadata(self, model_id: ModelID) -> object:
        """Fetch (and cache) ModelInfo for a model from the Hub."""
        if model_id not in self._metadata:
            from huggingface_hub import HfApi

            api = HfApi()
            self._metadata[model_id] = api.model_info(model_id)
        return self._metadata[model_id]

    def __iter__(self) -> Iterator[ModelID]:
        return iter(self._model_ids)

    def __len__(self) -> int:
        return len(self._model_ids)

    def __repr__(self) -> str:
        return f"ModelCollection({self._model_ids!r})"

    def to_list(self) -> list[ModelID]:
        return list(self._model_ids)
