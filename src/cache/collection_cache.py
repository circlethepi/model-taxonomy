from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.core.distance import DistanceMatrix
from src.core.geometry import GeometryResult


class CollectionCache:
    """Cache for pairwise distance matrices and geometry results over model collections.

    Directory layout::

        cache_root/collections/{collection_hash}/
            collection_info.json        ← model entries + reconstruction info
            distance_matrix.safetensors ← NxN float32 distance matrix
            coordinates/
                pca.safetensors         ← one file per geometry method
                mds.safetensors
                umap.safetensors

    The collection hash is derived from (sorted model IDs, taxonomy, metric) so
    the same collection always maps to the same directory.

    ``collection_info.json`` contains enough information to reconstruct the
    collection without re-running extraction: each entry records whether the
    model is a base model or a LoRA adapter, and for adapters, the slug that
    locates the entry in :class:`LoRACache`.
    """

    def __init__(self, cache_root: Path | str) -> None:
        self.root = Path(cache_root)
        self._collections_dir = self.root / "collections"

    # ------------------------------------------------------------------
    # Hash helpers
    # ------------------------------------------------------------------

    @staticmethod
    def collection_hash(
        model_ids: list[str],
        taxonomy: str,
        metric: str,
    ) -> str:
        """Derive a stable 16-char hex hash for a (model_ids, taxonomy, metric) triple."""
        payload = json.dumps(
            {"model_ids": sorted(model_ids), "taxonomy": taxonomy, "metric": metric},
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def _collection_dir(self, chash: str) -> Path:
        return self._collections_dir / chash

    # ------------------------------------------------------------------
    # Existence check
    # ------------------------------------------------------------------

    def exists(self, chash: str) -> bool:
        d = self._collection_dir(chash)
        return (d / "distance_matrix.safetensors").exists()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_distance_matrix(
        self,
        distance_matrix: DistanceMatrix,
        model_entries: list[dict] | None = None,
    ) -> str:
        """Persist a DistanceMatrix.

        Parameters
        ----------
        distance_matrix:
            The NxN pairwise distance matrix to save.
        model_entries:
            Optional ordered list of dicts (one per model) with reconstruction
            metadata.  If omitted, a minimal entry is created for each model ID
            (``entry_type`` set to ``"base_model"``).

        Returns
        -------
        str
            The collection hash (directory name).
        """
        from safetensors.numpy import save_file

        chash = self.collection_hash(
            distance_matrix.model_ids,
            distance_matrix.taxonomy,
            distance_matrix.metric,
        )
        coll_dir = self._collection_dir(chash)
        coll_dir.mkdir(parents=True, exist_ok=True)

        if model_entries is None:
            model_entries = [
                {"model_id": mid, "entry_type": "base_model"}
                for mid in distance_matrix.model_ids
            ]

        info = {
            "schema_version": "1",
            "collection_hash": chash,
            "taxonomy": distance_matrix.taxonomy,
            "metric": distance_matrix.metric,
            "model_entries": model_entries,
            "geometry_methods": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Write collection_info.json
        info_tmp = coll_dir / "collection_info.json.tmp"
        info_tmp.write_text(json.dumps(info, indent=2))
        os.replace(info_tmp, coll_dir / "collection_info.json")

        # Write distance_matrix.safetensors
        meta_bytes = np.frombuffer(
            json.dumps(
                {
                    "model_ids": distance_matrix.model_ids,
                    "metric": distance_matrix.metric,
                    "taxonomy": distance_matrix.taxonomy,
                }
            ).encode("utf-8"),
            dtype=np.uint8,
        )
        st_tmp = coll_dir / "distance_matrix.safetensors.tmp"
        save_file(
            {
                "matrix": np.ascontiguousarray(distance_matrix.matrix.astype(np.float32)),
                "_meta_json": meta_bytes,
            },
            str(st_tmp),
        )
        os.replace(st_tmp, coll_dir / "distance_matrix.safetensors")

        return chash

    def save_geometry(self, chash: str, geometry: GeometryResult) -> None:
        """Persist a GeometryResult and record the method in collection_info.json."""
        from safetensors.numpy import save_file

        coll_dir = self._collection_dir(chash)
        coords_dir = coll_dir / "coordinates"
        coords_dir.mkdir(parents=True, exist_ok=True)

        st_tmp = coords_dir / f"{geometry.method}.safetensors.tmp"
        save_file(
            {"coordinates": np.ascontiguousarray(geometry.coordinates.astype(np.float32))},
            str(st_tmp),
        )
        os.replace(st_tmp, coords_dir / f"{geometry.method}.safetensors")

        # Update geometry_methods list in collection_info.json
        info_path = coll_dir / "collection_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text())
            if geometry.method not in info["geometry_methods"]:
                info["geometry_methods"].append(geometry.method)
                info_tmp = coll_dir / "collection_info.json.tmp"
                info_tmp.write_text(json.dumps(info, indent=2))
                os.replace(info_tmp, info_path)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_distance_matrix(self, chash: str) -> DistanceMatrix:
        from safetensors.numpy import load_file

        coll_dir = self._collection_dir(chash)
        tensors = load_file(str(coll_dir / "distance_matrix.safetensors"))
        matrix = tensors["matrix"].astype(np.float64)
        meta = json.loads(tensors["_meta_json"].tobytes().decode("utf-8"))
        return DistanceMatrix(matrix=matrix, **meta)

    def load_geometry(self, chash: str, method: str) -> GeometryResult:
        from safetensors.numpy import load_file

        coll_dir = self._collection_dir(chash)
        tensors = load_file(str(coll_dir / "coordinates" / f"{method}.safetensors"))
        coordinates = tensors["coordinates"]

        # Reconstruct metadata from collection_info.json
        info = self.load_info(chash)
        model_ids = [e["model_id"] for e in info["model_entries"]]
        return GeometryResult(
            coordinates=coordinates,
            model_ids=model_ids,
            method=method,
            taxonomy=info["taxonomy"],
            n_components=coordinates.shape[1],
        )

    def load_info(self, chash: str) -> dict:
        """Return the collection_info.json dict."""
        return json.loads((self._collection_dir(chash) / "collection_info.json").read_text())

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_collections(self) -> list[str]:
        """Return all collection hashes present in the cache."""
        if not self._collections_dir.exists():
            return []
        return [
            d.name
            for d in sorted(self._collections_dir.iterdir())
            if d.is_dir() and (d / "distance_matrix.safetensors").exists()
        ]
