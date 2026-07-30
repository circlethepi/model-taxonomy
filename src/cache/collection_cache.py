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

        cache_root/collections/
            index.json                      ← readable catalogue of every collection
            {collection_hash}/
                collection_info.json        ← model entries + reconstruction info
                distance_matrix.safetensors ← NxN float32 distance matrix
                coordinates/
                    mds_1d.safetensors      ← one file per (method, n_components)
                    mds_2d.safetensors
                    pca_2d.safetensors

    The collection hash is derived from (sorted model IDs, taxonomy, metric) so
    the same collection always maps to the same directory.  Because that name is
    opaque, ``index.json`` catalogues every collection by taxonomy, metric, label
    and model list, so the cache can be read without opening each directory in
    turn.

    ``collection_info.json`` contains enough information to reconstruct the
    collection without re-running extraction: each entry records whether the
    model is a base model or a LoRA adapter, and for adapters, the slug that
    locates the entry in :class:`LoRACache`.

    Coordinates are keyed by **both** method and dimension.  A simplex projection
    needs the embedding at ``k-1`` dimensions while a plot needs two, and the two
    must coexist — keying on the method alone silently overwrote one with the
    other.  Files written under the older ``{method}.safetensors`` name are still
    read.
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

    @staticmethod
    def geometry_key(method: str, n_components: int) -> str:
        """Filename stem for a geometry: ``"mds", 2`` → ``"mds_2d"``.

        Shared with :class:`~src.core.analysis.TaxonomyAnalysis` so the on-disk
        cache and the in-memory profile name the same embedding identically.
        """
        from src.core.analysis import geometry_key as _key

        return _key(method, n_components)

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
        label: str | None = None,
        slice_key: dict | None = None,
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
        label:
            Human-readable name recorded in ``index.json`` — the only thing that
            makes an opaque hash directory identifiable at a glance.
        slice_key:
            Which sub-collection this is, e.g. ``{"n_samples": 10, "seed": 0}``.
            Also recorded in the index, so all slices of one sweep can be found
            without re-deriving them.

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
            "schema_version": "2",
            "collection_hash": chash,
            "taxonomy": distance_matrix.taxonomy,
            "metric": distance_matrix.metric,
            "label": label,
            "slice": slice_key or {},
            "model_entries": model_entries,
            "geometry_methods": [],
            "geometries": [],
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

        self._update_index(chash, info)
        return chash

    def save_geometry(self, chash: str, geometry: GeometryResult) -> None:
        """Persist a GeometryResult under ``{method}_{n_components}d``.

        Keying on the dimension as well as the method is what lets a 1-D
        embedding (for a simplex projection) and a 2-D one (for a plot) of the
        same collection coexist.
        """
        from safetensors.numpy import save_file

        coll_dir = self._collection_dir(chash)
        coords_dir = coll_dir / "coordinates"
        coords_dir.mkdir(parents=True, exist_ok=True)

        key = self.geometry_key(geometry.method, geometry.n_components)

        # Carry the geometry's own metadata alongside the coordinates, matching
        # GeometryResult.save — otherwise stress and metadata are lost on reload
        # and a cached geometry is not interchangeable with a freshly fitted one.
        meta_bytes = np.frombuffer(
            json.dumps(
                {
                    "model_ids": geometry.model_ids,
                    "method": geometry.method,
                    "taxonomy": geometry.taxonomy,
                    "n_components": geometry.n_components,
                    "stress": geometry.stress,
                    "metadata": geometry.metadata,
                }
            ).encode("utf-8"),
            dtype=np.uint8,
        )

        st_tmp = coords_dir / f"{key}.safetensors.tmp"
        save_file(
            {
                "coordinates": np.ascontiguousarray(
                    geometry.coordinates.astype(np.float32)
                ),
                "_meta_json": meta_bytes,
            },
            str(st_tmp),
        )
        os.replace(st_tmp, coords_dir / f"{key}.safetensors")

        info_path = coll_dir / "collection_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text())
            entry = {"key": key, "method": geometry.method,
                     "n_components": geometry.n_components}
            geometries = info.setdefault("geometries", [])
            if entry not in geometries:
                geometries.append(entry)
            # geometry_methods is kept for readers written against schema 1.
            methods = info.setdefault("geometry_methods", [])
            if geometry.method not in methods:
                methods.append(geometry.method)
            info_tmp = coll_dir / "collection_info.json.tmp"
            info_tmp.write_text(json.dumps(info, indent=2))
            os.replace(info_tmp, info_path)
            self._update_index(chash, info)

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

    def load_geometry(
        self, chash: str, method: str, n_components: int | None = None
    ) -> GeometryResult:
        """Load one stored geometry.

        With *n_components* omitted the method must have exactly one stored
        dimension; otherwise the available options are listed rather than one
        being picked arbitrarily.
        """
        from safetensors.numpy import load_file

        coords_dir = self._collection_dir(chash) / "coordinates"

        if n_components is None:
            matches = [(m, n) for m, n in self.list_geometries(chash) if m == method]
            if not matches:
                raise FileNotFoundError(
                    f"collection {chash} has no {method!r} geometry. Stored: "
                    f"{self.list_geometries(chash)}"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"collection {chash} stores {method!r} at "
                    f"{sorted(n for _, n in matches)} dimensions; pass "
                    "n_components to choose."
                )
            n_components = matches[0][1]

        path = coords_dir / f"{self.geometry_key(method, n_components)}.safetensors"
        if not path.exists():
            # Written before coordinates were keyed by dimension.
            legacy = coords_dir / f"{method}.safetensors"
            if not legacy.exists():
                raise FileNotFoundError(
                    f"no {method!r} geometry at {n_components}d for collection "
                    f"{chash}. Stored: {self.list_geometries(chash)}"
                )
            path = legacy

        tensors = load_file(str(path))
        coordinates = tensors["coordinates"]

        info = self.load_info(chash)
        if "_meta_json" in tensors:
            meta = json.loads(tensors["_meta_json"].tobytes().decode("utf-8"))
            return GeometryResult(coordinates=coordinates, **meta)

        # Legacy file: no embedded metadata, so rebuild what we can from the
        # collection record.
        return GeometryResult(
            coordinates=coordinates,
            model_ids=[e["model_id"] for e in info["model_entries"]],
            method=method,
            taxonomy=info["taxonomy"],
            n_components=coordinates.shape[1],
        )

    def list_geometries(self, chash: str) -> list[tuple[str, int]]:
        """Return ``[(method, n_components), ...]`` stored for a collection."""
        coords_dir = self._collection_dir(chash) / "coordinates"
        if not coords_dir.exists():
            return []
        out: list[tuple[str, int]] = []
        for path in sorted(coords_dir.glob("*.safetensors")):
            stem = path.stem
            if "_" in stem and stem.rsplit("_", 1)[1].endswith("d"):
                method, dim = stem.rsplit("_", 1)
                try:
                    out.append((method, int(dim[:-1])))
                    continue
                except ValueError:
                    pass  # not a dimension suffix after all — treat as legacy
            out.append((stem, -1))  # legacy file, dimension unknown until loaded
        return out

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

    # ------------------------------------------------------------------
    # Readable catalogue
    # ------------------------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self._collections_dir / "index.json"

    def load_index(self) -> dict[str, dict]:
        """Return the catalogue: ``{collection_hash: summary}``.

        The directory names are content hashes, which are stable but unreadable.
        This is how to find out what is in the cache without opening each
        collection in turn.
        """
        if not self.index_path.exists():
            return {}
        try:
            return json.loads(self.index_path.read_text())
        except json.JSONDecodeError:
            return {}

    def find(self, **criteria) -> list[str]:
        """Collection hashes whose index entry matches every criterion.

        ``cc.find(taxonomy="structural", metric="cosine")`` is the usual form.
        """
        out = []
        for chash, record in self.load_index().items():
            if all(record.get(k) == v for k, v in criteria.items()):
                out.append(chash)
        return sorted(out)

    def _update_index(self, chash: str, info: dict) -> None:
        """Merge one collection's summary into index.json, atomically.

        Locked because several SLURM jobs can write different collections into
        the same cache at once, and a read-modify-write of a shared file is
        exactly where that would corrupt.
        """
        from filelock import FileLock

        self._collections_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "taxonomy": info.get("taxonomy"),
            "metric": info.get("metric"),
            "label": info.get("label"),
            "slice": info.get("slice", {}),
            "n_models": len(info.get("model_entries", [])),
            "model_ids": [e["model_id"] for e in info.get("model_entries", [])],
            "geometries": info.get("geometries", []),
            "created_at": info.get("created_at"),
        }

        with FileLock(str(self._collections_dir / "index.lock")):
            index = self.load_index()
            index[chash] = record
            tmp = self._collections_dir / "index.json.tmp"
            tmp.write_text(json.dumps(index, indent=2, sort_keys=True))
            os.replace(tmp, self.index_path)
