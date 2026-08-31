from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.core.distance import DistanceMatrix
from src.core.geometry import GeometryResult

from ._draw_keyed import DrawKeyedCache


class CollectionCache:
    """Cache for pairwise distance matrices and geometry results over model collections.

    Directory layout::

        cache_root/07_collections/
            index.json                        ← readable catalogue of every collection
            {taxonomy}/
                {collection_key}/
                    collection_info.json      ← model entries + their artifact paths
                    {metric}_{surrogate_key}/
                        config.json           ← surrogate spec, per-model hashes, metric
                        distance_matrix.safetensors
                        coordinates/
                            mds_1d.safetensors
                            mds_2d.safetensors

    **What identifies a collection.**  A distance matrix is determined by which
    tensor was read for each model, not by the model IDs alone.  The key is
    therefore split in two:

    ``collection_key``
        A hash over the ordered ``(model_id, artifact_path)`` pairs, where the
        path is **relative to the cache root** and stops *before* ``surrogates/``
        — the stored artifact, which is the part that differs per model.
    ``surrogate_key``
        A hash over the ordered list of each model's surrogate hash.  Those
        hashes usually coincide, because a surrogate spec is keyed on the shared
        *query* draw rather than on each model's training recipe — but they are
        not required to, and they genuinely diverge when models are extracted
        against different query datasets.  Digesting the list handles both cases
        under one rule.

    Together they form the **handle**, ``{taxonomy}/{collection_key}/{metric}_{surrogate_key}``,
    which every method takes in place of the old opaque hash.  ``metric`` is
    spelled out so the layout can be read with ``ls``; the two hashes are opaque,
    so ``index.json`` catalogues every collection by taxonomy, metric, label and
    model list.

    This replaces a key of ``(sorted model IDs, taxonomy, metric)``, which was
    blind to the draw, embedder, view, normalization and pooling — so a
    collection built under one selector was returned unchanged for a different
    one, silently.  See ``docs/notes/TODO.md`` item 14.

    ``collection_info.json`` sits at the ``collection_key`` level, so everything
    about a collection is at the top and is shared by every metric and view
    computed over it.  Each leaf's ``config.json`` records the surrogate spec and
    **every per-model surrogate hash and path** — the directory name is a digest,
    so this file is what traces a collection back to its inputs.

    Coordinates are keyed by **both** method and dimension.  A simplex projection
    needs the embedding at ``k-1`` dimensions while a plot needs two, and the two
    must coexist — keying on the method alone silently overwrote one with the
    other.  Files written under the older ``{method}.safetensors`` name are still
    read.
    """

    def __init__(self, cache_root: Path | str) -> None:
        self.root = Path(cache_root)
        self._collections_dir = self.root / "07_collections"

    # ------------------------------------------------------------------
    # Hash helpers
    # ------------------------------------------------------------------

    @staticmethod
    def collection_key(model_entries: list[dict]) -> str:
        """Hash the models and the artifacts they were read from.

        *model_entries* is the ordered list written to ``collection_info.json``;
        only ``model_id`` and ``artifact_path`` take part in the hash, so adding
        descriptive fields later does not invalidate the cache.

        The paths must be **relative to the cache root**.  Absolute paths would
        make an entry unreachable from any other working directory — the failure
        item 13 records, where ``05_generated`` resolved 0 of 25 adapters because
        stored IDs were paths interpreted against the wrong root.
        """
        payload = [
            {"model_id": e["model_id"], "artifact_path": e.get("artifact_path")}
            for e in sorted(model_entries, key=lambda e: e["model_id"])
        ]
        for entry in payload:
            path = entry["artifact_path"]
            if path is not None and Path(path).is_absolute():
                raise ValueError(
                    f"artifact_path must be relative to the cache root, got the "
                    f"absolute path {path!r} for model {entry['model_id']!r}. An "
                    "absolute path would key this collection to one working "
                    "directory (TODO.md item 13)."
                )
        return DrawKeyedCache.config_hash({"models": payload})

    @staticmethod
    def surrogate_key(surrogate_hashes: list[str | None]) -> str:
        """Hash the ordered list of per-model surrogate hashes.

        One rule whether or not the models share a surrogate.  They usually do —
        a surrogate spec carries the *query* draw's recipe hash, which is common
        to every model — but that is a property of the data, not a guarantee, and
        it breaks as soon as models are extracted against different query
        datasets.  Digesting the list keeps such a collection addressable instead
        of refusing it or silently collapsing it onto one model's view.
        """
        return DrawKeyedCache.config_hash({"surrogates": list(surrogate_hashes)})

    @staticmethod
    def handle(
        taxonomy: str, collection_key: str, metric: str, surrogate_key: str
    ) -> str:
        """Assemble the relative path that names one stored distance matrix."""
        if "/" in metric:
            raise ValueError(f"metric {metric!r} cannot contain '/': it names a directory")
        return f"{taxonomy}/{collection_key}/{metric}_{surrogate_key}"

    def _collection_dir(self, handle: str) -> Path:
        return self._collections_dir / handle

    def _info_dir(self, handle: str) -> Path:
        """Where ``collection_info.json`` lives — one level above the leaf."""
        return self._collection_dir(handle).parent

    @staticmethod
    def geometry_key(
        method: str, n_components: int, mds_kwargs: dict | None = None
    ) -> str:
        """Filename stem for a geometry: ``"mds", 2`` → ``"mds_2d"``.

        Shared with :class:`~src.core.analysis.TaxonomyAnalysis` so the on-disk
        cache and the in-memory profile name the same embedding identically.

        *mds_kwargs* is folded in as ``mds@{hash}_2d`` when it differs from the
        default.  Without it, refitting at a different ``random_state`` returned
        the previous coordinates from cache — the same blind spot as item 14, one
        level down.  The default spelling is left bare so the common case stays
        readable and existing files keep their names.
        """
        from src.core.analysis import geometry_key as _key

        resolved = dict(mds_kwargs or {})
        resolved.setdefault("random_state", 0)
        if resolved == {"random_state": 0}:
            return _key(method, n_components)
        suffix = DrawKeyedCache.config_hash(resolved)[:8]
        return _key(f"{method}@{suffix}", n_components)

    # ------------------------------------------------------------------
    # Existence check
    # ------------------------------------------------------------------

    def exists(self, handle: str) -> bool:
        d = self._collection_dir(handle)
        return (d / "distance_matrix.safetensors").exists()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_distance_matrix(
        self,
        distance_matrix: DistanceMatrix,
        handle: str,
        model_entries: list[dict] | None = None,
        label: str | None = None,
        slice_key: dict | None = None,
        config: dict | None = None,
    ) -> str:
        """Persist a DistanceMatrix at *handle*.

        Parameters
        ----------
        distance_matrix:
            The NxN pairwise distance matrix to save.
        handle:
            ``{taxonomy}/{collection_key}/{metric}_{surrogate_key}``, from
            :meth:`handle`.  Taken rather than re-derived: the old code rebuilt
            the key from ``distance_matrix.metric`` while callers looked it up
            under the metric *name* they passed, so a ``"cka"`` collection was
            stored as ``"cka_linear"`` and never hit.
        model_entries:
            Ordered list of dicts (one per model) with reconstruction metadata,
            including the ``artifact_path`` that went into the key.  If omitted,
            a minimal entry is created for each model ID.
        label:
            Human-readable name recorded in ``index.json`` — the only thing that
            makes an opaque hash directory identifiable at a glance.
        slice_key:
            Which sub-collection this is, e.g. ``{"n_samples": 10, "seed": 0}``.
        config:
            The leaf's ``config.json`` payload: the surrogate spec, the per-model
            surrogate hashes, and the metric configuration.  This is what traces
            a collection back to its inputs, since the directory name is a digest.

        Returns
        -------
        str
            The handle it was written to.
        """
        from safetensors.numpy import save_file

        coll_dir = self._collection_dir(handle)
        coll_dir.mkdir(parents=True, exist_ok=True)

        if model_entries is None:
            model_entries = [
                {"model_id": mid, "entry_type": "base_model"}
                for mid in distance_matrix.model_ids
            ]

        info = {
            "schema_version": "4",
            "collection_key": handle.split("/")[1],
            "taxonomy": distance_matrix.taxonomy,
            "label": label,
            "slice": slice_key or {},
            "model_entries": model_entries,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # collection_info.json describes the collection, not this metric/view, so
        # it lives one level up and is shared by every leaf under it.  Several
        # leaves race to write it; the content is identical, so the write is
        # idempotent and the lock only needs to make it atomic.
        self._write_info(handle, info)

        leaf_config = {
            "schema_version": "1",
            "metric": distance_matrix.metric,
            **(config or {}),
        }
        cfg_tmp = coll_dir / "config.json.tmp"
        cfg_tmp.write_text(json.dumps(leaf_config, indent=2, sort_keys=True))
        os.replace(cfg_tmp, coll_dir / "config.json")

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
        # float64, not float32. A distance matrix is computed in float64 and read
        # back as float64, so storing it at half the width means a warm run and a
        # cold run disagree from the eighth significant digit — invisible in a
        # figure, but it defeats the only test that the reuse is correct, which
        # is that the two runs produce the same `matrix_sha256`. At 16 models
        # this buys the fidelity for two kilobytes. Entries already written at
        # float32 still load: `load_distance_matrix` casts whatever it finds.
        st_tmp = coll_dir / "distance_matrix.safetensors.tmp"
        save_file(
            {
                "matrix": np.ascontiguousarray(
                    distance_matrix.matrix.astype(np.float64)),
                "_meta_json": meta_bytes,
            },
            str(st_tmp),
        )
        os.replace(st_tmp, coll_dir / "distance_matrix.safetensors")

        self._update_index(handle, info, leaf_config)
        return handle

    def _write_info(self, handle: str, info: dict) -> None:
        """Write ``collection_info.json`` at the collection level, atomically."""
        from filelock import FileLock

        info_dir = self._info_dir(handle)
        info_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(info_dir / "info.lock")):
            tmp = info_dir / "collection_info.json.tmp"
            tmp.write_text(json.dumps(info, indent=2))
            os.replace(tmp, info_dir / "collection_info.json")

    def save_geometry(
        self,
        handle: str,
        geometry: GeometryResult,
        mds_kwargs: dict | None = None,
    ) -> None:
        """Persist a GeometryResult under ``{method}_{n_components}d``.

        Keying on the dimension as well as the method is what lets a 1-D
        embedding (for a simplex projection) and a 2-D one (for a plot) of the
        same collection coexist.  *mds_kwargs* additionally keeps two fits at
        different ``random_state`` apart.
        """
        from safetensors.numpy import save_file

        coll_dir = self._collection_dir(handle)
        coords_dir = coll_dir / "coordinates"
        coords_dir.mkdir(parents=True, exist_ok=True)

        key = self.geometry_key(geometry.method, geometry.n_components, mds_kwargs)

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

        # float64 for the same reason the distance matrix is: a cached embedding
        # has to be interchangeable with a freshly fitted one, and the stress and
        # Procrustes residual reported beside it are read to six decimals.
        st_tmp = coords_dir / f"{key}.safetensors.tmp"
        save_file(
            {
                "coordinates": np.ascontiguousarray(
                    geometry.coordinates.astype(np.float64)
                ),
                "_meta_json": meta_bytes,
            },
            str(st_tmp),
        )
        os.replace(st_tmp, coords_dir / f"{key}.safetensors")

        # Geometries belong to the leaf, not the collection: coordinates are fitted
        # to one metric's distance matrix, so two metrics over the same models have
        # different ones.  They were recorded in collection_info.json when that file
        # was per-metric; it no longer is.
        cfg_path = coll_dir / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            entry = {"key": key, "method": geometry.method,
                     "n_components": geometry.n_components}
            geometries = cfg.setdefault("geometries", [])
            if entry not in geometries:
                geometries.append(entry)
            cfg_tmp = coll_dir / "config.json.tmp"
            cfg_tmp.write_text(json.dumps(cfg, indent=2, sort_keys=True))
            os.replace(cfg_tmp, cfg_path)
            info_path = self._info_dir(handle) / "collection_info.json"
            info = json.loads(info_path.read_text()) if info_path.exists() else {}
            self._update_index(handle, info, cfg)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_distance_matrix(self, handle: str) -> DistanceMatrix:
        from safetensors.numpy import load_file

        coll_dir = self._collection_dir(handle)
        tensors = load_file(str(coll_dir / "distance_matrix.safetensors"))
        matrix = tensors["matrix"].astype(np.float64)
        meta = json.loads(tensors["_meta_json"].tobytes().decode("utf-8"))
        return DistanceMatrix(matrix=matrix, **meta)

    def load_geometry(
        self,
        handle: str,
        method: str,
        n_components: int | None = None,
        mds_kwargs: dict | None = None,
    ) -> GeometryResult:
        """Load one stored geometry.

        With *n_components* omitted the method must have exactly one stored
        dimension; otherwise the available options are listed rather than one
        being picked arbitrarily.
        """
        from safetensors.numpy import load_file

        coords_dir = self._collection_dir(handle) / "coordinates"
        stem_method = self.geometry_key(method, 1, mds_kwargs).rsplit("_", 1)[0]

        if n_components is None:
            matches = [
                (m, n) for m, n in self.list_geometries(handle) if m == stem_method
            ]
            if not matches:
                raise FileNotFoundError(
                    f"collection {handle} has no {method!r} geometry. Stored: "
                    f"{self.list_geometries(handle)}"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"collection {handle} stores {method!r} at "
                    f"{sorted(n for _, n in matches)} dimensions; pass "
                    "n_components to choose."
                )
            n_components = matches[0][1]

        path = (
            coords_dir
            / f"{self.geometry_key(method, n_components, mds_kwargs)}.safetensors"
        )
        if not path.exists():
            # Written before coordinates were keyed by dimension.
            legacy = coords_dir / f"{method}.safetensors"
            if not legacy.exists():
                raise FileNotFoundError(
                    f"no {method!r} geometry at {n_components}d for collection "
                    f"{handle}. Stored: {self.list_geometries(handle)}"
                )
            path = legacy

        tensors = load_file(str(path))
        coordinates = tensors["coordinates"]

        info = self.load_info(handle)
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

    def list_geometries(self, handle: str) -> list[tuple[str, int]]:
        """Return ``[(method, n_components), ...]`` stored for a collection.

        A geometry fitted at non-default ``mds_kwargs`` reports its method as
        ``mds@{hash}``, so it is listed separately rather than colliding with the
        default fit.
        """
        coords_dir = self._collection_dir(handle) / "coordinates"
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

    def load_info(self, handle: str) -> dict:
        """Return the collection_info.json dict, read from the collection level."""
        return json.loads(
            (self._info_dir(handle) / "collection_info.json").read_text()
        )

    def load_config(self, handle: str) -> dict:
        """Return the leaf ``config.json``: surrogate spec, per-model hashes, metric."""
        return json.loads((self._collection_dir(handle) / "config.json").read_text())

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_collections(self) -> list[str]:
        """Return every handle present in the cache.

        Walks ``{taxonomy}/{collection_key}/{metric}_{surrogate_key}`` rather than
        globbing, so a stray file or a ``_legacy`` directory of quarantined
        entries cannot be mistaken for a collection.
        """
        if not self._collections_dir.exists():
            return []
        out: list[str] = []
        for tax_dir in sorted(self._collections_dir.iterdir()):
            if not tax_dir.is_dir() or tax_dir.name.startswith("_"):
                continue
            for key_dir in sorted(tax_dir.iterdir()):
                if not key_dir.is_dir():
                    continue
                for leaf in sorted(key_dir.iterdir()):
                    if leaf.is_dir() and (leaf / "distance_matrix.safetensors").exists():
                        out.append(f"{tax_dir.name}/{key_dir.name}/{leaf.name}")
        return out

    # ------------------------------------------------------------------
    # Readable catalogue
    # ------------------------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self._collections_dir / "index.json"

    def load_index(self) -> dict[str, dict]:
        """Return the catalogue: ``{handle: summary}``.

        Two of the three handle components are content hashes, which are stable
        but unreadable.  This is how to find out what is in the cache without
        opening each collection in turn.
        """
        if not self.index_path.exists():
            return {}
        try:
            return json.loads(self.index_path.read_text())
        except json.JSONDecodeError:
            return {}

    def find(self, **criteria) -> list[str]:
        """Handles whose index entry matches every criterion.

        ``cc.find(taxonomy="structural", metric="cosine")`` is the usual form.
        ``collection_key`` and ``surrogate_key`` are also filterable, which is
        how "every metric over these same representations" is asked for.
        """
        out = []
        for handle, record in self.load_index().items():
            if all(record.get(k) == v for k, v in criteria.items()):
                out.append(handle)
        return sorted(out)

    def _update_index(self, handle: str, info: dict, config: dict) -> None:
        """Merge one collection's summary into index.json, atomically.

        Locked because several SLURM jobs can write different collections into
        the same cache at once, and a read-modify-write of a shared file is
        exactly where that would corrupt.
        """
        from filelock import FileLock

        self._collections_dir.mkdir(parents=True, exist_ok=True)
        taxonomy, collection_key, leaf = handle.split("/")
        metric, _, surrogate_key = leaf.rpartition("_")
        record = {
            "taxonomy": taxonomy or info.get("taxonomy"),
            "metric": metric,
            "collection_key": collection_key,
            "surrogate_key": surrogate_key,
            "label": info.get("label"),
            "slice": info.get("slice", {}),
            "n_models": len(info.get("model_entries", [])),
            "model_ids": [e["model_id"] for e in info.get("model_entries", [])],
            "geometries": config.get("geometries", []),
            "created_at": info.get("created_at"),
        }

        with FileLock(str(self._collections_dir / "index.lock")):
            index = self.load_index()
            index[handle] = record
            tmp = self._collections_dir / "index.json.tmp"
            tmp.write_text(json.dumps(index, indent=2, sort_keys=True))
            os.replace(tmp, self.index_path)
