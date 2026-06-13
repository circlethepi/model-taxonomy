from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np

from src.core.protocols import Taxonomy, DistanceMetric, ComputeBackend, ModelID
from src.core.representation import ModelRepresentation


class SlurmBackend(ComputeBackend):
    """Runs extraction as SLURM array jobs via submitit.

    Each model gets its own SLURM job (one GPU per job). The taxonomy is
    pickle-serialized automatically by submitit, so the only requirement is
    that Taxonomy and its dependencies are importable on the cluster nodes.

    Representations are collected after all jobs finish. Distance computation
    runs locally after all representations are collected, since it is
    CPU-only and typically fast relative to model inference.

    slurm_params examples:
        {"slurm_partition": "gpu", "slurm_gpus_per_task": 1,
         "slurm_mem_gb": 40, "timeout_min": 120}
    """

    def __init__(
        self,
        slurm_params: dict,
        results_dir: Path | str = Path("./slurm_jobs"),
        n_distance_jobs: int = 1,
    ) -> None:
        self.slurm_params = slurm_params
        self.results_dir = Path(results_dir)
        self.n_distance_jobs = n_distance_jobs

    def map_extract(
        self,
        taxonomy: Taxonomy,
        model_ids: Sequence[ModelID],
    ) -> list[ModelRepresentation]:
        try:
            import submitit
        except ImportError:
            raise ImportError(
                "submitit is required for SlurmBackend. "
                "Install it with: pip install submitit"
            )

        self.results_dir.mkdir(parents=True, exist_ok=True)
        executor = submitit.AutoExecutor(folder=str(self.results_dir))
        executor.update_parameters(**self.slurm_params)

        jobs = executor.map_array(taxonomy.extract, list(model_ids))
        return [job.result() for job in jobs]

    def map_distances(
        self,
        metric: DistanceMetric,
        representations: Sequence[ModelRepresentation],
    ) -> np.ndarray:
        from src.compute.local import LocalBackend

        local = LocalBackend(n_jobs=self.n_distance_jobs)
        return local.map_distances(metric, representations)
