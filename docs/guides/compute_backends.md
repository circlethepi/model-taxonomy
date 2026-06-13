# Compute Backends

The `ComputeBackend` handles two tasks: distributing surrogate extraction across models (Step 1), and parallelizing distance computation (Step 2). Both `LocalBackend` and `SlurmBackend` implement the same interface and are interchangeable.

## LocalBackend

Runs everything on the local machine.

```python
from src import LocalBackend

# Sequential — required when models share a single GPU
backend = LocalBackend(n_jobs=1)

# Parallel distance computation (CPU-only, safe to parallelize)
backend = LocalBackend(n_jobs=8)

# Use all available CPU cores
backend = LocalBackend(n_jobs=-1)
```

### When to use `n_jobs=1`

Always use `n_jobs=1` for extraction when models run on a GPU. Loading multiple models concurrently on one GPU will either exhaust memory or require careful sharding. The sequential loop is:

1. Load model A → extract → unload → 2. Load model B → extract → unload → ...

With `n_jobs=1`, extraction is sequential but distance computation (a CPU numpy operation) can still use multiple cores. To do both:

```python
# Extraction: sequential (n_jobs=1 applies to map_extract)
# Distances: parallel (n_jobs applies to map_distances too)
# These use the same n_jobs value — set to 1 if GPU, >1 for CPU-only models
backend = LocalBackend(n_jobs=1)
```

If you have CPU-only models (tiny models for testing, embedding models), `n_jobs > 1` is safe for extraction.

### For local development

`LocalBackend(n_jobs=1)` is the right choice for:
- Running quick experiments on a few small models
- Testing pipeline logic before scaling to a cluster
- Any situation where you have one GPU

---

## SlurmBackend

Submits extraction as a SLURM array job — one job per model. Each job loads its model on a dedicated GPU, runs inference, and returns the `ModelRepresentation`. After all jobs complete, distance computation runs locally.

```python
from src import SlurmBackend
from pathlib import Path

backend = SlurmBackend(
    slurm_params={
        "slurm_partition": "gpu",
        "slurm_gpus_per_task": 1,
        "slurm_mem_gb": 48,
        "timeout_min": 120,
        "slurm_cpus_per_task": 4,
    },
    results_dir=Path("./slurm_jobs"),
    n_distance_jobs=8,    # CPU cores for distance computation after collection
)
```

`slurm_params` is passed directly to `submitit.AutoExecutor.update_parameters()`. Any keyword argument accepted by submitit works here.

### How it works

```python
# Under the hood:
executor = submitit.AutoExecutor(folder="./slurm_jobs")
executor.update_parameters(**slurm_params)
jobs = executor.map_array(taxonomy.extract, model_ids)
representations = [job.result() for job in jobs]  # blocks until all jobs finish
```

The `taxonomy` object is pickle-serialized and sent to each SLURM node. This works automatically because `BehavioralTaxonomy` stores all configuration as plain Python objects (no open file handles, no live tensors).

After all jobs complete, distance computation runs on the submit node using `LocalBackend` internally.

### Requirements on the cluster

1. **Shared filesystem.** The `DiskCache` directory must be on a network filesystem visible to all nodes (e.g., NFS, Lustre). This allows completed jobs to write their representations to cache, and prevents duplicate computation if a job is retried.

2. **Package installed on all nodes.** The `taxonomy` environment must be active on compute nodes. If using conda:

   ```bash
   # In your SLURM job prologue or slurm_params:
   # "slurm_setup": ["conda activate taxonomy"]
   ```

   Or ensure the correct Python is on `PATH`.

3. **`HF_TOKEN` accessible on nodes.** For gated models, set the environment variable in the SLURM job or pass the token directly to `BehavioralTaxonomy`.

### Sizing jobs

| Model size | GPU memory | Recommended `slurm_mem_gb` | `timeout_min` |
|---|---|---|---|
| < 1B params | 2–4 GB | 16 | 30 |
| 1B–7B params | 6–14 GB | 32–48 | 60–120 |
| 7B–13B params | 14–26 GB | 64 | 120–180 |
| 13B–70B params | 26–140 GB | 80–160 | 180–360 |

For very large models, use `torch_dtype=torch.bfloat16` and `device_map="auto"` (the default in `BehavioralTaxonomy`) to spread across multiple GPUs. Request multiple GPUs with `slurm_gpus_per_task` accordingly.

### Local testing with `AutoExecutor`

`submitit.AutoExecutor` automatically falls back to a local executor when not running on a SLURM cluster. This lets you test the SLURM code path without a real cluster:

```python
# Same code; submitit runs jobs locally when SLURM is not available
backend = SlurmBackend(
    slurm_params={"timeout_min": 60},
    results_dir=Path("./local_slurm_test"),
)
result = analyzer.fit(model_ids)
```

### Handling failures

If a SLURM job fails (OOM, timeout, node failure), `job.result()` raises the exception. The cache prevents re-running successful jobs — only the failed model needs to be retried. Partially complete runs can be resumed by re-running `analyzer.fit()` with the same config and cache:

```python
# Models already in cache are skipped; only failed ones re-run
result = analyzer.fit(model_ids)
```

---

## Choosing a backend

| Situation | Recommended |
|---|---|
| Testing or small experiments (≤ 5 models, < 3B params) | `LocalBackend(n_jobs=1)` |
| Many small CPU-only models | `LocalBackend(n_jobs=-1)` |
| Moderate collection on a single multi-GPU node | `LocalBackend(n_jobs=1)` |
| Large collection of large models | `SlurmBackend(...)` |
| Reproducing a prior result with a cached run | Either (cache hit is fast) |
