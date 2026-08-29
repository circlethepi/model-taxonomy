# Experiment Suites

A **suite** is one configuration of the simplex3 experiment: which base model, which
dtype, which LoRA targets, how the rows are wrapped, how the work is sharded, how long
the jobs may take. `scripts/gen_simplex3.py` turns a suite into every YAML config and
sbatch script the experiment needs.

## The split: what the experiment is, versus how one run is configured

| Lives in | Holds |
|---|---|
| `scripts/gen_simplex3.py` | What the experiment **is** — the topic groups, the simplex enumeration, the sample counts, the seeds, the replicate count |
| `src/experiments/suite.py::Suite` | How one **run** is configured — base model, dtype, targets, prompt format, sharding, walls, memory, partitions |
| `src/models/profile.py` | What a **checkpoint** is — dtype, LoRA targets, prompt-format defaults, template pin |

Change a `Suite` field and you get the same experiment on different hardware or a
different model. Change a generator constant and you get a *different experiment*.

Why a generator rather than hand-written YAML: 16 proportions across four config
families, ten build shards and sixteen behavioral shards is ~50 files of near-identical
YAML whose only interesting content is three numbers per file. Transposing two group
weights in one of them would be invisible on review and would silently mislabel a point
of the simplex.

## Usage

```bash
python scripts/gen_simplex3.py                 # experiments/simplex3/,      jobs/simplex3/
python scripts/gen_simplex3.py --suite qwen    # experiments/simplex3_qwen/, jobs/simplex3_qwen/
python scripts/gen_simplex3.py --list          # enumerate the proportions and exit
```

**The default suite must regenerate byte-for-byte.** That is the regression test that
lifting the Llama values out of the script changed nothing:

```bash
python scripts/gen_simplex3.py && git diff --exit-code experiments/simplex3 jobs/simplex3
```

## Registered suites

| Suite | Base model | Query sets | Notable |
|---|---|---|---|
| `llama` (default) | `meta-llama/Llama-3.1-8B` | `full_context`, `question_only` | The original suite. `Suite()` with no arguments. |
| `qwen` | `Qwen/Qwen3.5-4B` | `question_only` | Chat prompts, log-prob level, generation-mode activations, a 10-point temperature sweep. |
| `llama3i` | `meta-llama/Llama-3.1-8B-Instruct` | `question_only` | Chat prompts at full parity with `qwen`. The missing cell: `simplex3` and `simplex3_qwen` differ in *both* model family and prompting regime, so this fixes one at a time — against `llama` it isolates base vs instruct, against `qwen` it isolates model family. |

`Suite.for_model(base_model, **overrides)` retargets a suite at another checkpoint,
pulling dtype, LoRA targets and prompt-format defaults from the model's
[profile](model_profiles.md) rather than taking them as arguments. Anything in
`overrides` wins — that is what the generator's `--torch-dtype` and `--target-modules`
flags use.

## Fields worth knowing

`Suite` is a frozen dataclass; it is read all over the generator and a late mutation
would silently produce files that disagree with each other. Defaults are exactly the
values the Llama suite ran under.

**Identity and naming**

- `tag` — `""` for the original suite, so `results/simplex3` and `experiments/simplex3`
  keep their names. Anything else suffixes them (`_qwen`).
- `query_sets` — `full_context` is title+content+answer, `question_only` is
  title+content. **Named rather than lettered** because the two sets' roles invert
  between a raw suite and a chat suite: under chat + completion-only loss, the
  question-only set is the in-distribution one, and a positional letter cannot carry
  that.
- `job_tokens` — the short name each query set gets in emitted job names. Kept separate
  from `query_sets` so the Llama suite's `s3_func_b` and `06_functional_b.sh` stay
  exactly as they are; those names are already in `sacct` history and log filenames.
- `per_device_train_batch_size` × `gradient_accumulation_steps` — the **effective batch
  must stay 16**. It determines `steps_for_budget` and therefore the `_b5008` suffix
  every adapter is named for, which is what the cross-suite join keys on.

**Which jobs get emitted**

All four of these default **off**, and that is the regression test rather than a
preference: `Suite()` must still regenerate the Llama tree byte-for-byte, so a level
added after those files were written has to be invisible until a suite asks for it.

- `emit_logprob_jobs` — the [log-probability level](logprob_taxonomy.md) and the
  temperature sweep built on it.
- `emit_gen_activation_job` — generation-mode activations, greedy only. Separate from
  the flag above because it needs no new code: `FunctionalTaxonomy` already hardcodes
  `do_sample=False`, so one decoding point cannot collide with anything.
- `temperature_sweep` — the temperatures to sweep, as a tuple. Empty emits no sweep.
  `T=0` is not in it: greedy is already on disk and is that point of the surface.
- `sweep_replicates` (8) and `sweep_shards` (4) — half the main run's R, which buys ten
  sweep points for the cost of five existing runs; four adapters per shard lands at the
  same wall the eight-shard R=16 run was sized against.

`emit_embed_jobs` / `emit_build_job` default on. The standalone build job is a fail-fast
gate for the *embedding* sweeps only — every other job runs `--steps build …` and writes
the recipes it needs.

**Cluster**

`train_time`, `behav_time`, `func_time`, `greedy_time`, `logprob_time`, `func_gen_time`,
`train_mem_gb`, `extract_mem_gb`, `gpu_partitions`. These carry per-suite comments
explaining their values — the Qwen suite needs more wall despite half the parameters
because 24 of its 32 layers take the slow torch fallback without flash-linear-attention,
and more host RAM because a 248,320-vocab tied `lm_head` materializes a
`(4, 512, 248320)` logits tensor.

> `gpu_partitions` lost `nvl` on 2026-08-26, when that partition stopped existing.
> `sbatch` rejects the whole list with `invalid partition specified: nvl` rather than
> skipping the dead name, so every GPU job in the tree failed at submit. `h200` took its
> place as the large-HBM tier.

**Deliberately not in `Suite` yet:** job emission itself. Generalizing sharding and
dependency chains into reusable objects is worth doing, but designing that abstraction
against a single worked example tends to produce the wrong seams.

## Smoke-testing a new base model

```bash
python scripts/smoke_base_model.py experiments/simplex3_qwen/train_shard0.yaml
```

Adding a base model has one failure mode worth real fear: the checkpoint loads, training
runs, the loss looks plausible, and **the decoder was randomly initialized the whole
time**. `from_pretrained` reports missing keys and then carries on, so nothing downstream
would notice. Test 2 is that check, and it is the reason the script exists.

| Stage | Checks |
|---|---|
| `t_template` (1b) | The profile matches the tokenizer, and the chat template renders one row to the same bytes twice |
| `t_load` | The checkpoint loads under the suite's dtype, and no weights were randomly initialized |
| `t_lora` | LoRA attaches to every layer it should, and the trainable-parameter count matches the profile's `expected_lora_params` |
| `t_train` | A short real training run through `finetune_all` |
| `t_generate` | A real `BehavioralTaxonomy` extraction: closure rate for a reasoning model, termination rate for one without |

Stages are ordered so each runs only if the one it depends on passed, and so the two
decisions that change what gets submitted — *does LoRA reach every layer*, *do
generations finish inside the budget* — are answered before the shards go out rather
than after. It runs the **real** code paths rather than reimplementing them: a smoke
test that exercises a parallel implementation proves nothing about the one that will run.

Stage 1b needs no GPU, so it runs first. It exists because `chat_template_sha` hashes
the *template*, not its output: a template that interpolates a date renders different
training prompts on different days while the pin still matches, which would make
adapters incomparable with nothing in any config to show it.

Nothing in the script names a model. It branches on values derived from the checkpoint —
a model with a `</think>` token is instrumented for closure, one without for termination
— and reads the two things that cannot be derived (`expected_lora_params`,
`excluded_lora_modules`) from the resolved profile. It is a **chat-suite** pre-flight:
`t_train` asserts completion-only loss and a recorded template hash, so it does not apply
to the raw `llama` suite.

`01_smoke.sh` is the one hand-written job file in each suite's tree, deliberately: a
smoke run is a gate on *adding a model*, not a step in the experiment, so it is not
generated and `submit_all.sh` does not reference it. Copy it when adding a suite.

## Adding a base model

1. Write `src/models/profiles/<family>.py` and add it to `PROFILES` — see
   [Model Profiles](model_profiles.md).
2. Add a `Suite(...).for_model("<org>/<model>")` entry to `SUITES` in
   `scripts/gen_simplex3.py`.
3. `python scripts/gen_simplex3.py --suite <name>`.
4. Copy `jobs/simplex3_<other>/01_smoke.sh`, adjusting the job name, partitions, log
   path and config path. It is not generated — see above. Then
   `mkdir -p .../results/simplex3_<name>/logs`: `submit_all.sh` creates it, but the
   prefetch and smoke jobs are submitted ahead of `submit_all.sh` by design, and
   SLURM fails a job whose `--output` path does not resolve with exit 53 and no log
   file, which is indistinguishable from nothing having happened.
5. Run it: `python scripts/smoke_base_model.py experiments/simplex3_<name>/train_shard0.yaml`.
6. Submit via `jobs/simplex3_<name>/submit_all.sh`.

Step 1 is where the care goes. `resolve` picks the **longest matching prefix**, so check
that a new profile neither captures a checkpoint that belongs to an existing one nor is
captured *by* one — `meta-llama/Llama-3.1-8B-Instruct` starts with `meta-llama/Llama-3`,
and getting that wrong emits a raw suite instead of a chat suite. `check_analysis.py`
pins both directions.

## See also

- [Model Profiles and Prompt Formats](model_profiles.md)
- [Compute Backends](compute_backends.md) — local execution and SLURM setup
- [Visualization](visualization.md) — regenerating the figure suite once the jobs land
