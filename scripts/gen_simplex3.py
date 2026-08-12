#!/usr/bin/env python
"""Generate every config and sbatch script for the yahoo 3-group simplex experiment.

Ten yahoo topics are partitioned into three semantic groups, datasets are drawn at
every 25% composition of those groups, and the resulting mixtures are measured at
three levels against LoRA adapters trained on the corners and interiors.  Earlier
yahoo experiments varied a two-topic mixture, so the recovered taxonomy could only
be compared against a one-dimensional ground truth; a 2-simplex has vertices, edges
and an interior, and a distance matrix can be right about one and wrong about
another.

Why a generator rather than hand-written YAML: 16 proportions across four config
families, ten build shards and sixteen behavioral shards is ~50 files of
near-identical YAML whose only interesting content is three numbers per file.
Transposing two group weights in one of them would be invisible on review and would
silently mislabel a point of the simplex.  Here the group definitions and the
simplex enumeration exist once, and the whole tree is reproducible with one command.

Usage:
    python scripts/gen_simplex3.py                 # write experiments/simplex3/, jobs/simplex3/
    python scripts/gen_simplex3.py --list          # enumerate the proportions and exit
"""

from __future__ import annotations

import argparse
from math import gcd
from pathlib import Path

# ── The experiment's fixed parameters ──────────────────────────────────────────

#: Topic ids per semantic group.  Order matters only for readability.
GROUPS: dict[str, list[int]] = {
    "g1": [0, 6, 7, 9],   # politics & government, business & finance,
                          # society & culture, entertainment & music
    "g2": [1, 3, 4],      # science & mathematics, computers & internet,
                          # education & reference
    "g3": [2, 5, 8],      # sports, family & relationships, health
}

DATASET_ID = "yahoo_answers_topics"
CLASS_FIELD = "topic"

#: The training/embedding projection: a row is title + body + answer, so every
#: level projects the same text.  See docs/notes/TODO.md item 11 for what the
#: alternative cost last time.
TEXT_FIELDS = ["question_title", "question_content", "best_answer"]
#: The query-set-B projection: question only, so the model must actually answer
#: rather than continue text that already contains the answer.
QUERY_B_FIELDS = ["question_title", "question_content"]

BASE_MODEL = "meta-llama/Llama-3.1-8B"
MODEL_SLUG = BASE_MODEL.replace("/", "--")

#: Absolute, so a job started from a worktree still writes into the one shared
#: cache rather than growing a second copy that disappears with the worktree.
REPO = "/weka/scratch/cpriebe1/MO/model-taxonomy"
CACHE_DIR = f"{REPO}/results/shared_cache"
OUTPUT_DIR = f"{REPO}/results/simplex3"

SEEDS = list(range(10))
N_SWEEP = "tens 3"          # -> [1, 10, 100, 1000]
TRAIN_N, TRAIN_SEED = 1000, 0
TOTAL_TRAIN_SAMPLES = 5000  # -> ceil(5000/16) = 313 steps -> 5008 samples seen
SAMPLES_SEEN = 5008
LORA_RANK, LORA_INIT_SEED = 16, 0

QUERY_N, QUERY_SEED = 100, 1
REPLICATES = 16
MAX_NEW_TOKENS = 128

TRAIN_SHARDS = 4
BEHAVIORAL_SHARDS = 8

GPU_PARTITIONS = "nvl,h100,l40s,a100"
CPU_PARTITION = "cpu"


# ── The simplex ────────────────────────────────────────────────────────────────

def proportions() -> list[tuple[str, tuple[int, int, int]]]:
    """``(name, (pct_g1, pct_g2, pct_g3))`` for every point of the 2-simplex.

    Fifteen grid points -- the compositions of 4 into 3 non-negative parts,
    ``C(6,2) = 15``: three pure vertices, six edge points, three edge midpoints and
    three interior points -- plus the even ``(1/3, 1/3, 1/3)`` mixture, which is not
    on a 25% grid and is the query draw.

    The even point's name rounds to ``033g1_033g2_033g3`` and sums to 99, not 100.
    Only the label rounds; the recipe carries exact 1:1:1 weights.
    """
    out: list[tuple[str, tuple[int, int, int]]] = []
    for i in range(5):
        for j in range(5 - i):
            k = 4 - i - j
            pct = (i * 25, j * 25, k * 25)
            out.append((name_for(pct), pct))
    out.append((name_for((33, 33, 33)), (33, 33, 33)))
    return out


def name_for(pct: tuple[int, int, int]) -> str:
    return "yahoo_" + "_".join(f"{p:03d}{g}" for p, g in zip(pct, GROUPS))


def weights_for(pct: tuple[int, int, int]) -> list[tuple[str, int]]:
    """``(group, integer weight)`` for the non-zero groups, in lowest terms.

    Entry weights are relative -- ``normalized_weights`` divides by the sum -- so
    2:1:1 and 0.50:0.25:0.25 are the same recipe.  Integers are used because they
    serialize exactly and a float ratio does not.

    Zero-weight groups are dropped rather than carried at ``weight: 0.0``.  That
    matches the existing ``yahoo_100t0_000t1`` convention, keeps each proportion's
    recipe_hash distinct via a differing entry set, and avoids filtering 1.4M rows
    to draw none.
    """
    nonzero = [(g, p) for g, p in zip(GROUPS, pct) if p > 0]
    divisor = 0
    for _, p in nonzero:
        divisor = gcd(divisor, p)
    return [(g, p // divisor) for g, p in nonzero]


# ── YAML emission ──────────────────────────────────────────────────────────────
#
# Written as text rather than via yaml.dump so the comments survive.  These files
# are read by people deciding whether a run meant what they think it meant, and a
# dumped mapping loses every reason.

def entry_yaml(group: str, weight: int, text_fields: list[str], indent: str = "      ") -> str:
    return (
        f"{indent}- dataset_id: {DATASET_ID}\n"
        f"{indent}  split: train\n"
        f"{indent}  weight: {weight}.0\n"
        f"{indent}  text_field: best_answer          # fallback projection; text_fields wins\n"
        f"{indent}  text_fields: [{', '.join(text_fields)}]\n"
        f'{indent}  text_separator: "\\n"\n'
        f"{indent}  class_field: {CLASS_FIELD}\n"
        f"{indent}  class_filter: {GROUPS[group]}\n"
        f"{indent}  class_sampling: pooled           # draw from the union, not per-topic quotas\n"
    )


def dataset_block(name: str, pct: tuple[int, int, int], *,
                  sweep: str | None = None, seeds: list[int] | None = None,
                  n_samples: int | None = None, seed: int | None = None,
                  text_fields: list[str] | None = None) -> str:
    fields = text_fields or TEXT_FIELDS
    head = f"  - name: {name}\n    recipe_type: class_aware\n"
    if sweep is not None:
        head += f"    n_samples_sweep: {sweep}\n"
    if seeds is not None:
        head += f"    seeds: {seeds}\n"
    if n_samples is not None:
        head += f"    n_samples: {n_samples}\n"
    if seed is not None:
        head += f"    seed: {seed}\n"
    head += f"    # {pct[0]}% g1 / {pct[1]}% g2 / {pct[2]}% g3\n    entries:\n"
    return head + "".join(
        entry_yaml(g, w, fields) for g, w in weights_for(pct)
    )


EVEN = (33, 33, 33)
QUERY_A_NAME = name_for(EVEN)
QUERY_B_NAME = QUERY_A_NAME + "_qtc"


def query_blocks(which: str) -> str:
    """The query recipe, defined inline so make_queries can find it.

    Content-addressed, so this block hashes to the same recipe as the sweep's even
    mixture and reuses the draw the build step already made -- naming it here does
    not create a second sample.  Requested at ``n_samples == n_queries`` because n
    enters the sampler: a 100-row draw is not the first 100 rows of a 1000-row one.
    """
    if which == "a":
        return dataset_block(QUERY_A_NAME, EVEN, n_samples=QUERY_N, seed=QUERY_SEED)
    return dataset_block(QUERY_B_NAME, EVEN, n_samples=QUERY_N, seed=QUERY_SEED,
                         text_fields=QUERY_B_FIELDS)


def adapter_name(name: str) -> str:
    return (f"{name}_n{TRAIN_N}_s{TRAIN_SEED:02d}"
            f"_r{LORA_RANK}_i{LORA_INIT_SEED:02d}_b{SAMPLES_SEEN}")


def adapter_path(name: str) -> str:
    return f"{CACHE_DIR}/03_adapters/{MODEL_SLUG}/{adapter_name(name)}"


HEADER = """# GENERATED BY scripts/gen_simplex3.py -- do not edit by hand.
# Re-run the generator instead; hand edits are lost and, worse, silently diverge
# from the other ~50 files that were meant to agree with this one.
#
"""


def preamble(name: str) -> str:
    return (
        f"name: {name}\n"
        f"output_dir: {OUTPUT_DIR}\n"
        f"cache_dir: {CACHE_DIR}\n\n"
    )


EMBEDDER = """      embedder:
        model_name: nomic-ai/nomic-embed-text-v1.5
        normalize_embeddings: true
        trust_remote_code: true
        # The literal prefix nomic wants is "search_document: ". Note the existing
        # surrogates spell this `document`, which resolves to the same prefix and
        # the same vectors -- but prompt_name is inside config_dict(), so the two
        # spellings are different embedder_hashes. Reuse this spelling or re-embed.
        prompt_name: search_document
"""


def write_sweep(seed: int) -> str:
    """Build + embed config for one seed: all 16 proportions at 4 sizes."""
    body = HEADER + (
        f"# Seed {seed} of the simplex sweep: 16 proportions x {{1,10,100,1000}} = 64 draws.\n"
        f"# Sharded by seed rather than by proportion so each shard still touches all\n"
        f"# three group filters -- source_registry memoises them, so that is three\n"
        f"# filter passes per shard instead of one per draw.\n#\n"
    )
    if seed == QUERY_SEED:
        body += (
            f"# This shard also builds query set B ({QUERY_B_NAME}), the question-only\n"
            f"# ablation. Composition feeds recipe_hash, so it is a distinct recipe -- but\n"
            f"# text_fields never touches the sampler's RNG and the seed is not derived\n"
            f"# from the hash, so it selects the SAME {QUERY_N} rows as query set A. The two\n"
            f"# are the same questions projected two ways, which is what makes the\n"
            f"# ablation controlled rather than two different samples.\n#\n"
        )
    body += "\n" + preamble(f"simplex3_sweep_s{seed:02d}")
    body += "datasets:\n"
    body += "\n".join(
        dataset_block(name, pct, sweep=N_SWEEP, seeds=[seed])
        for name, pct in proportions()
    )
    if seed == QUERY_SEED:
        body += "\n" + query_blocks("b")
    body += f"""
base_models:
  - {BASE_MODEL}

fine_tuning:
  enabled: false

# One (1, 768) centroid per draw. The surrogate is authored, not derived -- the
# full (N, 768) matrix is never stored and `mean` is not invertible -- so adding a
# second representation later means re-embedding all 640.
extraction:
  models: []
  device: cuda
  taxonomies:
    functional:
      enabled: false
    behavioral:
      enabled: false
    dataset_embedding:
      enabled: true
      representation: mean
{EMBEDDER}"""
    return body


def write_train(shard: int, names: list[str]) -> str:
    body = HEADER + (
        f"# Training shard {shard} of {TRAIN_SHARDS}: {len(names)} adapters.\n"
        f"# Only n={TRAIN_N}, seed={TRAIN_SEED} is trained on. total_train_samples\n"
        f"# {TOTAL_TRAIN_SAMPLES} quantizes UP to a step boundary at effective batch 16:\n"
        f"# ceil({TOTAL_TRAIN_SAMPLES}/16) = 313 steps, so {SAMPLES_SEEN} samples seen and the\n"
        f"# adapter is named _b{SAMPLES_SEEN}. Every LoRA parameter matches the existing 3B\n"
        f"# adapters; only the base model and the data differ.\n"
        f"#\n"
        f"# max_seq_length 512 is held for consistency, but the composition is three\n"
        f"# fields here rather than two, so truncation is more frequent than in the\n"
        f"# question_title + best_answer runs. Flagged, not changed.\n\n"
    ) + preamble(f"simplex3_train_shard{shard}")
    body += "datasets:\n"
    body += "\n".join(
        dataset_block(name, pct, sweep=f"[{TRAIN_N}]", seeds=[TRAIN_SEED])
        for name, pct in proportions() if name in names
    )
    body += f"""
base_models:
  - {BASE_MODEL}

fine_tuning:
  enabled: true
  datasets:
"""
    body += "".join(f"    - {n}\n" for n in names)
    body += f"""  n_samples: {TRAIN_N}
  seed: {TRAIN_SEED}
  lora_rank: {LORA_RANK}
  lora_alpha: 32
  target_modules: [q_proj, k_proj, v_proj, o_proj]
  lora_dropout: 0.05
  lora_init_seed: {LORA_INIT_SEED}
  learning_rate: 2.0e-4
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  max_seq_length: 512
  torch_dtype: float16
  total_train_samples: {TOTAL_TRAIN_SAMPLES}

extraction:
  models: []
  taxonomies:
    functional:
      enabled: false
    behavioral:
      enabled: false
    dataset_embedding:
      enabled: false
"""
    return body


LEVEL_DEFAULTS_NOTE = """  # All three levels carry an explicit `enabled`. Omitting a key does NOT mean
  # "skip": functional and behavioral default to true while dataset_embedding
  # defaults to false (extract_reprs.py:79, 97, 114). A behavioral shard that
  # merely omitted `functional` would re-extract every activation, eight times
  # over, racing the one job that is supposed to do it.
"""


def write_extract(level: str, query: str, shard: int | None, names: list[str]) -> str:
    qname = QUERY_A_NAME if query == "a" else QUERY_B_NAME
    qdesc = ("title + content + answer, matching the training composition"
             if query == "a" else "title + content only, question-only ablation")
    label = f"simplex3_{level}_{query}" + ("" if shard is None else f"_shard{shard}")
    body = HEADER + (
        f"# {level.capitalize()} extraction over query set {query.upper()} ({qdesc}).\n"
        f"# {len(names)} adapter(s)"
        + ("" if shard is None else f", shard {shard} of {BEHAVIORAL_SHARDS}")
        + ".\n#\n"
    )
    if level == "functional":
        body += (
            "# One job for all 16: HFInferenceTaxonomy loads the base model once and\n"
            "# swaps adapters onto it, so 16 models amortize a single 8B load, and\n"
            "# input-mode extraction is one forward pass per query with no decoding.\n\n"
        )
    else:
        body += (
            f"# Sharded because this is the expensive half: 16 adapters x {QUERY_N} queries\n"
            f"# x {REPLICATES} replicates x {MAX_NEW_TOKENS} tokens is ~25,600 generations, ~4 GPU-hours.\n"
            f"# Shards write to disjoint adapter directories. Paying the 8B load eight\n"
            f"# times buys eight short slots that actually backfill.\n\n"
        )
    body += preamble(label)
    body += "datasets:\n" + query_blocks(query)
    body += f"\nbase_models:\n  - {BASE_MODEL}\n\nfine_tuning:\n  enabled: false\n\n"
    body += "extraction:\n  models:\n"
    body += "".join(f"    - {adapter_path(n)}\n" for n in names)
    body += f"""  queries_dataset: {qname}
  n_queries: {QUERY_N}
  device: cuda
  torch_dtype: float16
"""
    if level == "behavioral":
        body += (
            "  # Held at 2, and this is not cosmetic: one RNG generator serves a whole\n"
            "  # generate() call, so under sampling a different batch_size gives\n"
            "  # different text at the same seed. batch_size is deliberately OUT of the\n"
            "  # cache key, so a shard run at another value would silently collide with\n"
            "  # a correct entry rather than fail. Do not retune per GPU.\n"
            "  batch_size: 2\n"
        )
    body += "\n  taxonomies:\n" + LEVEL_DEFAULTS_NOTE
    if level == "functional":
        body += f"""    functional:
      enabled: true
      activation_mode: input      # input only; no generation activations
      layer_indices: null         # every hidden state: 33 for an 8B
      pooling: mean
      normalize_activations: true
      batch_size: 16
    behavioral:
      enabled: false
    dataset_embedding:
      enabled: false
"""
    else:
        body += f"""    functional:
      enabled: false
    dataset_embedding:
      enabled: false
    behavioral:
      enabled: true
      max_new_tokens: {MAX_NEW_TOKENS}
      replicates: {REPLICATES}
      do_sample: true
      temperature: 1.0
      top_p: 1.0
      top_k: null
      generation_seed: 0
{EMBEDDER}"""
    return body


# ── sbatch emission ────────────────────────────────────────────────────────────

def sbatch(job: str, partition: str, gpu: bool, mem_gb: int, time: str,
           command: str, log_dir: str) -> str:
    gres = "#SBATCH --gres=gpu:1\n" if gpu else ""
    return f"""#!/bin/bash
# GENERATED BY scripts/gen_simplex3.py -- do not edit by hand.
#SBATCH --job-name={job}
#SBATCH --partition={partition}
{gres}#SBATCH --mem={mem_gb}G
#SBATCH --cpus-per-task=8
#SBATCH --time={time}
#SBATCH --output={log_dir}/{job}-%j.out

set -euo pipefail
source ~/.bashrc
conda activate taxonomy-env
cd {REPO}

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/weka/home/mohata1/scratchcpriebe1/MO/huggingface_cache

{command}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true",
                        help="Print the proportions and their weights, write nothing.")
    parser.add_argument("--root", default=".", help="Repository root to write into.")
    args = parser.parse_args()

    props = proportions()
    if args.list:
        for name, pct in props:
            ratio = ":".join(str(w) for _, w in weights_for(pct))
            groups = ",".join(g for g, _ in weights_for(pct))
            print(f"  {name}   {pct}   {groups} = {ratio}")
        print(f"\n{len(props)} proportions "
              f"x {len(SEEDS)} seeds x 4 sizes = {len(props) * len(SEEDS) * 4} draws; "
              f"{len(props)} adapters")
        return

    root = Path(args.root)
    exp = root / "experiments" / "simplex3"
    jobs = root / "jobs" / "simplex3"
    logs = f"{OUTPUT_DIR}/logs"
    exp.mkdir(parents=True, exist_ok=True)
    jobs.mkdir(parents=True, exist_ok=True)

    written: list[str] = []

    def emit(path: Path, text: str) -> None:
        path.write_text(text)
        written.append(str(path))

    # 0. Prefetch the base model. CPU, first in the chain, so the one genuinely
    #    unknown dependency fails fast and fails cheap.
    emit(jobs / "00_prefetch.sh", sbatch(
        "s3_prefetch", CPU_PARTITION, False, 32, "1:00:00",
        f'python -c "\n'
        f'from huggingface_hub import snapshot_download\n'
        f"p = snapshot_download('{BASE_MODEL}', allow_patterns=["
        f"'*.json','*.safetensors','*.model','tokenizer*'])\n"
        f"print('cached at', p)\n"
        f'"',
        logs,
    ))

    # 1-2. Build (CPU) then embed (GPU), one shard per seed, one config for both.
    for seed in SEEDS:
        emit(exp / f"sweep_s{seed:02d}.yaml", write_sweep(seed))
        emit(jobs / f"01_build_s{seed:02d}.sh", sbatch(
            f"s3_build_s{seed:02d}", CPU_PARTITION, False, 96, "4:00:00",
            f"python scripts/build_datasets.py experiments/simplex3/sweep_s{seed:02d}.yaml",
            logs,
        ))
        emit(jobs / f"02_embed_s{seed:02d}.sh", sbatch(
            f"s3_embed_s{seed:02d}", GPU_PARTITIONS, True, 48, "1:00:00",
            f"python scripts/extract_reprs.py experiments/simplex3/sweep_s{seed:02d}.yaml"
            f" --taxonomy dataset_embedding",
            logs,
        ))

    # 3. Training, four adapters per shard.
    names = [n for n, _ in props]
    shards = [names[i::TRAIN_SHARDS] for i in range(TRAIN_SHARDS)]
    for i, shard_names in enumerate(shards):
        emit(exp / f"train_shard{i}.yaml", write_train(i, shard_names))
        emit(jobs / f"03_train_shard{i}.sh", sbatch(
            f"s3_train{i}", GPU_PARTITIONS, True, 80, "2:00:00",
            f"python scripts/run_experiment.py experiments/simplex3/train_shard{i}.yaml"
            f" --steps build finetune",
            logs,
        ))

    # 4-7. Extraction, both query sets.
    bshards = [names[i::BEHAVIORAL_SHARDS] for i in range(BEHAVIORAL_SHARDS)]
    for q, num in (("a", 4), ("b", 6)):
        emit(exp / f"functional_{q}.yaml", write_extract("functional", q, None, names))
        emit(jobs / f"0{num}_functional_{q}.sh", sbatch(
            f"s3_func_{q}", GPU_PARTITIONS, True, 64, "2:00:00",
            f"python scripts/extract_reprs.py experiments/simplex3/functional_{q}.yaml"
            f" --taxonomy functional",
            logs,
        ))
        for i, shard_names in enumerate(bshards):
            emit(exp / f"behavioral_{q}_shard{i}.yaml",
                 write_extract("behavioral", q, i, shard_names))
            emit(jobs / f"0{num + 1}_behavioral_{q}_shard{i}.sh", sbatch(
                f"s3_behav_{q}{i}", GPU_PARTITIONS, True, 64, "1:30:00",
                f"python scripts/extract_reprs.py"
                f" experiments/simplex3/behavioral_{q}_shard{i}.yaml"
                f" --taxonomy behavioral",
                logs,
            ))

    emit(jobs / "submit_all.sh", submit_all())
    (jobs / "submit_all.sh").chmod(0o755)

    print(f"Wrote {len(written)} files:")
    print(f"  {len(list(exp.glob('*.yaml')))} configs in {exp}")
    print(f"  {len(list(jobs.glob('*.sh')))} scripts in {jobs}")
    print(f"\n{len(props)} proportions, {len(props) * len(SEEDS) * 4} draws, "
          f"{len(props)} adapters.")
    print(f"Submit with: bash {jobs}/submit_all.sh")


def submit_all() -> str:
    """The dependency chain, as a script rather than a paragraph of instructions."""
    return f"""#!/bin/bash
# GENERATED BY scripts/gen_simplex3.py -- do not edit by hand.
#
# Submits the whole experiment with afterok dependencies.
#
# CHECK THE QUEUE FIRST. The partition ordering and every wall in these scripts
# were chosen against a specific snapshot of free capacity, and a snapshot written
# into a config is stale by the time it runs:
#
#     sinfo -o "%P %a %D %t %G"
#     squeue --states=PD -o "%P %R" | sort | uniq -c
#     sinfo -T                      # migration reservations
#
# Then confirm the prefetch and build jobs actually START before trusting the rest
# of the chain -- everything downstream is afterok on them.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p {OUTPUT_DIR}/logs

sb() {{ sbatch --parsable "$@"; }}

PREFETCH=$(sb 00_prefetch.sh)
echo "prefetch      $PREFETCH"

BUILD=""
for s in {' '.join(f'{s:02d}' for s in SEEDS)}; do
  J=$(sb 01_build_s$s.sh)
  BUILD="$BUILD:$J"
  echo "build   s$s   $J"
done
BUILD=${{BUILD#:}}

for s in {' '.join(f'{s:02d}' for s in SEEDS)}; do
  J=$(sb --dependency=afterok:$BUILD 02_embed_s$s.sh)
  echo "embed   s$s   $J"
done

TRAIN=""
for i in {' '.join(str(i) for i in range(TRAIN_SHARDS))}; do
  J=$(sb --dependency=afterok:$PREFETCH:$BUILD 03_train_shard$i.sh)
  TRAIN="$TRAIN:$J"
  echo "train   $i     $J"
done
TRAIN=${{TRAIN#:}}

# Query set A is the primary result; B is the ablation. Both depend only on
# training, so all 18 extraction jobs are eligible at once and the queue orders
# them. If capacity is tight, the B behavioral shards are the drop candidate --
# they are half the GPU time here, and functional B still costs ~15 minutes.
J=$(sb --dependency=afterok:$TRAIN 04_functional_a.sh); echo "func    A     $J"
for i in {' '.join(str(i) for i in range(BEHAVIORAL_SHARDS))}; do
  J=$(sb --dependency=afterok:$TRAIN 05_behavioral_a_shard$i.sh)
  echo "behav   A$i    $J"
done
J=$(sb --dependency=afterok:$TRAIN 06_functional_b.sh); echo "func    B     $J"
for i in {' '.join(str(i) for i in range(BEHAVIORAL_SHARDS))}; do
  J=$(sb --dependency=afterok:$TRAIN 07_behavioral_b_shard$i.sh)
  echo "behav   B$i    $J"
done

echo
echo "Submitted. Watch with: squeue -u $USER -o '%.10i %.14j %.9P %.2t %.10M %R'"
"""


if __name__ == "__main__":
    main()
