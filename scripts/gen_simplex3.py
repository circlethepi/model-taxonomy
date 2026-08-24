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

One generator, several suites: what the experiment *is* lives here, and what one
*run* of it is configured as lives in :class:`src.experiments.suite.Suite`.  A
second base model is a Suite entry plus a model profile, not a forked copy of
this file -- forking would duplicate the simplex enumeration and the level
defaults, and the copies would drift the first time a shard count changed.

The default suite must regenerate byte-for-byte::

    python scripts/gen_simplex3.py && git diff --exit-code experiments/simplex3 jobs/simplex3

Usage:
    python scripts/gen_simplex3.py                 # write experiments/simplex3/, jobs/simplex3/
    python scripts/gen_simplex3.py --suite qwen    # write experiments/simplex3_qwen/, jobs/simplex3_qwen/
    python scripts/gen_simplex3.py --list          # enumerate the proportions and exit
"""

from __future__ import annotations

import argparse
import sys
from math import gcd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.suite import Suite  # noqa: E402

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
#: The question-only projection: question only, so the model must actually answer
#: rather than continue text that already contains the answer.
#:
#: Named rather than lettered.  The two query sets' *roles* invert between a raw
#: suite and a chat suite -- under a chat template with completion-only loss the
#: training prompt is the question, so the question-only set becomes the
#: in-distribution probe and the full-context one the contaminated one, the
#: reverse of the raw suite -- and a positional letter cannot carry that.
#:
#: Still needed on the raw path even though a chat suite expresses the same list
#: as ``prompt_format.user_fields``: this is what renders the Llama suite's
#: ``text_fields``, and that output must stay byte-identical.
QUERY_QUESTION_ONLY_FIELDS = ["question_title", "question_content"]

#: Measured on a 1000-row draw of the even mixture at seed 1:
#:     question_title      empty in   0.0%
#:     question_content    empty in  46.3%
#:     best_answer         empty in   2.2%
#: So `question_content` is absent from nearly half of yahoo, and query set B is
#: a bare title for about half its prompts.  That does not invalidate the
#: ablation -- B is still question-only -- but B's prompts are substantially
#: shorter than A's and about half carry no body at all, which belongs in any
#: caption comparing the two.  It also means the three-field training
#: composition is effectively title+answer for half the corpus, i.e. much closer
#: to the existing yahoo_qa adapters than the field list suggests.
FIELD_EMPTINESS_NOTE = "question_content is empty in ~46% of yahoo rows"

#: Absolute, so a job started from a worktree still writes into the one shared
#: cache rather than growing a second copy that disappears with the worktree.
REPO = "/weka/scratch/cpriebe1/MO/model-taxonomy"
#: Not a Suite field: the cache is shared *between* suites on purpose.  Draws and
#: dataset embeddings are model-free, so a second base model reuses them rather
#: than re-deriving 640 centroids.
CACHE_DIR = f"{REPO}/results/shared_cache"

#: Which configuration is being generated.  Rebound by ``main()`` from --suite;
#: the default is the Llama run, whose emitted files must not change.
SUITE = Suite()


def output_dir() -> str:
    return f"{REPO}/results/simplex3{SUITE.suffix}"

SEEDS = list(range(10))
N_SWEEP = "tens 3"          # -> [1, 10, 100, 1000]
TRAIN_N, TRAIN_SEED = 1000, 0
TOTAL_TRAIN_SAMPLES = 5000  # -> ceil(5000/16) = 313 steps -> 5008 samples seen
SAMPLES_SEEN = 5008
LORA_RANK, LORA_INIT_SEED = 16, 0

QUERY_N, QUERY_SEED = 100, 1
REPLICATES = 16
MAX_NEW_TOKENS = 128

#: Sampled runs hold batch_size at 2 because one RNG stream serves a whole
#: generate() call, so batch shape determines the text (first-order) and the value
#: is excluded from the cache key.  **Greedy is not subject to that**: no RNG is
#: seeded at all (`src/taxonomy/behavioral.py:304`), and batch size only flips
#: argmax on fp16 near-ties -- a last-bit effect, measured at 6/8 sequences
#: byte-identical between batch 1 and batch 8.
#:
#: That freedom is the whole cost argument. Decode time is dominated by the number
#: of generate() calls x max_new_tokens sequential steps, not by how wide each
#: batch is, so greedy at batch_size 2 would cost nearly what R=16 costs while
#: producing a sixteenth of the text. At 16 it is ~7 calls per adapter instead of
#: 50, which is what makes this job cheap.
GREEDY_BATCH_SIZE = 16

TRAIN_SHARDS = 4
BEHAVIORAL_SHARDS = 8

#: GPU partition order is a Suite field -- it is a property of what the run needs
#: from the hardware, not of the experiment.  CPU stays here: nothing about it
#: varies by model.
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
QUERY_FULL_CONTEXT_NAME = name_for(EVEN)
QUERY_QUESTION_ONLY_NAME = QUERY_FULL_CONTEXT_NAME + "_qtc"


def query_blocks(which: str) -> str:
    """The query recipe, defined inline so make_queries can find it.

    Content-addressed, so this block hashes to the same recipe as the sweep's even
    mixture and reuses the draw the build step already made -- naming it here does
    not create a second sample.  Requested at ``n_samples == n_queries`` because n
    enters the sampler: a 100-row draw is not the first 100 rows of a 1000-row one.

    Under a chat suite the ``text_fields`` here stop deciding the prompt -- the
    prompt is rendered from ``prompt_format.user_fields`` -- but the block still
    selects the rows, so it stays as it is.
    """
    if which == "full_context":
        return dataset_block(QUERY_FULL_CONTEXT_NAME, EVEN, n_samples=QUERY_N, seed=QUERY_SEED)
    return dataset_block(QUERY_QUESTION_ONLY_NAME, EVEN, n_samples=QUERY_N, seed=QUERY_SEED,
                         text_fields=QUERY_QUESTION_ONLY_FIELDS)


def adapter_name(name: str) -> str:
    """The adapter's directory leaf.

    The ``_f{format_id}`` suffix appears only for a non-raw prompt format, so
    every existing adapter name is unchanged, and two suites that wrap the same
    base model differently cannot land on the same path.  Downstream joins strip
    it the way they already strip the ``_b{samples}`` budget token.
    """
    stem = (f"{name}_n{TRAIN_N}_s{TRAIN_SEED:02d}"
            f"_r{LORA_RANK}_i{LORA_INIT_SEED:02d}_b{SAMPLES_SEEN}")
    return f"{stem}_f{format_id()}" if format_id() else stem


def format_id() -> str | None:
    """The current suite's prompt-format id, or None when raw."""
    from src.datasets._chat_projection import PromptFormat

    return PromptFormat.from_config(prompt_format_block()).format_id()


def prompt_format_block() -> dict | None:
    """The suite's ``prompt_format`` mapping, with the field lists filled in.

    The user fields are the question-only projection and the answer field is the
    answer column, so a chat suite's training prompt is the same text the
    question-only query set is -- which is what makes the training shape and the
    extraction shape the same shape.
    """
    if not SUITE.prompt_format:
        return None
    return {
        **SUITE.prompt_format,
        "user_fields": list(QUERY_QUESTION_ONLY_FIELDS),
        "answer_fields": [TEXT_FIELDS[-1]],
    }


def adapter_path(name: str) -> str:
    return f"{CACHE_DIR}/03_adapters/{SUITE.model_slug}/{adapter_name(name)}"


def prompt_format_yaml() -> str:
    """The ``prompt_format:`` block, or nothing at all for a raw suite.

    Emitting nothing is what keeps the Llama YAML byte-identical; an absent block
    reads as ``PromptFormat()`` on both the training and the extraction side.
    """
    block = prompt_format_block()
    if not block:
        return ""
    kwargs = block.get("chat_template_kwargs") or {}
    rendered = (
        "{}" if not kwargs
        else "{" + ", ".join(f"{k}: {str(v).lower()}" for k, v in sorted(kwargs.items())) + "}"
    )
    return (
        "# How a row becomes model input. Top-level, not under fine_tuning:, because\n"
        "# the SAME block builds the extraction prompts -- one key describing both\n"
        "# sides is what makes the training shape and the query shape unable to drift.\n"
        "prompt_format:\n"
        f"  format: {block['format']}\n"
        f"  user_fields: [{', '.join(block['user_fields'])}]\n"
        f"  answer_fields: [{', '.join(block['answer_fields'])}]\n"
        f"  chat_template_kwargs: {rendered}   # {{}} = the template's own defaults\n\n"
    )


HEADER = """# GENERATED BY scripts/gen_simplex3.py -- do not edit by hand.
# Re-run the generator instead; hand edits are lost and, worse, silently diverge
# from the other ~50 files that were meant to agree with this one.
#
"""


def preamble(name: str) -> str:
    return (
        f"name: {name}\n"
        f"output_dir: {output_dir()}\n"
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
            f"# This shard also builds the question-only query set ({QUERY_QUESTION_ONLY_NAME}).\n"
            f"# Composition feeds recipe_hash, so it is a distinct recipe -- but\n"
            f"# text_fields never touches the sampler's RNG and the seed is not derived\n"
            f"# from the hash, so it selects the SAME {QUERY_N} rows as the full-context set.\n"
            f"# The two are the same questions projected two ways, which is what makes the\n"
            f"# ablation controlled rather than two different samples.\n#\n"
        )
    body += "\n" + preamble(f"simplex3_sweep_s{seed:02d}")
    body += "datasets:\n"
    body += "\n".join(
        dataset_block(name, pct, sweep=N_SWEEP, seeds=[seed])
        for name, pct in proportions()
    )
    if seed == QUERY_SEED:
        body += "\n" + query_blocks("question_only")
    body += f"""
base_models:
  - {SUITE.base_model}

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


def write_embed_matrix() -> str:
    """Re-embed the 16 trained draws, keeping every per-document vector.

    ``write_sweep`` stores only the ``mean`` centroid, and a `02` surrogate is
    authored rather than derived (``docs/notes/dataset_embedding_layout.md`` §4),
    so a second representation costs a re-embed.  That note's warning is about
    the *whole* cache -- 640 draws, 1,984,744 texts, 6.1 GB -- and does not apply
    here: this is the 16 draws the adapters were actually trained on, at the one
    size they were trained at, which is 16,000 texts and ~49 MB.

    The point of keeping the matrix is the metrics it unlocks.  A ``(1, 768)``
    centroid is one point, so CKA, Bures-Wasserstein, MMD and the energy distance
    all have nothing to work with; the full cloud gives the dataset level the
    same metric coverage the other three levels already have, and lets a mixture
    be compared as a *distribution* over documents rather than as its average.

    The draws replay from ``SampledDatasetCache`` -- `01_datasets` holds the
    source indices for all 16 -- so this touches the embedder and not HuggingFace.
    The embedder block is the shared ``EMBEDDER`` literal, unchanged, which is
    what makes the new surrogate land beside the existing ``mean`` under the same
    ``{embedder_hash}/`` rather than in a directory of its own.
    """
    body = HEADER + (
        f"# Re-embed the {len(proportions())} trained draws at n={TRAIN_N}, "
        f"seed={TRAIN_SEED},\n"
        f"# keeping the full (n, 768) matrix instead of the (1, 768) centroid.\n"
        f"# Adds a second surrogate beside the existing `mean`; nothing is replaced.\n#\n"
    )
    body += "\n" + preamble(f"simplex3{SUITE.suffix}_embed_matrix")
    body += "datasets:\n"
    body += "\n".join(
        dataset_block(name, pct, n_samples=TRAIN_N, seed=TRAIN_SEED)
        for name, pct in proportions()
    )
    body += f"""
base_models:
  - {SUITE.base_model}

fine_tuning:
  enabled: false

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
      representation: matrix
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
  - {SUITE.base_model}

{prompt_format_yaml()}fine_tuning:
  enabled: true
  datasets:
"""
    body += "".join(f"    - {n}\n" for n in names)
    body += f"""  n_samples: {TRAIN_N}
  seed: {TRAIN_SEED}
  lora_rank: {LORA_RANK}
  lora_alpha: 32
  target_modules: [{', '.join(SUITE.target_modules)}]
  lora_dropout: 0.05
  lora_init_seed: {LORA_INIT_SEED}
  learning_rate: 2.0e-4
  per_device_train_batch_size: {SUITE.per_device_train_batch_size}
  gradient_accumulation_steps: {SUITE.gradient_accumulation_steps}
  max_seq_length: 512
  torch_dtype: {SUITE.torch_dtype}
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
    qname = QUERY_FULL_CONTEXT_NAME if query == "full_context" else QUERY_QUESTION_ONLY_NAME
    if SUITE.prompt_format:
        # Under a chat template with completion-only loss the training prompt IS
        # the question, so the roles are the reverse of the raw suite's.  Saying
        # so here rather than carrying the raw suite's phrasing forward, which
        # would be actively wrong.
        qdesc = ("title + content only, matching the training prompt; note "
                 + FIELD_EMPTINESS_NOTE + ", so ~half these prompts are a bare title")
    else:
        qdesc = ("title + content + answer, matching the training composition"
                 if query == "full_context" else
                 "title + content only, question-only ablation; note "
                 + FIELD_EMPTINESS_NOTE + ", so ~half these prompts are a bare title")
    token = SUITE.job_token(query)
    label = f"simplex3_{level}_{token}" + ("" if shard is None else f"_shard{shard}")
    body = HEADER + (
        f"# {level.capitalize()} extraction over the {query} query set ({qdesc}).\n"
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
    elif level == "greedy":
        body += (
            f"# The DETERMINISTIC control for the sampled runs: one continuation per\n"
            f"# query, no sampling variance, exactly reproducible. It is also the mode\n"
            f"# the pre-replicates measurements used, so it is directly comparable to\n"
            f"# the 2026-08-05 table in docs/notes/TODO.md.\n"
            f"#\n"
            f"# replicates MUST be 1. BehavioralTaxonomy raises on replicates > 1 with\n"
            f"# do_sample: false rather than storing R copies of one greedy continuation\n"
            f"# (src/taxonomy/behavioral.py:110-114).\n"
            f"#\n"
            f"# temperature / top_p / top_k / generation_seed are deliberately ABSENT.\n"
            f"# Under greedy they are nulled in the sampling hash (GREEDY_SAMPLING,\n"
            f"# behavioral.py:150-151) so a temperature that was never applied cannot\n"
            f"# change the digest; writing them here would only imply they matter.\n"
            f"# This also means greedy lands in its own cache entry and cannot collide\n"
            f"# with the R={REPLICATES} runs over the same adapters and draw.\n"
            f"#\n"
            f"# One job, unsharded: {QUERY_N} queries at batch {GREEDY_BATCH_SIZE} is ~7 generate()\n"
            f"# calls per adapter against ~50 for the sampled runs, so all 16 adapters\n"
            f"# finish in well under an hour.\n\n"
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
    body += f"\nbase_models:\n  - {SUITE.base_model}\n\n"
    body += prompt_format_yaml()
    body += "fine_tuning:\n  enabled: false\n\n"
    body += "extraction:\n  models:\n"
    body += "".join(f"    - {adapter_path(n)}\n" for n in names)
    body += f"""  queries_dataset: {qname}
  n_queries: {QUERY_N}
  device: cuda
  torch_dtype: {SUITE.torch_dtype}
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
    elif level == "greedy":
        body += (
            f"  # Raised to {GREEDY_BATCH_SIZE}, which is safe here and NOT safe for the sampled\n"
            f"  # runs. Greedy seeds no RNG at all, so batch size only flips argmax on\n"
            f"  # fp16 near-ties -- measured at 6/8 sequences byte-identical between\n"
            f"  # batch 1 and batch 8. Decode cost is driven by the number of generate()\n"
            f"  # calls, not their width, so this is the difference between ~7 calls per\n"
            f"  # adapter and ~50. Lower it if an 8B at {GREEDY_BATCH_SIZE} x (prompt + {MAX_NEW_TOKENS}) OOMs.\n"
            f"  batch_size: {GREEDY_BATCH_SIZE}\n"
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
    elif level == "greedy":
        body += f"""    functional:
      enabled: false
    dataset_embedding:
      enabled: false
    behavioral:
      enabled: true
      max_new_tokens: {MAX_NEW_TOKENS}
      replicates: 1               # enforced: >1 with do_sample:false raises
      do_sample: false            # THE difference from the sampled runs
{EMBEDDER}"""
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

# Source conda's profile script directly, NOT ~/.bashrc. A non-interactive shell
# -- which is what SLURM gives a batch script -- returns early from ~/.bashrc, so
# the `conda` shell function is never defined and `conda activate` dies with
# "Run 'conda init' before 'conda activate'". This is the incantation the
# existing jobs use (jobs/qa_pairs_train.sh:22).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate taxonomy-env
cd {REPO}

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/weka/home/mohata1/scratchcpriebe1/MO/huggingface_cache

{command}
"""


#: The configurations this generator knows how to emit.  ``llama`` is the suite
#: that already ran and must regenerate unchanged; ``qwen`` is the instruct run.
#: Per-model values are not restated here -- ``for_model`` reads them from the
#: model's profile, so a third base model is a profile file plus one line.
SUITES = {
    "llama": Suite(),
    "qwen": Suite(
        tag="qwen",
        query_sets=("question_only",),
        job_tokens={"question_only": "qonly"},
        # The build job gates the embedding sweeps, and this suite emits none:
        # dataset embeddings are model-free and the 640 centroids already exist.
        # Every remaining job runs `--steps build ...` and writes its own recipe.
        emit_embed_jobs=False,
        emit_build_job=False,
        job_prefix="s3q",
        # Roughly per-token parity with the 8B despite half the parameters: no
        # flash-linear-attention in the env, so 24 of 32 layers take the slow
        # torch fallback.  Behavioral is up most because thinking-on means every
        # sequence runs the full 128 decode steps.
        train_time="3:00:00",
        behav_time="2:30:00",
        func_time="1:30:00",
        greedy_time="2:00:00",
        # Host RAM. The 248,320-vocab tied lm_head is 15% of the model and
        # materializes a (4, 512, 248320) logits tensor; nothing at Llama's 128k
        # vocab needed this much headroom.
        train_mem_gb=96,
        extract_mem_gb=96,
        # bf16 throughput and HBM headroom first.
        gpu_partitions="h100,nvl,a100,l40s",
    ).for_model("Qwen/Qwen3.5-4B"),
}


def main() -> None:
    global SUITE

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="llama", choices=sorted(SUITES),
                        help="Which configuration to emit (default: llama).")
    parser.add_argument("--base-model", help="Override the suite's base model.")
    parser.add_argument("--tag", help="Override the suite's directory suffix.")
    parser.add_argument("--torch-dtype", help="Override the suite's dtype.")
    parser.add_argument("--target-modules",
                        help="Override the LoRA targets, comma-separated.")
    parser.add_argument("--list", action="store_true",
                        help="Print the proportions and their weights, write nothing.")
    parser.add_argument("--root", default=".", help="Repository root to write into.")
    args = parser.parse_args()

    from dataclasses import replace

    SUITE = SUITES[args.suite]
    if args.base_model:
        SUITE = SUITE.for_model(args.base_model)
    overrides = {}
    if args.tag is not None:
        overrides["tag"] = args.tag
    if args.torch_dtype:
        overrides["torch_dtype"] = args.torch_dtype
    if args.target_modules:
        overrides["target_modules"] = tuple(
            m.strip() for m in args.target_modules.split(",") if m.strip()
        )
    if overrides:
        SUITE = replace(SUITE, **overrides)

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
    slug = f"simplex3{SUITE.suffix}"
    exp = root / "experiments" / slug
    jobs = root / "jobs" / slug
    logs = f"{output_dir()}/logs"
    exp.mkdir(parents=True, exist_ok=True)
    jobs.mkdir(parents=True, exist_ok=True)

    written: list[str] = []

    def emit(path: Path, text: str) -> None:
        path.write_text(text)
        written.append(str(path))

    # 0. Prefetch the base model. CPU, first in the chain, so the one genuinely
    #    unknown dependency fails fast and fails cheap.
    #
    #    '*.jinja' is in the pattern list for models that ship their chat template
    #    as a standalone file. Qwen3.5 publishes it BOTH ways -- chat_template.jinja
    #    and the chat_template key of tokenizer_config.json, byte-identical -- so
    #    the tokenizer loads either way; fetching both keeps a prefetched cache
    #    resolving the template from the same file a fresh download would.
    emit(jobs / "00_prefetch.sh", sbatch(
        f"{SUITE.job_prefix}_prefetch", CPU_PARTITION, False, 32, "1:00:00",
        f'python -c "\n'
        f'from huggingface_hub import snapshot_download\n'
        f"p = snapshot_download('{SUITE.base_model}', allow_patterns=["
        f"'*.json','*.safetensors','*.model','tokenizer*','*.jinja','merges.txt'])\n"
        f"print('cached at', p)\n"
        f'"',
        logs,
    ))

    # 1. Build, one CPU job for all ten seeds.
    #
    #    Always run through run_experiment.py, never build_datasets.py directly:
    #    build_datasets does NOT expand n_samples_sweep/seeds, so invoked on its own
    #    it writes one recipe per *base* name and validates at the default n=100
    #    seed=42 instead of the 64 draws the block actually describes.
    #    run_experiment expands first (run_experiment.py:81).
    #
    #    This step is cheap -- it writes recipe JSON and nothing else. The draws
    #    themselves are materialised on demand by the sample cache during
    #    embedding, so this job is a fail-fast gate rather than the sampling work:
    #    640 recipe blocks are resolved and hashed before any GPU is held.
    if SUITE.emit_embed_jobs:
        for seed in SEEDS:
            emit(exp / f"sweep_s{seed:02d}.yaml", write_sweep(seed))
            emit(jobs / f"02_embed_s{seed:02d}.sh", sbatch(
                f"{SUITE.job_prefix}_embed_s{seed:02d}", SUITE.gpu_partitions, True,
                48, "2:00:00",
                f"python scripts/run_experiment.py experiments/{slug}/sweep_s{seed:02d}.yaml"
                f" --steps build extract --taxonomy dataset_embedding",
                logs,
            ))
    # 2. The `matrix` re-embed, emitted for every suite.
    #
    #    Unlike the sweep above this is *not* gated on emit_embed_jobs. That flag
    #    exists because the 640 centroids already exist and a second suite must
    #    not recompute them; this job authors a surrogate that does not exist yet
    #    under either suite. It is also model-free -- the same 16 recipes hash the
    #    same way whichever base model is being studied -- so whichever suite is
    #    run first satisfies both.
    emit(exp / "embed_matrix.yaml", write_embed_matrix())
    emit(jobs / "02_embed_matrix.sh", sbatch(
        f"{SUITE.job_prefix}_embed_matrix", SUITE.gpu_partitions, True,
        48, "2:00:00",
        f"python scripts/run_experiment.py experiments/{slug}/embed_matrix.yaml"
        f" --steps build extract --taxonomy dataset_embedding",
        logs,
    ))

    if SUITE.emit_build_job:
        emit(jobs / "01_build.sh", sbatch(
            f"{SUITE.job_prefix}_build", CPU_PARTITION, False, 32, "1:00:00",
            "\n".join(
                f"python scripts/run_experiment.py"
                f" experiments/{slug}/sweep_s{seed:02d}.yaml --steps build"
                for seed in SEEDS
            ),
            logs,
        ))

    # 3. Training, four adapters per shard.
    names = [n for n, _ in props]
    shards = [names[i::TRAIN_SHARDS] for i in range(TRAIN_SHARDS)]
    for i, shard_names in enumerate(shards):
        emit(exp / f"train_shard{i}.yaml", write_train(i, shard_names))
        emit(jobs / f"03_train_shard{i}.sh", sbatch(
            f"{SUITE.job_prefix}_train{i}", SUITE.gpu_partitions, True,
            SUITE.train_mem_gb, SUITE.train_time,
            f"python scripts/run_experiment.py experiments/{slug}/train_shard{i}.yaml"
            f" --steps build finetune",
            logs,
        ))

    # 4-7. Extraction, one pair of stages per query set.
    bshards = [names[i::BEHAVIORAL_SHARDS] for i in range(BEHAVIORAL_SHARDS)]
    for k, q in enumerate(SUITE.query_sets):
        tok = SUITE.job_token(q)
        num = 4 + 2 * k
        emit(exp / f"functional_{tok}.yaml", write_extract("functional", q, None, names))
        emit(jobs / f"0{num}_functional_{tok}.sh", sbatch(
            f"{SUITE.job_prefix}_func_{tok}", SUITE.gpu_partitions, True,
            SUITE.extract_mem_gb, SUITE.func_time,
            # --steps build extract, not extract alone: make_queries reads the
            # query recipe from {output_dir}/datasets/{queries_dataset}.recipe.json,
            # and this config's query block is named differently from the sweep's
            # expanded blocks, so nothing else writes that file.
            f"python scripts/run_experiment.py experiments/{slug}/functional_{tok}.yaml"
            f" --steps build extract --taxonomy functional",
            logs,
        ))
        for i, shard_names in enumerate(bshards):
            emit(exp / f"behavioral_{tok}_shard{i}.yaml",
                 write_extract("behavioral", q, i, shard_names))
            emit(jobs / f"0{num + 1}_behavioral_{tok}_shard{i}.sh", sbatch(
                f"{SUITE.job_prefix}_behav_{tok}{i}", SUITE.gpu_partitions, True,
                SUITE.extract_mem_gb, SUITE.behav_time,
                f"python scripts/run_experiment.py"
                f" experiments/{slug}/behavioral_{tok}_shard{i}.yaml"
                f" --steps build extract --taxonomy behavioral",
                logs,
            ))

    # 8. Greedy, one unsharded job per query set. ~7 generate() calls per adapter
    #    against ~50 for the sampled runs, so 16 adapters fit comfortably in an hour.
    for q in SUITE.query_sets:
        tok = SUITE.job_token(q)
        emit(exp / f"greedy_{tok}.yaml", write_extract("greedy", q, None, names))
        emit(jobs / f"08_greedy_{tok}.sh", sbatch(
            f"{SUITE.job_prefix}_greedy_{tok}", SUITE.gpu_partitions, True,
            SUITE.extract_mem_gb, SUITE.greedy_time,
            f"python scripts/run_experiment.py experiments/{slug}/greedy_{tok}.yaml"
            f" --steps build extract --taxonomy behavioral",
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
mkdir -p {output_dir()}/logs

sb() {{ sbatch --parsable "$@"; }}

PREFETCH=$(sb 00_prefetch.sh)
echo "prefetch      $PREFETCH"
{_submit_build()}
TRAIN=""
for i in {' '.join(str(i) for i in range(TRAIN_SHARDS))}; do
  J=$(sb --dependency=afterok:$PREFETCH{'':s}{_train_dep()} 03_train_shard$i.sh)
  TRAIN="$TRAIN:$J"
  echo "train   $i     $J"
done
TRAIN=${{TRAIN#:}}

# Extraction depends only on training, so every extraction job is eligible at
# once and the queue orders them.
{_submit_extract()}
# Greedy: the deterministic control, one job per query set. Its own cache entries
# (GREEDY_SAMPLING nulls the sampling fields), so it cannot collide with the
# R={REPLICATES} runs over the same adapters and draw.
for q in {' '.join(SUITE.job_token(q) for q in SUITE.query_sets)}; do
  J=$(sb --dependency=afterok:$TRAIN 08_greedy_$q.sh)
  echo "greedy  $q     $J"
done

echo
echo "Submitted. Watch with: squeue -u $USER -o '%.10i %.14j %.9P %.2t %.10M %R'"
"""


def _submit_build() -> str:
    """The build + embed stanza, or nothing when this suite reuses them."""
    if not SUITE.emit_build_job:
        return (
            "\n# No build or embed jobs: dataset embeddings are model-free "
            "(DatasetEmbeddingCache\n# is keyed on recipe and embedder, never on "
            "the model), so this suite reuses the\n# 640 cached centroids. Every "
            "job below runs `--steps build ...` and writes the\n# handful of "
            "recipes it needs itself.\n"
        )
    return f"""
# One cheap CPU job: it writes 640 recipe blocks and nothing else, so it is a
# fail-fast gate rather than the sampling work. Draws are materialised on demand
# by the sample cache during embedding.
BUILD=$(sb 01_build.sh)
echo "build         $BUILD"

for s in {' '.join(f'{s:02d}' for s in SEEDS)}; do
  J=$(sb --dependency=afterok:$BUILD 02_embed_s$s.sh)
  echo "embed   s$s   $J"
done
"""


def _train_dep() -> str:
    return ":$BUILD" if SUITE.emit_build_job else ""


def _submit_extract() -> str:
    out = []
    for k, q in enumerate(SUITE.query_sets):
        tok = SUITE.job_token(q)
        num = 4 + 2 * k
        out.append(
            f'J=$(sb --dependency=afterok:$TRAIN 0{num}_functional_{tok}.sh); '
            f'echo "func    {tok}     $J"\n'
            f"for i in {' '.join(str(i) for i in range(BEHAVIORAL_SHARDS))}; do\n"
            f"  J=$(sb --dependency=afterok:$TRAIN 0{num + 1}_behavioral_{tok}_shard$i.sh)\n"
            f'  echo "behav   {tok}$i    $J"\n'
            f"done\n"
        )
    return "".join(out)


if __name__ == "__main__":
    main()
