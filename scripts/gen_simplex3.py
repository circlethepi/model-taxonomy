#!/usr/bin/env python
"""Generate every config and sbatch script for a grouped-mixture simplex experiment.

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

One generator, two axes: what the experiment *is* lives here, and the two things
that vary between runs of it live in dataclasses.  A
:class:`src.experiments.suite.Suite` says *how* a run is configured -- base
model, dtype, LoRA targets, walls, sharding.  A
:class:`src.experiments.data_simplex_spec.DataSimplexSpec` says *what corpus the
simplex is built over* -- the dataset, the vertex axis, the group partition, the
projections, the grid, the draw sizes, the embedder.  A second base model is a
Suite entry plus a model profile; a second dataset is a spec entry.  Neither is a
forked copy of this file -- forking would duplicate the simplex enumeration and
the level defaults, and the copies would drift the first time a shard count
changed.

The yahoo tree of every existing suite must regenerate byte-for-byte::

    for s in llama qwen llama3i nemo olmo2; do python scripts/gen_simplex3.py --suite $s; done
    git diff --exit-code experiments jobs

Note that the number of groups is *not* fixed at three despite this file's name:
yahoo is a 2-simplex over three groups, dolly and oasst1 are 3-simplexes over
four.  The name is kept because it is in five directory names, five job-name
prefixes and a great deal of sacct history.

Usage:
    python scripts/gen_simplex3.py                    # yahoo x llama -> experiments/simplex3/
    python scripts/gen_simplex3.py --suite qwen       # yahoo x qwen  -> experiments/simplex3_qwen/
    python scripts/gen_simplex3.py --dataset dolly --suite qwen       # -> simplex3_dolly_qwen/
    python scripts/gen_simplex3.py --dataset dolly --data-tree        # -> simplex3_dolly_data/
    python scripts/gen_simplex3.py --list             # enumerate the proportions and exit
"""

from __future__ import annotations

import argparse
import sys
from math import gcd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.data_simplex_spec import SPECS, DataSimplexSpec  # noqa: E402
from src.experiments.suite import Suite  # noqa: E402

# ── The experiment's fixed parameters ──────────────────────────────────────────

# Everything that describes the *corpus* -- the group partition, the vertex axis,
# the text projections, the draw sizes, the embedder, the prose caveats -- lives
# in src/experiments/data_simplex_spec.py, one DataSimplexSpec per dataset. What
# is left here is what is true of the experiment whichever corpus it is run over.

#: Absolute, so a job started from a worktree still writes into the one shared
#: cache rather than growing a second copy that disappears with the worktree.
#:
#: The `jhu/` component appeared on 2026-08-26: /weka/scratch/cpriebe1 no longer
#: resolves, so the pre-existing job scripts died before running a line (SLURM
#: cannot open --output, exit 53, and no log to say why). The cache under the new
#: root is the same tree the earlier jobs wrote -- this is a path change, not a
#: move, so nothing needs re-deriving.
REPO = "/weka/scratch/jhu/cpriebe1/MO/model-taxonomy"
#: Not a Suite field: the cache is shared *between* suites on purpose.  Draws and
#: dataset embeddings are model-free, so a second base model reuses them rather
#: than re-deriving 640 centroids.
CACHE_DIR = f"{REPO}/results/shared_cache"

#: Which configuration is being generated.  Rebound by ``main()`` from --suite;
#: the default is the Llama run, whose emitted files must not change.
SUITE = Suite()

#: Which corpus.  Rebound by ``main()`` from --dataset; the default is yahoo, for
#: the same reason -- five trees on disk were emitted under it.
SPEC = SPECS["yahoo"]

#: True while the standalone dataset build tree is being emitted.  Its configs
#: name no base model, because nothing in them loads one; see ``base_models_yaml``.
DATA_TREE = False


def base_models_yaml() -> str:
    """The ``base_models:`` block.

    Empty for a dataset build tree.  Every job in that tree runs the dataset
    level only -- ``extraction.models`` is ``[]`` and fine-tuning is off -- so no
    model is ever loaded, and naming one would imply the centroids depend on it
    when ``DatasetEmbeddingCache`` keys on recipe and embedder alone.
    """
    if DATA_TREE:
        return ("base_models: []   # none: this tree builds the dataset level, "
                "which is model-free\n")
    return f"base_models:\n  - {SUITE.base_model}\n"


def slug() -> str:
    """The directory leaf both trees are written under.

    Dataset first, then suite, so ``simplex3`` stays ``simplex3`` and a second
    corpus on the same model reads as ``simplex3_dolly_qwen`` rather than as a
    variant of the qwen suite.
    """
    if DATA_TREE:
        return f"simplex3{SPEC.suffix}_data"
    return f"simplex3{SPEC.suffix}{SUITE.suffix}"


def output_dir() -> str:
    """Where results and logs land.

    Carries the dataset token as well as the suite's.  It must: adapters live in
    the shared cache under the base model, but ``experiment.yaml`` here is
    first-writer-wins and ``logs/`` filenames carry no dataset token, so two
    corpora sharing one results directory would interleave silently.
    """
    return f"{REPO}/results/{slug()}"


def samples_seen() -> int:
    """The ``_b5008`` token every adapter is named for.

    Derived from the suite's effective batch rather than written down: the budget
    quantizes up to a whole number of steps, so the number in the name is not the
    number in the config, and a suite at another effective batch would otherwise
    be named for a budget it never saw.
    """
    return SPEC.samples_seen(SUITE.effective_batch)


SEEDS = list(range(10))
LORA_RANK, LORA_INIT_SEED = 16, 0

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

#: GPU partition order is a Suite field -- it is a property of what the run needs
#: from the hardware, not of the experiment.  CPU stays here: nothing about it
#: varies by model.
#:
#: "cpu" was the name until the 2026-08-26 cluster reorganization, the same one
#: that removed "nvl" from the GPU list.  There is now exactly one non-GPU
#: partition and it is called "med" (`sinfo -a` lists no "cpu" at all), so every
#: prefetch job emitted before that date fails at SUBMIT time with
#: "invalid partition specified: cpu" -- and because submit_all.sh runs under
#: `set -e` with the whole chain hanging afterok off the prefetch, that aborts
#: the submit before a single job is queued.
CPU_PARTITION = "med"


# ── The simplex ────────────────────────────────────────────────────────────────

def proportions() -> list[tuple[str, tuple[int, ...]]]:
    """``(name, pct_per_group)`` for every point of the simplex.

    The grid points are the compositions of ``spec.grid`` into ``K`` non-negative
    parts, ``C(grid + K - 1, K - 1)`` of them.  Yahoo's three groups on the 25%
    grid give ``C(6,2) = 15``: three pure vertices, six edge points, three edge
    midpoints and three interior points.  Four groups on the same grid give
    ``C(7,3) = 35``.

    The even mixture is appended only when it is not already a grid point -- that
    is, only when ``K`` does not divide ``grid``.  Yahoo's ``(1/3, 1/3, 1/3)``
    is not on a 25% grid, so it is the appended sixteenth and is also the query
    draw; its name rounds to ``033g1_033g2_033g3`` and sums to 99, not 100.  Only
    the label rounds, and the recipe underneath carries exact 1:1:1 weights.  At
    K=4 the even point is ``(25,25,25,25)``, already enumerated and exact, so
    nothing is appended and nothing rounds.

    Enumeration order is lexicographic on the first ``K-1`` parts with the last
    determined, which is the order the original hand-written triple loop
    produced.  It is load-bearing only for byte-parity with the trees already on
    disk, but that is reason enough not to change it.
    """
    step = 100 // SPEC.grid
    out = [(name_for(tuple(part * step for part in parts)),
            tuple(part * step for part in parts))
           for parts in _compositions(SPEC.grid, SPEC.n_groups)]
    if not SPEC.even_is_grid_point:
        out.append((name_for(SPEC.even_pct), SPEC.even_pct))
    return out


def proportions_for(spec: DataSimplexSpec) -> list[tuple[str, tuple[int, ...]]]:
    """``proportions()`` for a spec that is not the bound one.

    Needed once, in ``main()``: the shard counts follow the adapter count, and
    that has to be known before the spec is bound for emission.
    """
    global SPEC
    was, SPEC = SPEC, spec
    try:
        return proportions()
    finally:
        SPEC = was


def _compositions(total: int, parts: int):
    """Every way to write *total* as an ordered sum of *parts* non-negative ints."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def name_for(pct: tuple[int, ...]) -> str:
    return f"{SPEC.name_prefix}_" + "_".join(
        f"{p:03d}{g}" for p, g in zip(pct, SPEC.groups))


def weights_for(pct: tuple[int, ...]) -> list[tuple[str, int]]:
    """``(group, integer weight)`` for the non-zero groups, in lowest terms.

    Entry weights are relative -- ``normalized_weights`` divides by the sum -- so
    2:1:1 and 0.50:0.25:0.25 are the same recipe.  Integers are used because they
    serialize exactly and a float ratio does not.

    Zero-weight groups are dropped rather than carried at ``weight: 0.0``.  That
    matches the existing ``yahoo_100t0_000t1`` convention, keeps each proportion's
    recipe_hash distinct via a differing entry set, and avoids filtering 1.4M rows
    to draw none.
    """
    nonzero = [(g, p) for g, p in zip(SPEC.groups, pct) if p > 0]
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
    """One ``entries:`` block: this group's rows, at this relative weight.

    ``text_field:`` names the spec's answer column, not a literal.  It is only
    the fallback projection -- ``text_fields`` wins -- but a fallback naming a
    column the dataset does not have is a trap waiting for the first caller that
    takes the fallback path.

    ``class_filter`` is emitted bare, so string members are unquoted YAML.  Safe
    for all three datasets as written, and a hazard if a fourth is added: ``no``
    (Norwegian), ``on``, ``off``, ``y`` and ``n`` are YAML 1.1 booleans and would
    silently become ``False``/``True`` and match no row.
    """
    return (
        f"{indent}- dataset_id: {SPEC.dataset_id}\n"
        f"{indent}  split: train\n"
        f"{indent}  weight: {weight}.0\n"
        f"{indent}  text_field: {SPEC.answer_field}          # fallback projection; text_fields wins\n"
        f"{indent}  text_fields: [{', '.join(text_fields)}]\n"
        f'{indent}  text_separator: "\\n"\n'
        f"{indent}  class_field: {SPEC.class_field}\n"
        f"{indent}  class_filter: {SPEC.groups[group]}\n"
        f"{indent}  class_sampling: pooled           # draw from the union, not per-topic quotas\n"
    )


def dataset_block(name: str, pct: tuple[int, ...], *,
                  sweep: str | None = None, seeds: list[int] | None = None,
                  n_samples: int | None = None, seed: int | None = None,
                  text_fields: list[str] | None = None) -> str:
    fields = text_fields or list(SPEC.text_fields)
    head = f"  - name: {name}\n    recipe_type: class_aware\n"
    if sweep is not None:
        head += f"    n_samples_sweep: {sweep}\n"
    if seeds is not None:
        head += f"    seeds: {seeds}\n"
    if n_samples is not None:
        head += f"    n_samples: {n_samples}\n"
    if seed is not None:
        head += f"    seed: {seed}\n"
    mix = " / ".join(f"{p}% {g}" for p, g in zip(pct, SPEC.groups))
    head += f"    # {mix}\n    entries:\n"
    return head + "".join(
        entry_yaml(g, w, fields) for g, w in weights_for(pct)
    )


def even() -> tuple[int, ...]:
    """The equal mixture, which is the query draw for every level."""
    return SPEC.even_pct


def query_full_context_name() -> str:
    return name_for(even())


def query_question_only_name() -> str:
    return query_full_context_name() + "_qtc"


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
        return dataset_block(query_full_context_name(), even(),
                             n_samples=SPEC.query_n, seed=SPEC.query_seed)
    return dataset_block(query_question_only_name(), even(),
                         n_samples=SPEC.query_n, seed=SPEC.query_seed,
                         text_fields=list(SPEC.query_fields))


def adapter_name(name: str) -> str:
    """The adapter's directory leaf.

    The ``_f{format_id}`` suffix appears only for a non-raw prompt format, so
    every existing adapter name is unchanged, and two suites that wrap the same
    base model differently cannot land on the same path.  Downstream joins strip
    it the way they already strip the ``_b{samples}`` budget token.
    """
    stem = (f"{name}_n{SPEC.train_n}_s{SPEC.train_seed:02d}"
            f"_r{LORA_RANK}_i{LORA_INIT_SEED:02d}_b{samples_seen()}")
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
        "user_fields": list(SPEC.query_fields),
        "answer_fields": [SPEC.answer_field],
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


def embedder_block() -> str:
    """The ``embedder:`` mapping, whose model is a property of the dataset.

    Multilingual for oasst1 out of necessity -- ``nomic-embed-text-v1.5`` is
    English-only, and embedding ``ru`` and ``zh`` with it would make the language
    vertices an artefact of the embedder rather than of the data -- and for dolly
    so the two new corpora are mutually comparable from the start.  Yahoo stays
    on v1.5 because its configs have already run; it joins the others through an
    additive re-embed that writes beside the existing artefacts.

    ``_PREFIX_REQUIRED_MODELS`` matches by prefix on ``nomic-ai/nomic-embed-text``,
    so v2-moe is already covered by the prefix machinery and uses the same
    ``search_document: `` spelling.  Both models emit 768 dimensions.
    """
    return """      embedder:
        model_name: """ + SPEC.embedder_model + """
        normalize_embeddings: true
        trust_remote_code: true
        # The literal prefix nomic wants is "search_document: ". Note the existing
        # surrogates spell this `document`, which resolves to the same prefix and
        # the same vectors -- but prompt_name is inside config_dict(), so the two
        # spellings are different embedder_hashes. Reuse this spelling or re-embed.
        prompt_name: search_document
"""


def write_sweep(seed: int) -> str:
    """Build + embed config for one seed: every proportion at every draw size."""
    props = proportions()
    sizes = SPEC.sweep_size_list
    body = HEADER + (
        f"# Seed {seed} of the simplex sweep: {len(props)} proportions x "
        f"{{{','.join(str(n) for n in sizes)}}} = {len(props) * len(sizes)} draws.\n"
        f"# Sharded by seed rather than by proportion so each shard still touches all\n"
        f"# {_num_word(SPEC.n_groups)} group filters -- source_registry memoises them, "
        f"so that is {_num_word(SPEC.n_groups)}\n"
        f"# filter passes per shard instead of one per draw.\n#\n"
    )
    if SPEC.caveats.get("draw"):
        body += "# " + SPEC.caveats["draw"].replace("\n", "\n# ") + "\n#\n"
    if seed == SPEC.query_seed:
        body += (
            f"# This shard also builds the question-only query set ({query_question_only_name()}).\n"
            f"# Composition feeds recipe_hash, so it is a distinct recipe -- but\n"
            f"# text_fields never touches the sampler's RNG and the seed is not derived\n"
            f"# from the hash, so it selects the SAME {SPEC.query_n} rows as the full-context set.\n"
            f"# The two are the same questions projected two ways, which is what makes the\n"
            f"# ablation controlled rather than two different samples.\n#\n"
        )
    body += "\n" + preamble(f"simplex3_sweep_s{seed:02d}")
    body += "datasets:\n"
    body += "\n".join(
        dataset_block(name, pct, sweep=_render_sweep_sizes(), seeds=[seed])
        for name, pct in props
    )
    if seed == SPEC.query_seed:
        body += "\n" + query_blocks("question_only")
    body += f"""
{base_models_yaml()}
fine_tuning:
  enabled: false

# One (1, 768) centroid per draw. The surrogate is authored, not derived -- the
# full (N, 768) matrix is never stored and `mean` is not invertible -- so adding a
# second representation later means re-embedding all {len(props) * len(SEEDS) * len(sizes)}.
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
{embedder_block()}"""
    return body


def _render_sweep_sizes() -> str:
    """``n_samples_sweep:``'s value, rendered the way the spec spells it.

    A string goes in literally, so ``tens 3`` stays the compact spelling the
    recipe parser expands; a list renders as itself.
    """
    if isinstance(SPEC.sweep_sizes, str):
        return SPEC.sweep_sizes
    return "[" + ", ".join(str(n) for n in SPEC.sweep_sizes) + "]"


def _num_word(n: int) -> str:
    """Small numbers spelled out, for prose emitted into a comment."""
    return {2: "two", 3: "three", 4: "four", 5: "five"}.get(n, str(n))


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
    The embedder block is the shared ``embedder_block()`` one, unchanged, which is
    what makes the new surrogate land beside the existing ``mean`` under the same
    ``{embedder_hash}/`` rather than in a directory of its own.
    """
    body = HEADER + (
        f"# Re-embed the {len(proportions())} trained draws at n={SPEC.train_n}, "
        f"seed={SPEC.train_seed},\n"
        f"# keeping the full (n, 768) matrix instead of the (1, 768) centroid.\n"
        f"# Adds a second surrogate beside the existing `mean`; nothing is replaced.\n#\n"
    )
    body += "\n" + preamble(f"{slug()}_embed_matrix")
    body += "datasets:\n"
    body += "\n".join(
        dataset_block(name, pct, n_samples=SPEC.train_n, seed=SPEC.train_seed)
        for name, pct in proportions()
    )
    body += f"""
{base_models_yaml()}
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
{embedder_block()}"""
    return body


def write_train(shard: int, names: list[str]) -> str:
    body = HEADER + (
        f"# Training shard {shard} of {SUITE.train_shards}: {len(names)} adapters.\n"
        f"# Only n={SPEC.train_n}, seed={SPEC.train_seed} is trained on. total_train_samples\n"
        f"# {SPEC.total_train_samples} quantizes UP to a step boundary at effective batch "
        f"{SUITE.effective_batch}:\n"
        f"# ceil({SPEC.total_train_samples}/{SUITE.effective_batch}) = "
        f"{SPEC.steps(SUITE.effective_batch)} steps, so {samples_seen()} samples seen and the\n"
        f"# adapter is named _b{samples_seen()}. Every LoRA parameter matches the existing 3B\n"
        f"# adapters; only the base model and the data differ.\n"
        f"#\n"
        f"# {SPEC.caveats.get('train', '').replace(chr(10), chr(10) + '# ')}\n\n"
    ) + preamble(f"simplex3_train_shard{shard}")
    body += "datasets:\n"
    body += "\n".join(
        dataset_block(name, pct, sweep=f"[{SPEC.train_n}]", seeds=[SPEC.train_seed])
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
    body += f"""  n_samples: {SPEC.train_n}
  seed: {SPEC.train_seed}
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
  total_train_samples: {SPEC.total_train_samples}

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
# Deliberately NOT extended when the logprob level was added, even though the
# same reasoning covers it: this string is rendered into every config the Llama
# suite already ran, and `Suite()` must regenerate those byte-for-byte.  The
# logprob level's own default -- false, for exactly the reason above read the
# other way -- is documented at its one decision point, extract_reprs.py.


def temp_token(t: float) -> str:
    """``t05`` for 0.5 — the temperature x10, zero-padded.

    Two digits so the files sort in temperature order, and prefixed so the token
    is unambiguous inside a job name that already carries a shard number.
    """
    return f"t{int(round(float(t) * 10)):02d}"


def _qdesc(which: str) -> str:
    """How this query set is described in the emitted header.

    ``which`` is a query set name, or ``"chat"``: under a chat template with
    completion-only loss the training prompt *is* the question, so the
    question-only set becomes the in-distribution probe and the full-context one
    the contaminated one -- the reverse of the raw suite.  One description cannot
    carry both readings, so the spec supplies three.

    The optional ``suffix`` entry is appended to every question-only description.
    It exists for yahoo, where ``question_content`` is empty in ~46% of rows, so
    about half of that set's prompts are a bare title -- a fact that belongs in
    any caption comparing the two sets.
    """
    desc = SPEC.query_desc[which]
    if which in ("chat", "question_only") and SPEC.caveats.get("query"):
        desc += "; note " + SPEC.caveats["query"] + SPEC.query_desc.get("suffix", "")
    return desc


def write_extract(
    level: str,
    query: str,
    shard: int | None,
    names: list[str],
    temperature: float | None = None,
) -> str:
    qname = query_full_context_name() if query == "full_context" else query_question_only_name()
    if SUITE.prompt_format:
        # Under a chat template with completion-only loss the training prompt IS
        # the question, so the roles are the reverse of the raw suite's.  Saying
        # so here rather than carrying the raw suite's phrasing forward, which
        # would be actively wrong.
        qdesc = _qdesc("chat")
    else:
        qdesc = _qdesc(query)
    token = SUITE.job_token(query)
    n_shards = SUITE.sweep_shards if level == "sweep" else SUITE.behavioral_shards
    label = (
        f"simplex3_{level}_{token}"
        + ("" if temperature is None else f"_{temp_token(temperature)}")
        + ("" if shard is None else f"_shard{shard}")
    )
    body = HEADER + (
        f"# {level.capitalize()} extraction over the {query} query set ({qdesc}).\n"
        f"# {len(names)} adapter(s)"
        + ("" if shard is None else f", shard {shard} of {n_shards}")
        + ".\n#\n"
    )
    if level == "functional":
        body += (
            "# One job for all 16: HFInferenceTaxonomy loads the base model once and\n"
            "# swaps adapters onto it, so 16 models amortize a single 8B load, and\n"
            "# input-mode extraction is one forward pass per query with no decoding.\n\n"
        )
    elif level == "functional_gen":
        body += (
            f"# GENERATION-mode activations: the hidden states the model occupies while\n"
            f"# PRODUCING text, as against the ones it occupies while reading it. Same\n"
            f"# level, same 16 adapters, same draw as 04_functional -- the one axis that\n"
            f"# level has never been read along.\n"
            f"#\n"
            f"# Greedy only, and that is a hard constraint rather than a choice:\n"
            f"# FunctionalTaxonomy hardcodes do_sample=False (functional.py:353) and\n"
            f"# ActivationCache's filename carries no sampling hash, so two decoding\n"
            f"# points would overwrite each other silently. Per-temperature generation\n"
            f"# activations need a sampling hash in DrawKeyedCache.mode_token first --\n"
            f"# a change to a key with entries already on disk.\n"
            f"#\n"
            f"# Additive: these land beside the existing input_* files in the same\n"
            f"# directory (generation{MAX_NEW_TOKENS}_* vs input_*) and save_activations skips\n"
            f"# any path that exists, so this job cannot disturb what is already there.\n"
            f"# `activation_mode: both` is then a READ-time union with no further work.\n\n"
        )
    elif level == "logprob":
        body += (
            f"# INPUT log-probabilities: the teacher-forced per-token log-prob and\n"
            f"# entropy of each query prompt. No generation at all -- this is the same\n"
            f"# masked forward pass 04_functional runs, read for what the model assigned\n"
            f"# rather than for where it sat.\n"
            f"#\n"
            f"# One job for all 16, like the functional one and for the same reason: the\n"
            f"# base model is loaded once and adapters are swapped onto it, and there is\n"
            f"# no decoding to pay for. Minutes of GPU each.\n"
            f"#\n"
            f"# The cost here is MEMORY. Per-token log-probs need logits at every\n"
            f"# position and this vocabulary is 248,320 wide, so the log_softmax is\n"
            f"# chunked over the sequence axis and the realized token gathered per chunk\n"
            f"# (src/taxonomy/logprob.py). batch_size and seq_chunk below are that\n"
            f"# budget; neither changes the stored numbers.\n\n"
        )
    elif level == "greedy_logprob":
        body += (
            f"# The T=0 point of the log-prob surface: the greedy run again, with\n"
            f"# collect_logprobs on.\n"
            f"#\n"
            f"# It re-generates rather than reusing the cached greedy entry, and it must:\n"
            f"# the generations live in 05_generated and the log-probs in 05A_logprobs, so\n"
            f"# a hit on the first alone would return cached text and write no log-probs\n"
            f"# at all. BehavioralTaxonomy's hit test requires both when collecting.\n"
            f"# Re-generation is exact -- greedy is deterministic -- so the existing\n"
            f"# entry is confirmed rather than replaced.\n\n"
        )
    elif level == "sweep":
        body += (
            f"# One point of the TEMPERATURE SWEEP: T={temperature}, R={SUITE.sweep_replicates}.\n"
            f"#\n"
            f"# The sweep resolves the log-prob surface along the decoding axis instead\n"
            f"# of at the single T=1.0 the R={REPLICATES} runs measured. Each point is its own\n"
            f"# cache entry -- temperature is inside the sampling hash and therefore\n"
            f"# inside the filename -- so ten temperatures are ten entries, not one\n"
            f"# silently reused.\n"
            f"#\n"
            f"# R={SUITE.sweep_replicates}, not {REPLICATES}, and uniformly so across all ten points. The cached\n"
            f"# R={REPLICATES} entry at T=1.0 is a DIFFERENT entry (replicates are in the\n"
            f"# filename), so this does not collide with it; re-running T=1.0 here is\n"
            f"# what keeps one point of a variance-vs-temperature curve from having half\n"
            f"# the sampling noise of the other nine.\n"
            f"#\n"
            f"# {SUITE.sweep_shards} shards of {len(names)} adapters: halving R halves the decode, so this\n"
            f"# lands at the same wall the {SUITE.behavioral_shards}-shard R={REPLICATES} runs were sized against.\n\n"
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
            f"# One job, unsharded: {SPEC.query_n} queries at batch {GREEDY_BATCH_SIZE} is ~7 generate()\n"
            f"# calls per adapter against ~50 for the sampled runs, so all 16 adapters\n"
            f"# finish in well under an hour.\n\n"
        )
    else:
        body += (
            f"# Sharded because this is the expensive half: {len(proportions())} adapters x {SPEC.query_n} queries\n"
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
  n_queries: {SPEC.query_n}
  device: cuda
  torch_dtype: {SUITE.torch_dtype}
"""
    if level == "functional_gen":
        body += (
            f"  # THIS is the batch size the functional level actually reads --\n"
            f"  # make_functional_taxonomy takes extraction.batch_size, not the one\n"
            f"  # nested under the level (scripts/_utils.py). Held at 8 because\n"
            f"  # generate() retains the hidden states of EVERY decode step: step 0\n"
            f"  # carries the full prompt across 33 layers, steps 1-{MAX_NEW_TOKENS - 1} one position\n"
            f"  # each, ~107 MB per sequence. Wider buys nothing -- decode cost is the\n"
            f"  # number of generate() calls, not their width.\n"
            f"  batch_size: 8\n"
        )
    elif level in ("behavioral", "sweep"):
        body += (
            "  # Held at 2, and this is not cosmetic: one RNG generator serves a whole\n"
            "  # generate() call, so under sampling a different batch_size gives\n"
            "  # different text at the same seed. batch_size is deliberately OUT of the\n"
            "  # cache key, so a shard run at another value would silently collide with\n"
            "  # a correct entry rather than fail. Do not retune per GPU.\n"
            "  batch_size: 2\n"
        )
    elif level in ("greedy", "greedy_logprob"):
        body += (
            f"  # Raised to {GREEDY_BATCH_SIZE}, which is safe here and NOT safe for the sampled\n"
            f"  # runs. Greedy seeds no RNG at all, so batch size only flips argmax on\n"
            f"  # fp16 near-ties -- measured at 6/8 sequences byte-identical between\n"
            f"  # batch 1 and batch 8. Decode cost is driven by the number of generate()\n"
            f"  # calls, not their width, so this is the difference between ~7 calls per\n"
            f"  # adapter and ~50. Lower it if an 8B at {GREEDY_BATCH_SIZE} x (prompt + {MAX_NEW_TOKENS}) OOMs.\n"
            f"  batch_size: {GREEDY_BATCH_SIZE}\n"
        )
        if level == "greedy_logprob":
            body += (
                f"  # At R=1 that is {GREEDY_BATCH_SIZE} rows, the same row count the sweep runs at\n"
                f"  # batch 2 x R={SUITE.sweep_replicates}, so the two logit stacks generate() accumulates\n"
                f"  # ({GREEDY_BATCH_SIZE} x {MAX_NEW_TOKENS} x 248,320 x 4 B each) cost the same ~4 GB here.\n"
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
    elif level == "functional_gen":
        body += f"""    functional:
      enabled: true
      activation_mode: generation   # decode-phase states, not input-phase
      max_new_tokens: {MAX_NEW_TOKENS}          # in the filename: generation{MAX_NEW_TOKENS}_mean_layerNNN
      layer_indices: null           # every hidden state
      pooling: mean
      normalize_activations: true
    behavioral:
      enabled: false
    dataset_embedding:
      enabled: false
"""
    elif level == "logprob":
        body += """    logprob:
      enabled: true
      mode: input                 # teacher-forced scoring; no decoding
      batch_size: 8               # bounded by the 248,320-way log_softmax
      seq_chunk: 64               # positions per softmax chunk; memory only
    functional:
      enabled: false
    behavioral:
      enabled: false
    dataset_embedding:
      enabled: false
"""
    elif level in ("greedy", "greedy_logprob"):
        body += f"""    functional:
      enabled: false
    dataset_embedding:
      enabled: false
    behavioral:
      enabled: true
      max_new_tokens: {MAX_NEW_TOKENS}
      replicates: 1               # enforced: >1 with do_sample:false raises
      do_sample: false            # THE difference from the sampled runs
"""
        if level == "greedy_logprob":
            body += (
                "      # Adds the 05A_logprobs half of this entry. It also changes the\n"
                "      # cache-hit test -- a hit now requires the log-prob file too -- so\n"
                "      # this job re-decodes the cached greedy text rather than skipping.\n"
                "      # Greedy is deterministic, so that re-decode confirms the existing\n"
                "      # entry rather than replacing it.\n"
                "      collect_logprobs: true\n"
            )
        body += embedder_block()
    elif level == "sweep":
        body += f"""    functional:
      enabled: false
    dataset_embedding:
      enabled: false
    behavioral:
      enabled: true
      max_new_tokens: {MAX_NEW_TOKENS}
      replicates: {SUITE.sweep_replicates}
      do_sample: true
      temperature: {temperature}             # THE swept axis; inside the sampling hash
      top_p: 1.0
      top_k: null
      generation_seed: 0
      # Both distributions are stored: the warped one the sampler drew from, and
      # the model's own unprocessed one. Only the second is comparable across
      # temperatures, and the first is not recoverable from it -- see
      # src/taxonomy/behavioral.py. At T=1.0 with top_p 1.0 and no top_k every
      # processor is the identity and the two must coincide, which is a free
      # consistency check on this sweep.
      collect_logprobs: true
{embedder_block()}"""
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
{embedder_block()}"""
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
export HF_HOME=/weka/scratch/jhu/cpriebe1/MO/huggingface_cache

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
        # The shared default (2:30) was sized against a sampled *shard* --
        # 4 adapters, ~20 min.  This job decodes all 16 in one process at
        # ~12 min each, so it needs the full 16-model wall, not a shard's.
        func_gen_time="4:00:00",
        # Host RAM. The 248,320-vocab tied lm_head is 15% of the model and
        # materializes a (4, 512, 248320) logits tensor; nothing at Llama's 128k
        # vocab needed this much headroom.
        train_mem_gb=96,
        extract_mem_gb=96,
        # bf16 throughput and HBM headroom first.
        gpu_partitions="h100,h200,a100,l40s",
        # The log-prob level and everything built on it. Only this suite asks for
        # them: turning them on for the Llama suite would regenerate 43 files
        # into a tree whose jobs have already run.
        emit_logprob_jobs=True,
        emit_gen_activation_job=True,
        temperature_sweep=tuple(round(0.1 * i, 1) for i in range(1, 11)),
    ).for_model("Qwen/Qwen3.5-4B"),
    "llama3i": Suite(
        tag="llama3i",
        # As qwen: under a chat template with completion-only loss the question
        # *is* the training prompt, so question_only is the in-distribution probe
        # and full_context would measure something the adapters never saw.
        query_sets=("question_only",),
        job_tokens={"question_only": "qonly"},
        # Same reasoning as qwen: draws and centroids are model-free and already
        # on disk, and every job writes the recipes it needs itself.
        emit_embed_jobs=False,
        emit_build_job=False,
        job_prefix="s3li",
        # Wall times and memory are the Suite() defaults, which were sized against
        # exactly this shape -- 8B, 128k vocab, uniform attention -- and proven by
        # the simplex3 run.  Qwen's inflated values do not transfer: they paid for
        # a torch linear-attention fallback and a 248k-vocab lm_head, and neither
        # applies here.  No reasoning block means generations stop well inside the
        # 128-token budget, so behavioral should run shorter than the raw 8B suite
        # rather than longer.
        #
        # func_gen is the one wall with no precedent on this shape: it decodes all
        # 16 adapters in one process like the greedy job (proven at 1:30 here) and
        # additionally retains a hidden state per step.  3:00 for headroom, still
        # well under Qwen's 4:00, which was sized for a model that always ran the
        # full 128 steps.
        func_gen_time="3:00:00",
        # Full parity with the qwen suite: the log-prob level, generation-mode
        # activations, and the 10-point temperature sweep.  This is the point of
        # the run -- against simplex3 it isolates base vs instruct, against
        # simplex3_qwen it isolates model family, and neither comparison works if
        # the levels differ too.
        emit_logprob_jobs=True,
        emit_gen_activation_job=True,
        temperature_sweep=tuple(round(0.1 * i, 1) for i in range(1, 11)),
    ).for_model("meta-llama/Llama-3.1-8B-Instruct"),
    "nemo": Suite(
        tag="nemo",
        # As qwen and llama3i: under a chat template with completion-only loss
        # the question *is* the training prompt.
        query_sets=("question_only",),
        job_tokens={"question_only": "qonly"},
        # Same reasoning as the other chat suites: draws and centroids are
        # model-free and already on disk, and every job writes its own recipes.
        emit_embed_jobs=False,
        emit_build_job=False,
        job_prefix="s3nm",
        # Full parity with qwen and llama3i -- same query set, same levels, same
        # 10-point sweep.  A suite that emitted fewer levels would not be
        # comparable to the two it exists to sit beside.
        emit_logprob_jobs=True,
        emit_gen_activation_job=True,
        temperature_sweep=tuple(round(0.1 * i, 1) for i in range(1, 11)),
        # 12.2B bf16 = ~24.5 GB of weights, half again the llama3i suite's 8B.
        # These are STARTING values, not measured ones; the smoke run is what
        # confirms them, and train_time is the one to re-check against the first
        # 03_train_shard0 log before the full tree goes out.
        train_time="3:30:00",
        behav_time="2:30:00",
        func_time="2:00:00",
        greedy_time="2:30:00",
        logprob_time="1:30:00",
        func_gen_time="4:00:00",
        # Same host RAM the qwen suite asks for; no reason found to go higher,
        # and this model's 131k vocab is nowhere near Qwen's 248k.
        train_mem_gb=96,
        extract_mem_gb=96,
        # This checkpoint publishes its weights twice -- five sharded shards and
        # a consolidated.safetensors -- and the prefetch job's '*.safetensors'
        # would fetch both.
        prefetch_ignore=("consolidated.safetensors",),
        # gpu_partitions left at the default, but l40s is NOT safe for TRAINING
        # this model, and the note above that said it was has been measured
        # wrong.  2026-09-05, the dolly and oasst1 trees: 13 of 13 train shards
        # that landed on an l40s node (gl*) died in the backward pass with
        #     RuntimeError: Function MmBackward0 returned an invalid gradient at
        #     index 1 - expected device meta but got cuda:0
        # while 5 of 5 that landed on gh* (h100/h200) completed.  The mechanism
        # is device_map="auto": on a 48 GB card accelerate places part of the
        # model off-device, and the offloaded submodule hands back a meta
        # gradient.  It is not a clean OOM, so it does not look like one in the
        # log.  The yahoo tree never caught this because all four of its train
        # shards happened to schedule onto gh* nodes.
        #
        # Extraction is unaffected -- it is forward-only, and a forward through
        # an offloaded module is correct (just slower) -- so this field stays as
        # it is and the l40s capacity keeps serving the 18 behavioural shards.
        # Training was resubmitted with `sbatch --partition=h200,h100`, which
        # overrides the #SBATCH line without regenerating the tree.  If training
        # is ever driven from submit_all.sh again, pass that override there.
    ).for_model("mistralai/Mistral-Nemo-Instruct-2407"),
    "olmo2": Suite(
        tag="olmo2",
        query_sets=("question_only",),
        job_tokens={"question_only": "qonly"},
        emit_embed_jobs=False,
        emit_build_job=False,
        job_prefix="s3o2",
        emit_logprob_jobs=True,
        emit_gen_activation_job=True,
        temperature_sweep=tuple(round(0.1 * i, 1) for i in range(1, 11)),
        # Walls and memory left at the Suite() defaults, as the llama3i suite
        # deliberately does.  They are generous for a 1B, but over-requesting
        # costs only queue priority, while eight invented numbers cost a reader's
        # trust in all of them.  Trim after the first run measures actual walls.
    ).for_model("allenai/OLMo-2-0425-1B-Instruct"),
}


#: Job-name prefixes for a model suite over a non-yahoo corpus, extending the
#: five already in use -- ``s3`` (llama), ``s3q`` (qwen), ``s3li`` (llama3i),
#: ``s3nm`` (nemo), ``s3o2`` (olmo2).  These appear in sacct history and in log
#: filenames and are effectively permanent once submitted, so they are written
#: down rather than derived from the two names.
_SUITE_PREFIXES = {
    ("dolly", "qwen"): "s3dq",
    ("dolly", "llama3i"): "s3dli",
    ("dolly", "nemo"): "s3dnm",
    ("dolly", "olmo2"): "s3do2",
    ("oasst1", "qwen"): "s3oq",
    ("oasst1", "llama3i"): "s3oli",
    ("oasst1", "nemo"): "s3onm",
    ("oasst1", "olmo2"): "s3oo2",
}

#: Adapters per shard, held at the value every existing wall was tuned against.
#: Sharding is strided (``names[i::shards]``), so only the shard *count* is
#: chosen and an uneven split needs no special-casing: 35 adapters over 9 train
#: shards gives 4,4,4,4,4,4,4,4,3.
#:
#: Holding the adapters and raising the count, rather than holding the count and
#: doubling every wall, costs ~120 extra job files across the eight new trees and
#: buys two things: no wall needs re-tuning on eight suites at once, and a failed
#: shard's retry stays the size it is today.  These two numbers reproduce the
#: ``Suite`` defaults at yahoo's 16 adapters, which is why yahoo needs no branch.
TRAIN_ADAPTERS_PER_SHARD = 4
BEHAVIORAL_ADAPTERS_PER_SHARD = 2


def _suite_for_dataset(suite: Suite, dataset: str, suite_name: str,
                       n_adapters: int) -> Suite:
    """*suite* adjusted for the corpus it is being run over.

    Three things depend on the pair rather than on either half: the job-name
    prefix, the shard counts (which follow the adapter count), and whether this
    tree emits the dataset-level jobs -- a corpus with its own build tree owns
    those, so no model suite over it emits any.

    Yahoo is returned untouched: its five trees have already run.
    """
    if dataset == "yahoo":
        return suite
    from dataclasses import replace

    try:
        prefix = _SUITE_PREFIXES[(dataset, suite_name)]
    except KeyError:
        raise SystemExit(
            f"No job prefix for --dataset {dataset} --suite {suite_name}. Add one "
            f"to _SUITE_PREFIXES, checking it collides with none of "
            f"{sorted(_SUITE_PREFIXES.values())} nor with s3, s3q, s3li, s3nm, s3o2."
        ) from None
    return replace(
        suite,
        job_prefix=prefix,
        train_shards=-(-n_adapters // TRAIN_ADAPTERS_PER_SHARD),
        behavioral_shards=-(-n_adapters // BEHAVIORAL_ADAPTERS_PER_SHARD),
        emit_build_job=False,
        emit_embed_jobs=False,
        emit_embed_matrix_job=False,
    )


def main() -> None:
    global SUITE, SPEC

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="llama", choices=sorted(SUITES),
                        help="Which configuration to emit (default: llama).")
    parser.add_argument("--dataset", default="yahoo", choices=sorted(SPECS),
                        help="Which corpus the simplex is built over (default: yahoo).")
    parser.add_argument("--data-tree", action="store_true",
                        help="Emit only the dataset build tree -- the build job and the "
                             "embedding sweeps -- detached from any model suite.")
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

    SPEC = SPECS[args.dataset]
    props = proportions()

    # Listing the simplex is a property of the corpus alone, so it runs before
    # the suite is resolved -- otherwise `--dataset dolly --list` would fail on
    # the default suite's missing job prefix rather than printing 35 lines.
    if args.list:
        for name, pct in props:
            ratio = ":".join(str(w) for _, w in weights_for(pct))
            groups = ",".join(g for g, _ in weights_for(pct))
            print(f"  {name}   {pct}   {groups} = {ratio}")
        sizes = len(SPEC.sweep_size_list)
        print(f"\n{len(props)} proportions "
              f"x {len(SEEDS)} seeds x {sizes} sizes = "
              f"{len(props) * len(SEEDS) * sizes} draws; {len(props)} adapters")
        return

    if args.data_tree:
        write_data_tree(Path(args.root))
        return

    SUITE = _suite_for_dataset(SUITES[args.suite], args.dataset, args.suite,
                               len(props))
    # The temperature sweep is a Suite field, but dropping it is a decision about
    # how much a *corpus* is worth measuring. A spec that leaves it None inherits
    # the suite's; yahoo does, so the five existing trees are unaffected.
    if SPEC.temperature_sweep is not None:
        SUITE = replace(SUITE, temperature_sweep=SPEC.temperature_sweep)
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

    root = Path(args.root)
    tree = slug()
    exp = root / "experiments" / tree
    jobs = root / "jobs" / tree
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
    #
    #    ignore_patterns is emitted ONLY when the suite names something to skip.
    #    A repo may publish its weights twice -- Mistral-Nemo ships five sharded
    #    model-0000N-of-00005.safetensors AND a consolidated.safetensors -- and
    #    '*.safetensors' fetches both, silently doubling the download. Emitting
    #    nothing by default is what keeps the existing trees byte-for-byte.
    ignore = (
        ", ignore_patterns=["
        + ",".join(repr(p) for p in SUITE.prefetch_ignore)
        + "]"
    ) if SUITE.prefetch_ignore else ""
    emit(jobs / "00_prefetch.sh", sbatch(
        f"{SUITE.job_prefix}_prefetch", CPU_PARTITION, False, 32, "1:00:00",
        f'python -c "\n'
        f'from huggingface_hub import snapshot_download\n'
        f"p = snapshot_download('{SUITE.base_model}', allow_patterns=["
        f"'*.json','*.safetensors','*.model','tokenizer*','*.jinja','merges.txt']"
        f"{ignore})\n"
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
                f"python scripts/run_experiment.py experiments/{tree}/sweep_s{seed:02d}.yaml"
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
    if SUITE.emit_embed_matrix_job:
        emit(exp / "embed_matrix.yaml", write_embed_matrix())
        emit(jobs / "02_embed_matrix.sh", sbatch(
            f"{SUITE.job_prefix}_embed_matrix", SUITE.gpu_partitions, True,
            48, "2:00:00",
            f"python scripts/run_experiment.py experiments/{tree}/embed_matrix.yaml"
            f" --steps build extract --taxonomy dataset_embedding",
            logs,
        ))

    if SUITE.emit_build_job:
        emit(jobs / "01_build.sh", sbatch(
            f"{SUITE.job_prefix}_build", CPU_PARTITION, False, 32, "1:00:00",
            "\n".join(
                f"python scripts/run_experiment.py"
                f" experiments/{tree}/sweep_s{seed:02d}.yaml --steps build"
                for seed in SEEDS
            ),
            logs,
        ))

    # 3. Training, four adapters per shard.
    names = [n for n, _ in props]
    shards = [names[i::SUITE.train_shards] for i in range(SUITE.train_shards)]
    for i, shard_names in enumerate(shards):
        emit(exp / f"train_shard{i}.yaml", write_train(i, shard_names))
        emit(jobs / f"03_train_shard{i}.sh", sbatch(
            f"{SUITE.job_prefix}_train{i}", SUITE.gpu_partitions, True,
            SUITE.train_mem_gb, SUITE.train_time,
            f"python scripts/run_experiment.py experiments/{tree}/train_shard{i}.yaml"
            f" --steps build finetune",
            logs,
        ))

    # 4-7. Extraction, one pair of stages per query set.
    bshards = [names[i::SUITE.behavioral_shards] for i in range(SUITE.behavioral_shards)]
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
            f"python scripts/run_experiment.py experiments/{tree}/functional_{tok}.yaml"
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
                f" experiments/{tree}/behavioral_{tok}_shard{i}.yaml"
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
            f"python scripts/run_experiment.py experiments/{tree}/greedy_{tok}.yaml"
            f" --steps build extract --taxonomy behavioral",
            logs,
        ))

    # 9. Input log-probabilities, one unsharded job per query set. Nothing is
    #    decoded, so this is the cheapest GPU job in the suite.
    if SUITE.emit_logprob_jobs:
        for q in SUITE.query_sets:
            tok = SUITE.job_token(q)
            emit(exp / f"logprob_input_{tok}.yaml",
                 write_extract("logprob", q, None, names))
            emit(jobs / f"09_logprob_input_{tok}.sh", sbatch(
                f"{SUITE.job_prefix}_lp_input", SUITE.gpu_partitions, True,
                SUITE.extract_mem_gb, SUITE.logprob_time,
                f"python scripts/run_experiment.py"
                f" experiments/{tree}/logprob_input_{tok}.yaml"
                f" --steps build extract --taxonomy logprob",
                logs,
            ))

    # 10. The temperature sweep: one job per (temperature, shard). This is the
    #     expensive half of the addition -- ten decoding points over 16 adapters.
    if SUITE.temperature_sweep:
        sshards = [names[i::SUITE.sweep_shards] for i in range(SUITE.sweep_shards)]
        for q in SUITE.query_sets:
            tok = SUITE.job_token(q)
            for t in SUITE.temperature_sweep:
                tt = temp_token(t)
                for i, shard_names in enumerate(sshards):
                    emit(exp / f"behavioral_{tok}_{tt}_shard{i}.yaml",
                         write_extract("sweep", q, i, shard_names, temperature=t))
                    emit(jobs / f"10_behavioral_{tok}_{tt}_shard{i}.sh", sbatch(
                        f"{SUITE.job_prefix}_behav_{tok}_{tt}_{i}",
                        SUITE.gpu_partitions, True,
                        SUITE.extract_mem_gb, SUITE.behav_time,
                        f"python scripts/run_experiment.py"
                        f" experiments/{tree}/behavioral_{tok}_{tt}_shard{i}.yaml"
                        f" --steps build extract --taxonomy behavioral",
                        logs,
                    ))

    # 11. Greedy again, with log-probs on: the T=0 point of the same surface.
    if SUITE.emit_logprob_jobs:
        for q in SUITE.query_sets:
            tok = SUITE.job_token(q)
            emit(exp / f"greedy_logprob_{tok}.yaml",
                 write_extract("greedy_logprob", q, None, names))
            emit(jobs / f"11_greedy_logprob_{tok}.sh", sbatch(
                f"{SUITE.job_prefix}_greedy_lp_{tok}", SUITE.gpu_partitions, True,
                SUITE.extract_mem_gb, SUITE.greedy_time,
                f"python scripts/run_experiment.py"
                f" experiments/{tree}/greedy_logprob_{tok}.yaml"
                f" --steps build extract --taxonomy behavioral",
                logs,
            ))

    # 12. Generation-mode activations, greedy, all 16 in one job. No new code:
    #     these are the functional level read along the one axis it never was.
    if SUITE.emit_gen_activation_job:
        for q in SUITE.query_sets:
            tok = SUITE.job_token(q)
            emit(exp / f"functional_gen_{tok}.yaml",
                 write_extract("functional_gen", q, None, names))
            emit(jobs / f"12_functional_gen_{tok}.sh", sbatch(
                f"{SUITE.job_prefix}_func_gen_{tok}", SUITE.gpu_partitions, True,
                SUITE.extract_mem_gb, SUITE.func_gen_time,
                f"python scripts/run_experiment.py"
                f" experiments/{tree}/functional_gen_{tok}.yaml"
                f" --steps build extract --taxonomy functional",
                logs,
            ))

    emit(jobs / "submit_all.sh", submit_all())
    (jobs / "submit_all.sh").chmod(0o755)

    print(f"Wrote {len(written)} files:")
    print(f"  {len(list(exp.glob('*.yaml')))} configs in {exp}")
    print(f"  {len(list(jobs.glob('*.sh')))} scripts in {jobs}")
    n_sizes = len(SPEC.sweep_size_list)
    print(f"\n{len(props)} proportions, {len(props) * len(SEEDS) * n_sizes} draws, "
          f"{len(props)} adapters.")
    print(f"Submit with: bash {jobs}/submit_all.sh")


def write_data_tree(root: Path) -> None:
    """Emit the standalone per-dataset build tree.

    The draws and the dataset-level centroids are model-free -- ``DatasetEmbeddingCache``
    keys on recipe and embedder, never on the model -- so they are built exactly
    once per corpus rather than once per suite.  Yahoo's were built by whichever
    suite carried ``emit_build_job``, which puts a dataset property on a Suite and
    makes "which model do we run first" load-bearing for a reason unrelated to the
    model.  A new corpus gets its own tree instead, and every model suite over it
    emits neither the build job nor the embedding sweeps.

    The jobs' resources -- partitions, memory, wall -- come from ``Suite()``'s
    defaults, which is what the yahoo embed jobs already ran under.  Nothing here
    loads a base model, so no suite's model-specific tuning applies.
    """
    global SUITE, DATA_TREE
    SUITE = Suite(job_prefix=_data_tree_prefix())
    DATA_TREE = True

    tree = slug()
    exp = root / "experiments" / tree
    jobs = root / "jobs" / tree
    logs = f"{output_dir()}/logs"
    exp.mkdir(parents=True, exist_ok=True)
    jobs.mkdir(parents=True, exist_ok=True)

    written: list[str] = []

    def emit(path: Path, text: str) -> None:
        path.write_text(text)
        written.append(str(path))

    for seed in SEEDS:
        emit(exp / f"sweep_s{seed:02d}.yaml", write_sweep(seed))
        emit(jobs / f"02_embed_s{seed:02d}.sh", sbatch(
            f"{SUITE.job_prefix}_embed_s{seed:02d}", SUITE.gpu_partitions, True,
            48, "2:00:00",
            f"python scripts/run_experiment.py experiments/{tree}/sweep_s{seed:02d}.yaml"
            f" --steps build extract --taxonomy dataset_embedding",
            logs,
        ))
    emit(exp / "embed_matrix.yaml", write_embed_matrix())
    emit(jobs / "02_embed_matrix.sh", sbatch(
        f"{SUITE.job_prefix}_embed_matrix", SUITE.gpu_partitions, True,
        48, "2:00:00",
        f"python scripts/run_experiment.py experiments/{tree}/embed_matrix.yaml"
        f" --steps build extract --taxonomy dataset_embedding",
        logs,
    ))
    emit(jobs / "01_build.sh", sbatch(
        f"{SUITE.job_prefix}_build", CPU_PARTITION, False, 32, "1:00:00",
        "\n".join(
            f"python scripts/run_experiment.py"
            f" experiments/{tree}/sweep_s{seed:02d}.yaml --steps build"
            for seed in SEEDS
        ),
        logs,
    ))

    props = proportions()
    n_sizes = len(SPEC.sweep_size_list)
    emit(jobs / "submit_all.sh", f"""#!/bin/bash
# GENERATED BY scripts/gen_simplex3.py --dataset {SPEC.name_prefix} --data-tree
# -- do not edit by hand.
#
# The {SPEC.name_prefix} dataset level, built once and shared by every model suite over
# this corpus. No model is loaded anywhere in this tree: DatasetEmbeddingCache
# keys on recipe and embedder, so the centroids these jobs write are the same
# centroids whichever base model is later studied against them.
#
# CHECK THE QUEUE FIRST -- the partition ordering and the walls were chosen
# against a snapshot of free capacity, and a snapshot written into a config is
# stale by the time it runs:
#
#     sinfo -o "%P %a %D %t %G"
#     squeue --states=PD -o "%P %R" | sort | uniq -c

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p {output_dir()}/logs

sb() {{ sbatch --parsable "$@"; }}

# One cheap CPU job: it writes {len(props) * len(SEEDS) * n_sizes} recipe blocks and nothing else, so it is a
# fail-fast gate rather than the sampling work. Draws are materialised on demand
# by the sample cache during embedding.
BUILD=$(sb 01_build.sh)
echo "build         $BUILD"

for s in {' '.join(f'{seed:02d}' for seed in SEEDS)}; do
  J=$(sb --dependency=afterok:$BUILD 02_embed_s$s.sh)
  echo "embed   s$s   $J"
done

J=$(sb --dependency=afterok:$BUILD 02_embed_matrix.sh)
echo "embed   mat   $J"

echo
echo "Submitted. Watch with: squeue -u $USER -o '%.10i %.14j %.9P %.2t %.10M %R'"
""")
    (jobs / "submit_all.sh").chmod(0o755)

    print(f"Wrote {len(written)} files:")
    print(f"  {len(list(exp.glob('*.yaml')))} configs in {exp}")
    print(f"  {len(list(jobs.glob('*.sh')))} scripts in {jobs}")
    print(f"\n{len(props)} proportions x {len(SEEDS)} seeds x {n_sizes} sizes = "
          f"{len(props) * len(SEEDS) * n_sizes} draws, plus {len(props)} for embed_matrix.")
    print(f"Submit with: bash {jobs}/submit_all.sh")


#: Job-name prefixes for the dataset build trees.  They land in sacct history and
#: in log filenames and are effectively permanent, so they are written down here
#: rather than derived from the dataset name.
_DATA_TREE_PREFIXES = {"dolly": "s3dd", "oasst1": "s3od"}


def _data_tree_prefix() -> str:
    try:
        return _DATA_TREE_PREFIXES[SPEC.name_prefix]
    except KeyError:
        raise SystemExit(
            f"--data-tree has no job prefix for {SPEC.name_prefix!r}. Add one to "
            f"_DATA_TREE_PREFIXES, checking it collides with none of "
            f"{sorted(_DATA_TREE_PREFIXES.values())} or the suite prefixes."
        ) from None


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
for i in {' '.join(str(i) for i in range(SUITE.train_shards))}; do
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
{_submit_logprob()}
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


def _submit_logprob() -> str:
    """The log-prob, sweep and generation-activation stanzas, or nothing.

    Empty for a suite that emits none of them, which is what keeps the Llama
    suite's ``submit_all.sh`` byte-identical.

    Everything here is ``afterok:$TRAIN`` like the rest of extraction. Training
    is long since complete, so in practice these are submitted with no dependency
    at all -- but the dependency stays in the generated script because it is the
    record of what the chain requires, and re-running the suite from scratch has
    to work.
    """
    if not (SUITE.emit_logprob_jobs or SUITE.temperature_sweep
            or SUITE.emit_gen_activation_job):
        return ""
    toks = " ".join(SUITE.job_token(q) for q in SUITE.query_sets)
    out = ["\n# The log-probability level: what each adapter BELIEVES about the shared\n"
           "# draw, as against what it says. Nothing here disturbs an existing entry.\n"]
    if SUITE.emit_logprob_jobs:
        out.append(
            f"for q in {toks}; do\n"
            f"  J=$(sb --dependency=afterok:$TRAIN 09_logprob_input_$q.sh)\n"
            f'  echo "lp-in   $q     $J"\n'
            f"done\n"
        )
    if SUITE.temperature_sweep:
        tts = " ".join(temp_token(t) for t in SUITE.temperature_sweep)
        out.append(
            f"\n# The sweep: {len(SUITE.temperature_sweep)} temperatures x "
            f"{SUITE.sweep_shards} shards at R={SUITE.sweep_replicates}. Each point is\n"
            f"# its own cache entry -- temperature is in the sampling hash and so in the\n"
            f"# filename -- so none of these can silently reuse another's numbers.\n"
            f"for t in {tts}; do\n"
            f"  for q in {toks}; do\n"
            f"    for i in {' '.join(str(i) for i in range(SUITE.sweep_shards))}; do\n"
            f"      J=$(sb --dependency=afterok:$TRAIN"
            f" 10_behavioral_${{q}}_${{t}}_shard$i.sh)\n"
            f'      echo "sweep   $q $t $i  $J"\n'
            f"    done\n"
            f"  done\n"
            f"done\n"
        )
    if SUITE.emit_logprob_jobs:
        out.append(
            f"\n# T=0 of the same surface. Re-decodes rather than skipping: a hit needs\n"
            f"# the log-prob file too, and greedy re-decodes exactly.\n"
            f"for q in {toks}; do\n"
            f"  J=$(sb --dependency=afterok:$TRAIN 11_greedy_logprob_$q.sh)\n"
            f'  echo "lp-gr   $q     $J"\n'
            f"done\n"
        )
    if SUITE.emit_gen_activation_job:
        out.append(
            f"\n# Generation-mode activations. Writes beside the input-mode files in the\n"
            f"# same directory; save_activations skips paths that exist.\n"
            f"for q in {toks}; do\n"
            f"  J=$(sb --dependency=afterok:$TRAIN 12_functional_gen_$q.sh)\n"
            f'  echo "func-gen $q    $J"\n'
            f"done\n"
        )
    return "".join(out)


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
            f"for i in {' '.join(str(i) for i in range(SUITE.behavioral_shards))}; do\n"
            f"  J=$(sb --dependency=afterok:$TRAIN 0{num + 1}_behavioral_{tok}_shard$i.sh)\n"
            f'  echo "behav   {tok}$i    $J"\n'
            f"done\n"
        )
    return "".join(out)


if __name__ == "__main__":
    main()
