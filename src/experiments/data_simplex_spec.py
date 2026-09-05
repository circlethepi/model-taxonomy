"""What varies between two *datasets* the same simplex experiment is run over.

The counterpart of :class:`src.experiments.suite.Suite`, and the split between
them is the point.  A ``Suite`` says *how* a run is configured -- base model,
dtype, LoRA targets, walls, sharding.  A ``DataSimplexSpec`` says *what corpus
the simplex is built over* -- the dataset, the categorical axis whose values
become the corners, the partition of those values into groups, the projections
of a row into text, the grid the mixtures are drawn on, the draw sizes, the
embedder, and the caveats that belong in that dataset's emitted YAML.

``scripts/gen_simplex3.py`` is then a function of the pair: ``--suite qwen
--dataset dolly`` writes one tree, ``--suite qwen`` (yahoo, the default) writes
another, and neither can drift from the other because the enumeration of the
simplex exists once.

**Yahoo's spec must reproduce the module constants it replaced, exactly.**  The
five existing suites' trees have already run, so ``--dataset yahoo`` is a
regression test: its emitted files must not move by a byte.  That is why
``embedder_model`` is a field with yahoo pinned at ``nomic-embed-text-v1.5``
even though the project is standardising on ``v2-moe``.  Yahoo reaches the new
embedder through an *additive* re-embed that writes beside the old artefacts,
not by rewriting configs whose jobs are already on disk.

**Vertex axis** -- the categorical column whose values become the pure corners.
Yahoo's is ``topic``, dolly's is ``category``, oasst1's is ``lang``.  It is the
``class_field`` in the emitted recipes; the name exists because "the class
field" says where it lives and not what it does.

**Grid denominator** -- the integer ``grid`` such that mixtures are drawn at
every multiple of ``1/grid``.  All three datasets use 4, i.e. the 25% grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Repository root, absolute, so a job started from a worktree still finds the
#: derived sources.  Mirrors ``gen_simplex3.REPO``, which cannot be imported
#: from here without a cycle.
_REPO = "/weka/scratch/jhu/cpriebe1/MO/model-taxonomy"

#: Where ``scripts/build_oasst1_pairs.py`` writes.  A local directory of parquet
#: rather than a Hub id, so ``source_registry.revision_of`` returns None and
#: ``validate`` guards it on ``num_rows`` alone -- which is why the ``v1`` is in
#: the path.  See that script's docstring.
OASST1_PAIRS = f"{_REPO}/results/shared_cache/00_sources/oasst1_pairs_v1"


@dataclass(frozen=True)
class DataSimplexSpec:
    """One corpus the simplex experiment can be run over.

    Frozen for the same reason ``Suite`` is: it is read from a dozen emission
    functions, and a late mutation would produce files that disagree with each
    other rather than an error.
    """

    #: ``""`` for yahoo, so ``experiments/simplex3`` and ``results/simplex3``
    #: keep their names.  Anything else prefixes the suite's own suffix, giving
    #: ``simplex3_dolly_qwen``.
    suffix: str = ""

    #: The token every adapter name starts with: ``yahoo_100g1_000g2_000g3``.
    #: Separate from ``suffix`` because one names directories and the other
    #: names recipes, and they are only incidentally the same word.
    name_prefix: str = "yahoo"

    dataset_id: str = "yahoo_answers_topics"

    #: The **vertex axis**: the column whose values the groups partition.
    class_field: str = "topic"

    #: ``group name -> the axis values it pools``.  Insertion order is the
    #: order of the weight vector, so it is load-bearing for every name this
    #: generator emits and for every score parsed back out of one.
    groups: dict = field(default_factory=lambda: {
        "g1": [0, 6, 7, 9],   # politics & government, business & finance,
                              # society & culture, entertainment & music
        "g2": [1, 3, 4],      # science & mathematics, computers & internet,
                              # education & reference
        "g3": [2, 5, 8],      # sports, family & relationships, health
    })

    #: Human-readable group labels for figure legends, one per group.  Falls
    #: back to the group key when a dataset does not name them.
    group_display: tuple[str, ...] = ()

    #: The training/embedding projection: every level projects the same text.
    text_fields: tuple[str, ...] = (
        "question_title", "question_content", "best_answer")

    #: The question-only projection, so the model must answer rather than
    #: continue text that already contains the answer.  Under a chat suite this
    #: is also what ``prompt_format.user_fields`` renders.
    query_fields: tuple[str, ...] = ("question_title", "question_content")

    #: The answer column.  Stated rather than derived as ``text_fields[-1]``:
    #: the identity holds for all three datasets, but relying on it makes the
    #: field order silently load-bearing, and it is also what the ``text_field:``
    #: fallback line in every emitted entry must name.
    answer_field: str = "best_answer"

    #: **Grid denominator.**  Mixtures are the compositions of this many parts.
    grid: int = 4

    #: The draw sizes of the embedding sweep, rendered into
    #: ``n_samples_sweep:``.  A string renders literally (``tens 3`` expands to
    #: ``[1, 10, 100, 1000]``); a list renders as itself.
    sweep_sizes: object = "tens 3"

    train_n: int = 1000
    train_seed: int = 0
    #: The fine-tuning budget.  The samples-seen figure in every adapter name is
    #: derived from this and the suite's effective batch, never written down.
    total_train_samples: int = 5000

    query_n: int = 100
    query_seed: int = 1

    embedder_model: str = "nomic-ai/nomic-embed-text-v1.5"

    #: ``None`` inherits the suite's sweep; ``()`` drops it.  Dropping the
    #: temperature sweep is a decision about how much a *dataset* is worth
    #: measuring, not about the model, which is why it can be overridden here.
    temperature_sweep: tuple[float, ...] | None = None

    #: Prose hazards emitted into this dataset's YAML, by where they belong.
    #: Keys: ``query`` (appended to the query-set description in every
    #: extraction config), ``train`` (the truncation note in the training
    #: configs), ``draw`` (a note about the draws themselves, in the sweep
    #: configs).  A missing key emits nothing.
    caveats: dict = field(default_factory=dict)

    #: The figure subtitle for this corpus, printed under the cross-level MDS
    #: panels. Prose, and per-corpus rather than derived, because a derived
    #: string ("mixtures of 4 groups from the oasst1 dataset") says less than the
    #: sentence someone would write, and because yahoo's already exists on
    #: figures that must not move.
    subtitle: str = "Mixtures from 3 topic groupings from the Yahoo Answers Dataset"

    #: The query-set descriptions, by query set, plus a ``chat`` key used when
    #: the suite wraps rows in a chat template -- under completion-only loss the
    #: question *is* the training prompt, so the two sets' roles invert and one
    #: description cannot serve both paths.
    query_desc: dict = field(default_factory=dict)

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def group_names(self) -> list[str]:
        return list(self.groups)

    @property
    def display_names(self) -> list[str]:
        return list(self.group_display) or self.group_names

    @property
    def sweep_size_list(self) -> list[int]:
        """The draw sizes, expanded.

        ``sweep_sizes`` renders *literally* into ``n_samples_sweep:`` so the YAML
        keeps the compact ``tens 3`` spelling the recipe parser understands, but
        the generator needs the actual sizes for its own arithmetic -- the draw
        counts in the emitted comments, and nothing else.
        """
        if isinstance(self.sweep_sizes, str):
            kind, _, count = self.sweep_sizes.partition(" ")
            if kind != "tens":
                raise ValueError(
                    f"sweep_sizes={self.sweep_sizes!r}: only the 'tens N' "
                    f"spelling can be expanded here. Give a list instead."
                )
            return [10 ** i for i in range(int(count) + 1)]
        return list(self.sweep_sizes)

    @property
    def even_pct(self) -> tuple[int, ...]:
        """The equal mixture's *label*, which rounds and need not sum to 100.

        At K=3 that is ``(33, 33, 33)``, summing to 99: only the label rounds,
        and the recipe underneath carries exact 1:1:1 weights.  At K=4 it is
        ``(25, 25, 25, 25)``, which is exact and is already a grid point.
        """
        return tuple([round(100 / self.n_groups)] * self.n_groups)

    @property
    def even_is_grid_point(self) -> bool:
        """Whether the equal mixture already appears in the grid enumeration.

        True exactly when the number of groups divides the grid denominator, in
        which case it must not be appended a second time.
        """
        return self.grid % self.n_groups == 0

    def samples_seen(self, effective_batch: int) -> int:
        """The budget rounded UP to a step boundary -- the ``_b5008`` token.

        ``total_train_samples`` quantizes up because the trainer runs whole
        steps, so the number in the adapter's name is not the number in the
        config.  Derived rather than written down: it is the token the
        cross-suite join is keyed on, and a suite at another effective batch
        would otherwise be named for a budget it never saw.
        """
        return -(-self.total_train_samples // effective_batch) * effective_batch

    def steps(self, effective_batch: int) -> int:
        return -(-self.total_train_samples // effective_batch)


#: Yahoo, the original.  Every field is the module constant it replaced in
#: ``scripts/gen_simplex3.py``, so ``--dataset yahoo`` regenerates the five
#: existing trees byte-for-byte.
YAHOO = DataSimplexSpec(
    caveats={
        # Measured on a 1000-row draw of the even mixture at seed 1:
        #     question_title      empty in   0.0%
        #     question_content    empty in  46.3%
        #     best_answer         empty in   2.2%
        # So query set B is a bare title for about half its prompts.  That does
        # not invalidate the ablation -- B is still question-only -- but it
        # belongs in any caption comparing the two, and it means the
        # three-field training composition is effectively title+answer for half
        # the corpus.
        "query": "question_content is empty in ~46% of yahoo rows",
        "train": ("max_seq_length 512 is held for consistency, but the "
                  "composition is three\nfields here rather than two, so "
                  "truncation is more frequent than in the\nquestion_title + "
                  "best_answer runs. Flagged, not changed."),
    },
    query_desc={
        "full_context": "title + content + answer, matching the training composition",
        "question_only": "title + content only, question-only ablation",
        "chat": "title + content only, matching the training prompt",
        "suffix": ", so ~half these prompts are a bare title",
    },
)

#: databricks-dolly-15k, partitioned into four groups of two categories.
#:
#: The partition is measured rather than chosen by theme alone -- the last
#: column of the design document's table is each group's share of rows carrying
#: a non-empty ``context``, which runs 45.4 / 40.2 / 40.7 / 0.0 percent.  g4 is
#: the odd one out: ``creative_writing`` and ``open_qa`` never carry context at
#: all, so that vertex differs from the other three in row *shape* as well as in
#: task, which is a confound to state rather than to discover in a figure.
DOLLY = DataSimplexSpec(
    suffix="_dolly",
    name_prefix="dolly",
    dataset_id="databricks/databricks-dolly-15k",
    class_field="category",
    groups={
        "g1": ["classification", "closed_qa"],           # 3909 rows, 45.4% context
        "g2": ["summarization", "brainstorming"],        # 2954 rows, 40.2% context
        "g3": ["information_extraction", "general_qa"],  # 3697 rows, 40.7% context
        "g4": ["creative_writing", "open_qa"],           # 4451 rows,  0.0% context
    },
    group_display=("classification+closed_qa", "summarization+brainstorming",
                   "information_extraction+general_qa", "creative_writing+open_qa"),
    text_fields=("instruction", "context", "response"),
    query_fields=("instruction", "context"),
    answer_field="response",
    embedder_model="nomic-ai/nomic-embed-text-v2-moe",
    # Dropped, not inherited: greedy plus R=16 at T=1.0 is the whole behavioral
    # level for this dataset.  See docs/notes/simplex_dolly_oasst1_implementation.md.
    temperature_sweep=(),
    caveats={
        "query": ("`context` is empty for 100% of g4 (creative_writing, open_qa) "
                  "and present in ~40-45% of the other three groups"),
        "train": ("max_seq_length 512 is held for consistency with the yahoo "
                  "suites.\nMeasured character lengths of instruction+context+"
                  "response: 453 median,\n1704 at p90, 5300 at p99, 27311 max, "
                  "with 7.2% of rows over 2048\ncharacters -- roughly the "
                  "512-token limit. Flagged, not changed."),
        "draw": ("The `context` share is not uniform across the vertices: g4 "
                 "never carries one\nand the other three do ~40-45% of the "
                 "time, so a mixture's g4 weight moves\nthe row *shape* as well "
                 "as the task."),
    },
    subtitle="Mixtures of 4 instruction-category groupings from databricks-dolly-15k",
    query_desc={
        "full_context": "instruction + context + response, matching the training composition",
        "question_only": "instruction + context only, question-only ablation",
        "chat": "instruction + context only, matching the training prompt",
        "suffix": "",
    },
)

#: oasst1's best-reply pairs, four language vertices.
#:
#: Built by ``scripts/build_oasst1_pairs.py`` -- oasst1 ships conversation trees
#: of individual messages, not prompt/response rows, so the join is a versioned
#: preprocessing step rather than something a config can express.
#:
#: n=500 rather than 1000, and the sweep tops out at 500 for the same reason:
#: only three of the four vertices clear a 1000-row pool.  ``zh`` has 738 rows.
OASST1 = DataSimplexSpec(
    suffix="_oasst1",
    name_prefix="oasst1",
    dataset_id=OASST1_PAIRS,
    class_field="lang",
    groups={
        "g1": ["en"],   # 7837 best-reply pairs
        "g2": ["es"],   # 5256
        "g3": ["ru"],   # 1527
        "g4": ["zh"],   #  738
    },
    group_display=("en", "es", "ru", "zh"),
    text_fields=("prompt", "response"),
    query_fields=("prompt",),
    answer_field="response",
    sweep_sizes=[1, 10, 100, 500],
    train_n=500,
    total_train_samples=2500,
    embedder_model="nomic-ai/nomic-embed-text-v2-moe",
    temperature_sweep=(),
    caveats={
        "query": "prompts are the human turn, in the same language as the reply",
        "train": ("max_seq_length 512 is held for consistency with the yahoo "
                  "suites.\nMeasured character lengths of prompt+response: 850 "
                  "median, 2049 at p90 --\nroughly the 512-token limit at p90. "
                  "Flagged, not changed."),
        "draw": ("The ten sweep seeds are NOT ten independent samples at the "
                 "small vertices.\n`zh` has 738 rows, so an n=500 draw is 68% of "
                 "its pool on every seed, and\n`ru` at 1527 rows is 33%. Read "
                 "the seed spread at g3/g4-heavy mixtures as\nan understatement "
                 "of the sampling variance, not as a measurement of it."),
    },
    subtitle="Mixtures of 4 languages from the oasst1 best-reply pairs",
    query_desc={
        "full_context": "prompt + response, matching the training composition",
        "question_only": "prompt only, question-only ablation",
        "chat": "prompt only, matching the training prompt",
        "suffix": "",
    },
)

#: The corpora this generator knows how to emit.  ``yahoo`` is the one that
#: already ran and must regenerate unchanged.
SPECS = {"yahoo": YAHOO, "dolly": DOLLY, "oasst1": OASST1}
