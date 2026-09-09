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

**Sampled mixtures** -- ``grid`` alone decides the *resolution*, and the
enumeration then decides the *count*, which for an exhaustive grid is not a
free choice: ``C(grid + K - 1, K - 1)`` grows fast, so a 1% grid over three
groups is 5151 mixtures and there is no grid denominator that yields exactly
1000.  ``n_mixtures`` decouples the two by drawing a fixed-size subset of the
grid points instead, which is how the group-size sweep builds a pool of a
chosen size.  See ``docs/notes/group_size_sweep.md``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

#: Repository root, absolute, so a job started from a worktree still finds the
#: derived sources.  Mirrors ``gen_simplex3.REPO``, which cannot be imported
#: from here without a cycle.
_REPO = "/weka/scratch/jhu/cpriebe1/MO/model-taxonomy"

#: Where ``scripts/build_oasst1_pairs.py`` writes.  A local directory of parquet
#: rather than a Hub id, so ``source_registry.revision_of`` returns None and
#: ``validate`` guards it on ``num_rows`` alone -- which is why the ``v1`` is in
#: the path.  See that script's docstring.
OASST1_PAIRS = f"{_REPO}/results/shared_cache/00_sources/oasst1_pairs_v1"


def _compositions(total: int, parts: int):
    """Every way to write *total* as an ordered sum of *parts* non-negative ints."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


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

    #: **Sampled mixture mode.**  ``None`` enumerates every grid point, which is
    #: what all three original corpora do.  An integer instead draws that many
    #: *distinct* grid points uniformly without replacement, which is the only
    #: way to reach a pool of hundreds of mixtures: the exhaustive enumeration is
    #: ``C(grid + K - 1, K - 1)``, so a 1% grid over three groups is 5151 points,
    #: not 1000.
    #:
    #: The sample excludes the ``K`` pure vertices, which are then appended
    #: alongside the even mixture as *reference* mixtures.  So a sampled spec
    #: yields ``n_mixtures + K + 1`` proportions at K not dividing ``grid``, the
    #: sampled block carries no pure endpoint -- which is what makes
    #: ``simplex_geometry`` rather than the barycentric route the only way to
    #: score it -- and the references keep the names their adapters already have
    #: on disk, so a pool built this way retrains none of them.
    n_mixtures: int | None = None

    #: Seed for that draw.  Load-bearing: it names which 1000 of the 5151 points
    #: were trained, so changing it invalidates every adapter in the pool.
    mixture_seed: int = 0

    #: The draw sizes of the embedding sweep, rendered into
    #: ``n_samples_sweep:``.  A string renders literally (``tens 3`` expands to
    #: ``[1, 10, 100, 1000]``); a list renders as itself.
    sweep_sizes: object = "tens 3"

    train_n: int = 1000
    train_seed: int = 0
    #: The fine-tuning budget.  The samples-seen figure in every adapter name is
    #: derived from this and the suite's effective batch, never written down.
    total_train_samples: int = 5000

    #: **The training grid** -- an *nsweep* is a sweep over ``n_samples``, the
    #: size of the *training* draw, holding the mixture grid, the seeds and every
    #: optimizer setting fixed.  Empty means "the single draw named by
    #: ``train_n`` and ``train_seed``", which is what every spec that has already
    #: run carries and what keeps their trees byte-identical.  Non-empty is the
    #: cross product, so ``len(train_sizes) * len(train_seeds)`` draws of every
    #: mixture get trained.
    #:
    #: Not to be confused with ``sweep_sizes`` above, which varies the draw a
    #: *dataset representation* is computed over and exists at every size
    #: regardless of what was trained.  The two are independent axes that both
    #: render as an ``n``, which is the trap; see docs/terminology.md.
    train_sizes: tuple[int, ...] = ()
    train_seeds: tuple[int, ...] = ()

    #: **Budget mode.**  ``None`` keeps the fixed ``total_train_samples`` budget
    #: for every draw, which is what a single-draw spec wants.  An int makes the
    #: budget ``budget_per_sample * n``, i.e. a fixed number of *epochs* rather
    #: than a fixed number of samples, so every rung of an nsweep sees its own
    #: data the same number of times.  Without this a 10-row draw and a
    #: 10000-row draw would differ in two things at once and neither could be
    #: read as the effect of size.
    budget_per_sample: int | None = None

    query_n: int = 100
    query_seed: int = 1

    embedder_model: str = "nomic-ai/nomic-embed-text-v1.5"

    #: ``None`` inherits the suite's sweep; ``()`` drops it.  Dropping the
    #: temperature sweep is a decision about how much a *dataset* is worth
    #: measuring, not about the model, which is why it can be overridden here.
    temperature_sweep: tuple[float, ...] | None = None

    #: The sample-size sweep's seeds.  ``None`` inherits the generator's ten,
    #: ``()`` drops the sweep entirely and leaves only the ``embed_matrix`` job
    #: that embeds the training draws themselves -- the same None-inherits /
    #: empty-drops idiom as ``temperature_sweep`` above, for the same reason.
    #:
    #: It exists because the sweep is quadratic in the wrong thing: it embeds
    #: ``proportions x seeds x sizes`` draws, which is 640 for a 16-point corpus
    #: and 40,160 for a 1004-point one.  A pool built to measure *group size* has
    #: no use for the sample-size sweep, and paying for it would cost more GPU
    #: time than the fine-tuning does.
    sweep_seeds: tuple[int, ...] | None = None

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

    # -- the mixtures this spec names ------------------------------------------

    @property
    def step_pct(self) -> int:
        """Percentage points between adjacent grid points.

        Mixtures are labelled in whole percent -- ``100g1_000g2_000g3`` -- and
        that label is a component of every adapter directory on disk, so the grid
        must land on integer percentages or the names cannot express it.  A grid
        denominator that does not divide 100 is therefore rejected here rather
        than silently floored to zero.
        """
        if 100 % self.grid:
            raise ValueError(
                f"grid={self.grid} does not divide 100, so its mixtures have no "
                "whole-percent label. Adapter directory names are built from a "
                "three-digit percentage per group, so only grids dividing 100 "
                "(4, 5, 10, 20, 25, 50, 100) can be named."
            )
        return 100 // self.grid

    def grid_pcts(self) -> list[tuple[int, ...]]:
        """Every grid point, as whole-percent tuples, in enumeration order.

        Lexicographic on the first ``K-1`` parts with the last determined, which
        is the order the original hand-written triple loop in
        ``scripts/gen_simplex3.py`` produced.  Load-bearing only for byte-parity
        with the trees already on disk, but that is reason enough.
        """
        step = self.step_pct
        return [tuple(part * step for part in parts)
                for parts in _compositions(self.grid, self.n_groups)]

    def vertex_pcts(self) -> list[tuple[int, ...]]:
        """The ``K`` pure mixtures, in group order."""
        k = self.n_groups
        return [tuple(100 if i == j else 0 for i in range(k)) for j in range(k)]

    def mixture_pcts(self) -> list[tuple[int, ...]]:
        """The mixtures this spec asks for, exhaustive or sampled.

        Exhaustive (``n_mixtures is None``) is the grid plus the even mixture
        when the even mixture is not already a grid point -- unchanged from the
        enumeration the three original corpora ran under.

        Sampled draws ``n_mixtures`` distinct grid points uniformly without
        replacement from the **non-vertex** grid points, then appends the ``K``
        vertices and the even mixture.  Three consequences worth stating:

        * the sampled block contains no pure endpoint, so the vertices are
          *reference* models rather than members of the sample;
        * the appended references carry exactly the names an exhaustive run at a
          coarser grid already gave them, so their adapters, activations and
          generations are reused rather than retrained;
        * the draw is taken in enumeration order after sorting the chosen
          indices, so the emitted list depends on ``mixture_seed`` and on nothing
          else -- not on the iteration order of any set.
        """
        grid_pcts = self.grid_pcts()
        if self.n_mixtures is None:
            out = list(grid_pcts)
            if not self.even_is_grid_point:
                out.append(self.even_pct)
            return out

        vertices = self.vertex_pcts()
        vertex_set = set(vertices)
        pool = [pct for pct in grid_pcts if pct not in vertex_set]
        if self.n_mixtures > len(pool):
            raise ValueError(
                f"n_mixtures={self.n_mixtures} exceeds the {len(pool)} non-vertex "
                f"points of the 1/{self.grid} grid over {self.n_groups} groups. "
                "Raise `grid` to a finer denominator that still divides 100."
            )
        rng = random.Random(self.mixture_seed)
        chosen = sorted(rng.sample(range(len(pool)), self.n_mixtures))
        out = [pool[i] for i in chosen]
        out.extend(vertices)
        if not self.even_is_grid_point:
            out.append(self.even_pct)
        return out

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

    def train_grid(self) -> tuple[tuple[int, int], ...]:
        """The ``(n_samples, seed)`` training draws this spec asks for.

        A spec that never set ``train_sizes``/``train_seeds`` yields exactly the
        one pair it has always trained, so the grid is not a new concept for the
        four specs that predate it -- it is the old behaviour spelled as a
        one-element case.
        """
        sizes = self.train_sizes or (self.train_n,)
        seeds = self.train_seeds or (self.train_seed,)
        return tuple((n, s) for n in sizes for s in seeds)

    def budget(self, n: int | None = None) -> int:
        """The fine-tuning budget for a training draw of *n* rows.

        Under the default (``budget_per_sample is None``) the budget is the flat
        ``total_train_samples``, whatever the draw size -- which is what every
        existing spec means.  Under budget mode it is proportional to the draw,
        so the epoch count rather than the sample count is what is held fixed.
        """
        if self.budget_per_sample is None:
            return self.total_train_samples
        return self.budget_per_sample * (self.train_n if n is None else n)

    def samples_seen(self, effective_batch: int, n: int | None = None) -> int:
        """The budget rounded UP to a step boundary -- the ``_b5008`` token.

        The budget quantizes up because the trainer runs whole steps, so the
        number in the adapter's name is not the number in the config.  Derived
        rather than written down: it is the token the cross-suite join is keyed
        on, and a suite at another effective batch would otherwise be named for a
        budget it never saw.

        *n* is the training draw size, and only matters under budget mode.  It
        defaults to ``None`` so that every existing call site keeps naming the
        adapter it always named -- which is what makes the 16 ``_b5008`` yahoo
        adapters resolve from cache when the nsweep tree asks for them again.
        """
        return -(-self.budget(n) // effective_batch) * effective_batch

    def steps(self, effective_batch: int, n: int | None = None) -> int:
        return -(-self.budget(n) // effective_batch)


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

#: Yahoo again, at 1% resolution, with the mixtures **sampled** rather than
#: enumerated -- the pool the group-size sweep subsamples from.
#:
#: Three fields carry the whole design and each is load-bearing:
#:
#: ``grid=100`` is the finest resolution the naming scheme can express.  Adapter
#: directories label a mixture as three digits of *percent*
#: (``yahoo_100g1_000g2_000g3``), so a 0.1% grid has no name.  It costs nothing
#: scientifically: at ``train_n=1000`` a 1% weight is exactly 10 rows, so
#: ``_allocate_counts`` allocates with zero remainder and the realized mixture
#: equals the requested one, which was the only reason a fine grid was wanted.
#:
#: ``n_mixtures=1000`` draws 1000 of the 5148 non-vertex grid points.  The
#: exhaustive grid has no denominator yielding 1000, which is why the count is
#: decoupled from the resolution at all.
#:
#: ``name_prefix`` stays ``yahoo``, which is what makes the four reference models
#: free.  The three vertices and the even mixture are appended by
#: ``mixture_pcts`` under exactly the names their adapters already carry in
#: ``03_adapters/allenai--OLMo-2-0425-1B-Instruct/``, so they are reused rather
#: than retrained -- as is any of the other twelve 25%-grid points the draw
#: happens to land on.  ``suffix`` is what keeps the *trees* apart.
YAHOO_POOL = replace(
    YAHOO,
    suffix="_pool",
    grid=100,
    n_mixtures=1000,
    mixture_seed=0,
    # Neither is used by any of the four canonical perspectives this pool exists
    # to measure, and both are priced per adapter, so at 1004 adapters they would
    # dominate a suite whose entire point is that it is affordable.
    temperature_sweep=(),
    sweep_seeds=(),
    subtitle=("1000 sampled mixtures from 3 topic groupings of the Yahoo Answers "
              "Dataset, on the 1% grid"),
)


#: The **nsweep**: the same 16-point 25% grid as ``YAHOO``, crossed with ten
#: dataset seeds and four training-draw sizes.  It exists to answer one question
#: -- how much does the amount of *training* data change how well each taxonomy
#: level recovers the simplex -- and everything not on that axis is held fixed,
#: including the LoRA rank, the init seed, the mixture grid and the 100-query
#: 33/33/33 test set.
#:
#: ``budget_per_sample=5`` is five epochs at every rung.  At ``n=1000`` it
#: reproduces ``total_train_samples=5000`` exactly, which is not a coincidence
#: and is load-bearing: the sixteen ``_n1000_s00_r16_i00_b5008`` adapters the
#: yahoo tree already trained are hit by cache identity, so 624 of the 640 are
#: new and the remaining 16 double as a regression test that nothing about the
#: naming moved.
#:
#: ``name_prefix`` stays ``yahoo`` for that reuse; ``suffix`` is what keeps the
#: *trees* apart, exactly as for ``YAHOO_POOL``.
#:
#: ``sweep_sizes`` gains 10000 so the dataset level has an embedding of the
#: largest training draw.  It is the *embedding* sweep, a different axis from
#: ``train_sizes`` despite both being sizes -- see the field comments.  It is
#: deliberately left at five values while ``train_sizes`` has ten: it feeds the
#: corpus build tree, not this model tree, and no shard emitted here reads it.
YAHOO_NSWEEP = replace(
    YAHOO,
    suffix="_nsweep",
    # Deliberately NOT sorted.  `train_shard_plan` walks this tuple in order and
    # cuts shards rung by rung, so a shard's index is a function of the position
    # of its rung here.  The first four ran first; the six added on 2026-09-08
    # are appended rather than interleaved so that every already-generated and
    # already-submitted shard keeps the number -- and therefore the adapter list
    # -- it was submitted against.  Sorting this tuple would silently renumber
    # them.  Read it as "the original rungs, then the infill".
    train_sizes=(10, 100, 1000, 10000, 20, 50, 200, 500, 2000, 5000),
    train_seeds=tuple(range(10)),
    budget_per_sample=5,
    sweep_sizes=[1, 10, 100, 1000, 10000],
    # Priced per adapter and not read by any of the five canonical perspectives,
    # so at 640 adapters they would dominate a suite whose whole point is the
    # size axis.  Same reasoning as YAHOO_POOL, same two fields.
    temperature_sweep=(),
    subtitle=("Mixtures from 3 topic groupings of the Yahoo Answers Dataset, "
              "trained at ten dataset sizes over ten seeds"),
)


#: The corpora this generator knows how to emit.  ``yahoo`` is the one that
#: already ran and must regenerate unchanged.
SPECS = {"yahoo": YAHOO, "dolly": DOLLY, "oasst1": OASST1,
         "yahoo_pool": YAHOO_POOL, "yahoo_nsweep": YAHOO_NSWEEP}
