"""What varies between two *query mixtures* of the same fixed adapter fleet.

The third dataclass of the simplex generator, and the split is the same one
``Suite`` and ``DataSimplexSpec`` already make.  A ``Suite`` says *how* a run is
configured -- base model, dtype, walls, sharding.  A ``DataSimplexSpec`` says
*what corpus the simplex is built over*.  A :class:`QMixSpec` says *what the
probe is made of*: the composition of the 100-row query draw that a fleet of
already-trained adapters is generated from.

**qmix** ("query mixture") is the axis this file parameterizes: the fraction of
the query set that is the adapters' own training corpus, the rest being a
diluting corpus the adapters never saw.  One value of qmix is one
``(yahoo percentage, diluent)`` pair.  It varies the *probe*, never the adapter.

It is not any of the three sizes already in the project, and the distinction is
worth stating because all four are "a mixture" or "a size" in casual speech:

* the **mixture simplex** is what an adapter was *trained* on
  (``yahoo_100g1_000g2_000g3``) -- fixed here at the existing 16 points;
* ``nsamples_train`` is the size of the *training* draw -- fixed here at 1000;
* ``n_samples_sweep`` is the size of the *embedding* draw for the dataset level
  -- not used here at all;
* ``qmix`` is the composition of the *query* draw.

Nothing in this file trains anything.  A qmix tree emits extraction jobs only.

**Why a spec rather than a ``DataSimplexSpec`` entry.**  A ``DataSimplexSpec``
is single-corpus by construction: ``dataset_id``, ``class_field``,
``text_fields`` and ``answer_field`` are scalars on it, because the simplex it
describes is a partition of *one* corpus into groups.  A query mixture spans two
corpora with disjoint column names, so each side needs its own set of those
five fields.  The recipe layer already supports this -- ``ClassDatasetEntry``
carries ``dataset_id`` and ``class_field`` per entry, and
``src/datasets/_text_projection.row_text`` dispatches a row to whichever entry
names a column the row actually has -- so what is missing is only a place to
write the pair down.

**The union prompt format.**  Under a chat suite the prompt is rendered from
``prompt_format.user_fields``, not from the recipe's ``text_fields``, and
``_join_fields`` (``src/datasets/_chat_projection.py``) silently skips a column
the row does not have.  With the stock yahoo field list every diluent row would
therefore render an *empty user turn*, with no error anywhere -- 99 empty
prompts at the 1% point.  ``user_fields`` here is the **union** of both sides'
question projections, so each row renders in its own natural shape, and a yahoo
row renders byte-identically to how it rendered in training because the extra
names are simply absent from it.

That union changes ``PromptFormat.format_id()``, which is a component of adapter
directory names.  A qmix tree must therefore pin ``extraction.models`` to the
sixteen literal adapter paths named under the *stock* format rather than
re-deriving them; ``scripts/gen_simplex3.py:qmix_adapter_paths`` is where that
decoupling happens.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from src.experiments.data_simplex_spec import DOLLY, OASST1, OASST1_PAIRS_CJK, YAHOO


def _all_classes(spec) -> list:
    """Every value of a spec's vertex axis, pooled across its groups.

    A qmix draw is deliberately **unstratified**: the point of the axis is how
    much of the probe is yahoo, so imposing a second, per-group quota on top of
    that would confound the dilution with a change in the yahoo mixture itself.

    Sorted rather than left in group order.  ``class_filter`` is a membership
    test, so the order does not select different rows -- but it is emitted into
    YAML a person reads to check that a pool really is the whole corpus, and
    ``[0, 6, 7, 9, 1, 3, ...]`` does not read as "all ten topics".
    """
    return sorted(v for values in spec.groups.values() for v in values)


@dataclass(frozen=True)
class QMixArm:
    """One diluting corpus, and how a row of it becomes a query.

    Frozen for the same reason ``Suite`` and ``DataSimplexSpec`` are: it is read
    from several emission functions, and a late mutation would produce files
    that disagree with each other rather than an error.
    """

    #: The ``--qmix`` arm key, and the token in every emitted file name.
    key: str

    #: The token inside a recipe name: ``qmix_yahoo001_dolly099_qtc``.  Short,
    #: because it sits next to a three-digit percentage in a path component.
    name_token: str

    dataset_id: str

    #: The vertex axis of the diluent, used only to *filter* it -- there are no
    #: groups in a qmix draw, only a pool.
    class_field: str

    #: Which values of that axis the pool is.  All of them for dolly; ``zh``
    #: alone for oasst1, which is what makes this arm a language contrast rather
    #: than a second task contrast.
    class_filter: list

    #: The diluent's question-only projection -- what the recipe entry names, and
    #: therefore how ``row_text`` recognises a row of this corpus.
    query_fields: tuple[str, ...]

    #: The diluent's answer column: the ``text_field:`` fallback of its entry.
    answer_field: str

    #: How many rows the pool holds, for the emitted header.  Stated so a reader
    #: can see at a glance that a 100-row draw is not near the pool's capacity;
    #: ``ClassMixedDataset`` would otherwise scale a mixture down and only warn.
    pool_rows: int

    #: One sentence describing what dilution by this corpus is a contrast in.
    contrast: str

    #: Rows this arm's probe draws, when it differs from the spec's ``query_n``.
    #: None for both qmix arms; set by **qmix_v2**, whose two pools are drawn
    #: once at their own sizes (2000 dolly, 859 CJK) and subsampled afterwards.
    query_n: int | None = None


#: dolly, all eight categories pooled: an English instruction corpus.  Same
#: language as yahoo, different task shape -- an instruction with an optional
#: context, rather than a forum question.
DOLLY_ARM = QMixArm(
    key="dolly",
    name_token="dolly",
    dataset_id=DOLLY.dataset_id,
    class_field=DOLLY.class_field,
    class_filter=_all_classes(DOLLY),
    query_fields=DOLLY.query_fields,
    answer_field=DOLLY.answer_field,
    pool_rows=15011,
    contrast="same language, different task shape",
)

#: oasst1 best-reply pairs restricted to ``zh``: a different language entirely.
#:
#: 738 rows, not the whole oasst1 corpus.  That is the pool
#: ``scripts/build_oasst1_pairs.py`` leaves after its ``rank == 0`` best-reply
#: rule, which drops the 264 unranked ``zh`` singles -- the same pruning the
#: oasst1 training experiments ran under, so the two are the same corpus.
#: A 100-row draw is 14% of it, comfortably inside capacity.
OASST1ZH_ARM = QMixArm(
    key="oasst1zh",
    name_token="oasstzh",
    dataset_id=OASST1.dataset_id,
    class_field=OASST1.class_field,
    class_filter=["zh"],
    query_fields=OASST1.query_fields,
    answer_field=OASST1.answer_field,
    pool_rows=738,
    contrast="different language entirely",
)


#: oasst1 best-reply pairs in zh, ja and ko: the **oasst1cjk** arm of
#: **qmix_v2**.  The same ``rank == 0`` rule as ``OASST1ZH_ARM``, over the
#: ``oasst1_pairs_cjk`` table, which also reads oasst1's validation split: 859
#: rows (zh 768, ja 86, ko 5).  The qmix_v2 draw takes all of them -- it is the
#: pool every oasst1cjk subsample is cut from -- so n equals capacity here on
#: purpose, not by accident.
OASST1CJK_ARM = QMixArm(
    key="oasst1cjk",
    name_token="oasstcjk",
    dataset_id=OASST1_PAIRS_CJK,
    class_field=OASST1.class_field,
    class_filter=["ja", "ko", "zh"],
    query_fields=OASST1.query_fields,
    answer_field=OASST1.answer_field,
    pool_rows=859,
    contrast="different language entirely",
    query_n=859,
)


@dataclass(frozen=True)
class QMixSpec:
    """The query-mixture axis: what the probe is made of, at every point of it.

    One instance describes a whole qmix tree -- both diluents and the shared
    undiluted point -- because the 100%-yahoo probe belongs to neither arm and
    duplicating it would build the same draw twice under two names.
    """

    #: The corpus the fixed adapters were trained on, and the numerator of the
    #: qmix percentage.  Its ``groups`` are pooled, not honoured as groups.
    base_spec: object = YAHOO

    #: The diluting corpora, in emission order.
    arms: tuple[QMixArm, ...] = (DOLLY_ARM, OASST1ZH_ARM)

    #: Yahoo percentages of the probe, per arm.  ``0`` is the fully-diluted
    #: floor: what the levels recover from a probe the adapters' corpus is
    #: entirely absent from.  ``100`` is emitted once, outside the arms.
    yahoo_pcts: tuple[int, ...] = (0, 1, 2, 5, 10, 20, 50)

    #: The draw seeds every composition is repeated at.  Ten, because at 1% the
    #: probe holds exactly *one* yahoo row and the identity of that row is then
    #: the dominant source of variance; a single draw would measure none of it.
    seeds: tuple[int, ...] = tuple(range(10))

    #: Rows per probe.  Whole-percent targets against 100 rows, so
    #: ``_allocate_counts``'s largest-remainder allocation is exact at every
    #: point and the realized composition equals the requested one.
    query_n: int = 100

    #: The token every emitted recipe name starts with.  Deliberately **not** the
    #: ``NNNg1_NNNg2`` shape the mixture simplex uses: ``g1``/``g2`` mean
    #: *training* groups everywhere else in this project, and
    #: ``src/plots/simplex.py`` parses exactly that shape out of adapter ids, so
    #: a query set wearing it would invite a parse that means the wrong thing.
    name_prefix: str = "qmix"

    #: The tree's directory token: ``experiments/simplex3_qmix_olmo2``.
    suffix: str = "_qmix"

    #: Job-name prefix.  Written down rather than derived because it lands in
    #: sacct history and log filenames and is permanent once submitted.  Checked
    #: against the prefixes already in use: ``s3``, ``s3q``, ``s3li``, ``s3nm``,
    #: ``s3o2``, ``s3po2``, ``s3no2``, ``s3d*``, ``s3o*``.
    job_prefix: str = "s3qm"

    #: Whether the 100%-yahoo point is emitted.  qmix_v2 takes its yahoo pool
    #: from qbig_v2 instead, so it emits none.
    emit_pure: bool = True

    #: ``Suite`` fields this tree overrides, applied on top of ``--suite`` by
    #: ``write_qmix_tree``.  Empty for qmix; qmix_v2 pins sampled and greedy to
    #: l40s and sets the walls its 2000-row pools need.
    suite_overrides: dict = field(default_factory=dict)

    #: One sampled-behavioral job per adapter rather than one for the fleet.  At
    #: 100 rows the whole fleet fits one job; at 2000 rows x R=16 it would run
    #: ~13 h, so qmix_v2 shards it the way qbig_v2 does.
    shard_sampled: bool = False

    caveats: dict = field(default_factory=lambda: {
        "unstratified": (
            "The yahoo component is drawn UNSTRATIFIED -- all ten topics as one "
            "pool --\nwhere the canonical probe draws the even g1/g2/g3 mixture. "
            "So the 100%\npoint here is a NEW draw with its own recipe_hash, not "
            "the canonical probe,\nand it reuses none of that probe's cached "
            "generations. qmix numbers are\ncomparable within qmix, across yahoo "
            "percentage; they are not directly\ncomparable to the behavioral row "
            "of figure 2."
        ),
        "format": (
            "prompt_format.user_fields is the UNION of both corpora's question "
            "projections,\nbecause _join_fields skips a column the row does not "
            "have: with the stock\nyahoo list every diluent row would render an "
            "empty user turn and nothing\nwould raise. A yahoo row is unaffected "
            "-- the extra names are absent from it --\nso it renders exactly as "
            "it did in training. The union does change\nformat_id, which is a "
            "component of adapter directory names, so extraction.models\nbelow "
            "is pinned to literal paths rather than re-derived."
        ),
    })

    def n_for(self, arm: "QMixArm | None") -> int:
        """Rows the probe at *arm* draws: the arm's own size if it has one."""
        if arm is not None and arm.query_n is not None:
            return arm.query_n
        return self.query_n

    def base_query_fields(self) -> tuple[str, ...]:
        return tuple(self.base_spec.query_fields)

    def base_classes(self) -> list:
        return _all_classes(self.base_spec)

    def user_fields(self) -> list[str]:
        """The union ``prompt_format.user_fields``, for the **whole tree**.

        Base fields first, so a yahoo row's rendered order is the order it
        trained under, then each arm's in emission order.

        One union for every config rather than one per arm, and that is a
        decision rather than laziness.  ``format_id`` is a cache path component,
        so a per-arm union would put the dolly arm, the oasst arm and the shared
        100% reference point in three different subtrees -- and the reference
        point, which is what the whole curve is read against, would then sit in
        neither arm's.  Naming a column the row does not have costs nothing:
        ``_join_fields`` skips it, so every row still renders exactly its own
        fields under the tree-wide union.
        """
        fields = list(self.base_query_fields())
        for a in self.arms:
            fields += [f for f in a.query_fields if f not in fields]
        return fields

    def answer_fields(self) -> list[str]:
        """The union ``answer_fields``, in the same order and for the same reason."""
        fields = [self.base_spec.answer_field]
        for a in self.arms:
            if a.answer_field not in fields:
                fields.append(a.answer_field)
        return fields

    def points(self) -> list[tuple[QMixArm | None, int]]:
        """Every ``(arm, yahoo percentage)`` this tree probes at.

        The undiluted point is emitted once, ahead of the arms, and shared by
        both: with no diluent there is nothing to distinguish a dolly-arm
        pure-yahoo probe from an oasst-arm one, and building it twice would
        spend the same GPU hours to write the same numbers into two directories.
        """
        pure = [(None, 100)] if self.emit_pure else []
        return pure + [(arm, pct) for arm in self.arms
                       for pct in self.yahoo_pcts]

    def recipe_name(self, arm: QMixArm | None, pct: int) -> str:
        """The query recipe's name, which is also a cache path component.

        The seed is deliberately absent: the recipe does not carry it (the draw
        does), so ten seeds share one recipe file and separate downstream through
        ``n{n}_s{seed}`` in ``05_generated``.  Putting it in the name would write
        ten byte-identical recipe files and imply ten distinct recipes.
        """
        if arm is None:
            return f"{self.name_prefix}_yahoo100_qtc"
        return (f"{self.name_prefix}_yahoo{pct:03d}"
                f"_{arm.name_token}{100 - pct:03d}_qtc")

    def n_configs(self) -> int:
        return len(self.points()) * len(self.seeds)


#: **qmix_v2** (approved 2026-10-29): qmix rebuilt from one pool per corpus.
#: Instead of a separate 100-row draw at every (yahoo %, seed), each diluent is
#: drawn ONCE, at its 0%-yahoo recipe and seed 0 -- dolly at 2000 rows, oasst1cjk
#: at all 859 -- and every (arm, yahoo %, query count) cell is then composed by
#: subsampling that pool together with qbig_v2's 2000-row yahoo pool
#: (``figures/simplex3_qmix_v2_olmo2/subsample_qmix_v2.py``).  So this tree emits
#: only the two pools, and no 100% point: qbig_v2 is that point.
#:
#: Same fleet, same sampling (T=1.0, R=16, batch 2, seed 0) and same v2-moe
#: embedder as qmix.  Sampled and greedy are pinned to l40s, as qbig_v2's are,
#: so the pools share one architecture -- and therefore one random stream per
#: adapter -- with the yahoo pool they are composed with.
#:
#: Walls: qbig_v2's sampled shards ran ~67 min per adapter at 2000 rows x R=64
#: on l40s; R=16 is ~2/3 of that per row (qmix's own measurements), so ~45 min
#: against 3:00:00.  Greedy is qbig_v2's 2000-row job, ~84 min against 4:00:00.
DOLLY_ARM_2000 = replace(DOLLY_ARM, query_n=2000)

QMIX_V2 = QMixSpec(
    arms=(DOLLY_ARM_2000, OASST1CJK_ARM),
    yahoo_pcts=(0,),
    seeds=(0,),
    suffix="_qmix_v2",
    job_prefix="s3qm2",
    emit_pure=False,
    shard_sampled=True,
    suite_overrides={
        "behavioral_partitions": "l40s",
        "greedy_partitions": "l40s",
        "behav_time": "3:00:00",
        "greedy_time": "4:00:00",
    },
    caveats={
        "unstratified": (
            "A qmix_v2 POOL, not a probe: the diluent alone, drawn once at seed 0, "
            "from which\nevery (yahoo %, query count) cell is later composed by "
            "subsampling together with\nqbig_v2's 2000-row uniform yahoo pool "
            "(seed 2, format ea27ccee). Nothing reads\nthis draw as a probe in its "
            "own right."
        ),
        "format": QMixSpec().caveats["format"],
    },
)

#: Every qmix spec, kept in a registry for the same reason ``SPECS`` is: a
#: second one (a different base corpus, or a third diluent) is an entry here
#: rather than a forked generator.
QMIX_SPECS = {"yahoo": QMixSpec(), "yahoo_v2": QMIX_V2}
