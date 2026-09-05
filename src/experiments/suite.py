"""What varies between two runs of the same experiment.

``scripts/gen_simplex3.py`` describes one experiment — 16 proportions over three
Yahoo topic groups, one adapter each — and emits the YAML and sbatch files that
run it.  Running that same experiment on a second base model changes about a
dozen values and none of the structure, so the values live here and the
structure stays in the generator.

**The split.**  A :class:`Suite` says *how* a run is configured: which base
model, which dtype, which LoRA targets, how the rows are wrapped, how the work is
sharded, how long the jobs may take.  The generator keeps everything that says
*what the experiment is*: the topic groups, the simplex points, the sample
counts, the seeds, the replicate count.  Change a ``Suite`` field and you get the
same experiment on different hardware or a different model; change a generator
constant and you get a different experiment.

**Defaults are exactly the values the Llama suite ran under.**  That is not
tidiness, it is the regression test: ``Suite()`` must regenerate
``experiments/simplex3`` and ``jobs/simplex3`` byte-for-byte, so a diff is proof
that lifting these out of the script changed nothing.

**Per-model fields come from the profile, not from here.**  ``torch_dtype``,
``target_modules`` and the prompt-format defaults are properties of a
*checkpoint*, and :mod:`src.models.profile` is where a checkpoint is described.
:meth:`Suite.for_model` reads them from there so that adding a third base model
is a new profile file plus one line, rather than another set of literals to keep
in sync.

Deliberately *not* here yet: job emission itself.  Generalizing sharding and
dependency chains into reusable objects is worth doing, but designing that
abstraction against a single worked example tends to produce the wrong seams; it
waits until there are two real suites to generalize from.  See
``docs/notes/TODO.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Suite:
    """One configuration of the simplex3 experiment.

    Frozen because a suite is read all over the generator and a late mutation
    would silently produce files that disagree with each other.
    """

    #: Empty for the original suite, so ``results/simplex3`` and
    #: ``experiments/simplex3`` keep their names.  Anything else suffixes them.
    tag: str = ""

    base_model: str = "meta-llama/Llama-3.1-8B"
    torch_dtype: str = "float16"
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    #: ``{}`` renders no ``prompt_format:`` block at all, which is what makes the
    #: raw path byte-identical to the pre-prompt-format generator output.
    prompt_format: dict = field(default_factory=dict)

    #: Which query sets to emit.  ``full_context`` is title+content+answer,
    #: ``question_only`` is title+content.  Named rather than lettered because
    #: the two sets' *roles* invert between a raw suite and a chat suite — under
    #: chat + completion-only loss the question-only set is the in-distribution
    #: one — and a positional letter cannot carry that.
    query_sets: tuple[str, ...] = ("full_context", "question_only")

    #: Short token used in emitted job names and filenames.  Kept separate from
    #: the descriptive identifier above so the Llama suite's ``s3_func_b`` and
    #: ``06_functional_b.sh`` stay exactly as they are: those names are recorded
    #: in sacct history and in log filenames already on disk, and renaming them
    #: would orphan that trail for no gain.
    job_tokens: dict = field(
        default_factory=lambda: {"full_context": "a", "question_only": "b"}
    )

    #: The log-probability level and the temperature sweep built on it.  All
    #: four default off, and that is the regression test rather than a
    #: preference: ``Suite()`` must still regenerate ``experiments/simplex3`` and
    #: ``jobs/simplex3`` byte-for-byte, so a level added after those files were
    #: written has to be invisible until a suite asks for it.
    emit_logprob_jobs: bool = False
    #: Generation-mode activations, greedy only.  Separate from the flag above
    #: because it needs no new code at all — ``FunctionalTaxonomy`` already
    #: hardcodes ``do_sample=False``, so one decoding point cannot collide with
    #: anything — while the log-prob jobs need the new level.
    emit_gen_activation_job: bool = False
    #: The temperatures to sweep, as a tuple.  Empty emits no sweep.  ``T=0`` is
    #: not in it: greedy is already on disk and is that point of the surface.
    temperature_sweep: tuple[float, ...] = ()
    #: Replicates per sweep point.  Half the main run's R, which is what buys the
    #: sweep ten points for the cost of five of the existing runs.
    sweep_replicates: int = 8
    #: The number of sweep *shards*, not the adapters in one.  The code slices
    #: ``names[i::sweep_shards]``, so this is a divisor; the two readings happen
    #: to coincide at yahoo's 16 adapters over 4 shards, which is how the old
    #: docstring ("adapters per sweep shard") survived.  Resize with that in
    #: mind.
    #:
    #: Four rather than the main run's eight because halving R halves the decode,
    #: so four adapters at R=8 lands at the same wall the eight-shard R=16 run
    #: was sized against — 40 job files instead of 80, at no change to the slot
    #: length the partitions were chosen for.  Raise it and regenerate if the
    #: queue is congested; nothing downstream depends on the shard count.
    sweep_shards: int = 4

    #: Shard *counts* for training and the R=16 behavioral run, read the same way
    #: as ``sweep_shards`` above.  They live here rather than in the generator
    #: because sharding is a property of how much work a run has -- which is the
    #: number of adapters, times the walls this suite was tuned to -- rather than
    #: of the model or of the corpus.  The defaults are yahoo's 16 adapters at 4
    #: and 2 adapters per shard; a 35-adapter corpus holds those ratios by
    #: raising the counts to 9 and 18 rather than by doubling every wall.
    train_shards: int = 4
    behavioral_shards: int = 8

    emit_embed_jobs: bool = True
    #: The ``matrix`` re-embed of the trained draws.  Unlike the sweep above this
    #: defaults on for every suite, because it authors a surrogate that does not
    #: exist yet under any of them and is model-free, so whichever suite runs
    #: first satisfies the rest.  A corpus with its own dataset build tree turns
    #: it off here: that tree owns it, and two trees emitting it would race.
    emit_embed_matrix_job: bool = True
    #: The standalone build job is a fail-fast gate for the *embedding* sweeps
    #: only.  Every other job runs ``--steps build ...`` and writes the recipes
    #: it needs, so a suite that emits no embed jobs does not need this one.
    emit_build_job: bool = True

    job_prefix: str = "s3"

    train_time: str = "2:00:00"
    behav_time: str = "1:30:00"
    func_time: str = "2:00:00"
    greedy_time: str = "1:30:00"
    #: Input log-probs are one forward pass per query with no decoding — the
    #: same pass the functional job runs — so an hour covers all 16 adapters.
    logprob_time: str = "1:00:00"
    #: Generation-mode activations decode like the greedy job and additionally
    #: retain a hidden state per step, so they take the sampled runs' wall.
    func_gen_time: str = "2:30:00"
    train_mem_gb: int = 80
    extract_mem_gb: int = 64

    #: Effective batch is the product, and it must stay 16: it determines
    #: ``steps_for_budget`` and therefore the ``_b5008`` suffix every adapter is
    #: named for, which is what the cross-suite join is keyed on.
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    #: Weight files the prefetch job must NOT download, as ``huggingface_hub``
    #: glob patterns.  Empty emits no ``ignore_patterns`` argument at all, which
    #: is what keeps the three existing trees regenerating byte-for-byte -- the
    #: same discipline ``emit_logprob_jobs`` and friends already follow.
    #:
    #: It exists because the prefetch job's ``allow_patterns`` includes
    #: ``'*.safetensors'`` and a repo may publish its weights twice.
    #: ``mistralai/Mistral-Nemo-Instruct-2407`` ships five sharded
    #: ``model-0000N-of-00005.safetensors`` AND a ``consolidated.safetensors``,
    #: so without this the prefetch silently pulls ~24.5 GB it never reads.
    prefetch_ignore: tuple[str, ...] = ()

    #: `nvl` was here until 2026-08-26, when it stopped existing -- sbatch now
    #: rejects the whole list with "invalid partition specified: nvl" rather than
    #: skipping the dead name, so every GPU job in the tree failed at submit.
    #: h200 took its place as the large-HBM tier.
    gpu_partitions: str = "h200,h100,l40s,a100"

    @property
    def model_slug(self) -> str:
        return self.base_model.replace("/", "--")

    @property
    def suffix(self) -> str:
        """``""`` or ``"_qwen"`` — appended to experiment/job/result directories."""
        return f"_{self.tag}" if self.tag else ""

    @property
    def effective_batch(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def job_token(self, query_set: str) -> str:
        """The short name this query set gets in emitted job names."""
        return self.job_tokens.get(query_set, query_set)

    def for_model(self, base_model: str, **overrides) -> "Suite":
        """This suite retargeted at another checkpoint.

        Pulls dtype, LoRA targets and the prompt-format defaults from the model's
        profile rather than taking them as arguments, so the description of a
        checkpoint lives in exactly one place.  Anything passed in ``overrides``
        wins, which is what the generator's ``--torch-dtype`` and
        ``--target-modules`` flags use.
        """
        from src.models.profile import resolve

        profile = resolve(base_model)
        fmt: dict = {}
        if profile.prompt_format == "chat":
            fmt = {
                "format": "chat",
                "chat_template_kwargs": dict(profile.chat_template_kwargs),
            }
        return replace(
            self,
            base_model=base_model,
            torch_dtype=profile.torch_dtype,
            target_modules=tuple(profile.lora_target_modules),
            prompt_format=fmt,
            **overrides,
        )
