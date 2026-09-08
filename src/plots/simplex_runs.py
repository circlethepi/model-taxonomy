"""The simplex3 runs as one table: which suites exist, and how to read them.

Every simplex3 suite writes a ``crosslevel_scores.csv`` scoring each of its
perspectives against the ground-truth mixture geometry. There are twelve of
them — four base models trained on each of three corpora — and a figure that
compares runs has to know three things that live nowhere in those files: which
directory holds which ``(base model, corpus)`` pair, what visual encoding that
pair carries, and which row of the CSV is "the structural level" for a model
whose surrogates are named after its own architecture.

This module is that table. It exists because a second cross-run figure driver
now needs it, and a registry copied into two drivers is a registry that will
disagree with itself.

``figures/simplex3_aggregate/make_figures.py`` predates this module and keeps
its own four-perspective table. It is deliberately not migrated: its figures are
tracked outputs of a *different* selection — the single-layer functional
surrogate, no last-layer structural row — and rewriting it here would silently
restate those figures rather than add new ones.

Perspectives
------------
:data:`LEVELS` is the standing per-level default set: one representation per
level, two for structural, plus the last-layer structural handle. Surrogate
labels differ across architectures — a 16-layer OLMo names its final adapter
layer ``layer 15 · o_proj`` where a 40-layer Nemo names it ``layer 39``, and
hybrid-attention Qwen prefixes both with ``full-attn · `` — so a perspective
matches by substring or by regex against the CSV's own labels rather than by an
exact string that could only ever be right for one model. See
:meth:`Perspective.resolve`.

Scores
------
Two, and they run in opposite directions: ``dcor`` is a distance correlation
with the truth (0 → 1, higher is better) and ``procrustes`` is a disparity
against it (1 → 0, **lower** is better).

The disparity is reported at an embedding dimension, and the corpora do not
share one: yahoo mixes three groups so its truth lives in ``d2``, while dolly
and oasst1 mix four and theirs lives in ``d3``. :data:`PROCRUSTES_DIMS` names
the two defensible ways to put all twelve runs on one axis — every run at
``d2``, which is one measurement throughout but cannot represent a four-vertex
truth, or every run at its own ``K-1``, which is each corpus's honest number but
makes a yahoo bar and a dolly bar different measurements. Both are emitted;
which one answers a given question is the reader's call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .simplex_suite import read_scores_csv, select_score

# ── The runs ──────────────────────────────────────────────────────────────────

#: Base model -> (legend label, colour, marker, linestyle).
#:
#: The order is **ascending parameter count** and is the order models appear in
#: every legend and bar group; size is the one axis these four can be put on
#: that is not arbitrary. Colours are Okabe-Ito, which stays separable under the
#: common colour-vision deficiencies. Marker and linestyle travel with the model
#: rather than with its position, so reordering this table cannot silently
#: reassign them.
#:
#: The labels are the HF repo ids' final segments with the parameter count made
#: explicit where the name hides it — OLMo here is the **1B** instruct model,
#: not the 7B.
MODELS: dict[str, tuple[str, str, str, str]] = {
    "allenai/OLMo-2-0425-1B-Instruct":      ("OLMo-2-1B",   "#E69F00", "^", "-."),
    "Qwen/Qwen3.5-4B":                      ("Qwen3.5-4B",  "#CC79A7", "D", ":"),
    "meta-llama/Llama-3.1-8B-Instruct":     ("Llama-3.1-8B", "#0072B2", "o", "-"),
    "mistralai/Mistral-Nemo-Instruct-2407": ("Mistral-Nemo-12B", "#D55E00", "s", "--"),
}

#: Corpus -> (label, colour, what its vertices are).
#:
#: The order is the order the three were run in, and it is also an order of
#: increasing distance from a topic mixture: yahoo mixes topics, dolly mixes
#: instruction tasks, oasst1 mixes languages. Colours are Okabe-Ito again, and
#: are disjoint from the model colours in use — no figure here encodes model and
#: corpus in colour at the same time, but a reader moving between figures should
#: not have to check.
CORPORA: dict[str, tuple[str, str, str]] = {
    "yahoo":  ("yahoo",  "#009E73", "topic groups"),
    "dolly":  ("dolly",  "#56B4E9", "instruction tasks"),
    "oasst1": ("oasst1", "#999999", "languages"),
}

#: Base model -> the tag its dolly and oasst1 suite directories are named with.
#: The yahoo directories predate the tag scheme and are named individually in
#: :data:`_YAHOO_DIRS`.
_TAGS = {
    "allenai/OLMo-2-0425-1B-Instruct": "olmo2",
    "Qwen/Qwen3.5-4B": "qwen",
    "meta-llama/Llama-3.1-8B-Instruct": "llama3i",
    "mistralai/Mistral-Nemo-Instruct-2407": "nemo",
}

#: The yahoo suite whose scores are authoritative per model. ``qwen_v4`` rather
#: than ``qwen_v3`` because v4 is the restricted run the llama figures are read
#: against.
_YAHOO_DIRS = {
    "allenai/OLMo-2-0425-1B-Instruct": "simplex3_olmo2",
    "Qwen/Qwen3.5-4B": "simplex3_qwen_v4",
    "meta-llama/Llama-3.1-8B-Instruct": "simplex3_llama3i",
    "mistralai/Mistral-Nemo-Instruct-2407": "simplex3_nemo",
}


@dataclass(frozen=True)
class Run:
    """One ``(base model, corpus)`` pair and the suite directory holding it."""

    base_model: str
    corpus: str
    figure_dir: str

    @property
    def model_label(self) -> str:
        return MODELS[self.base_model][0]

    @property
    def color(self) -> str:
        return MODELS[self.base_model][1]

    @property
    def marker(self) -> str:
        return MODELS[self.base_model][2]

    @property
    def linestyle(self) -> str:
        return MODELS[self.base_model][3]

    @property
    def corpus_color(self) -> str:
        return CORPORA[self.corpus][1]

    @property
    def label(self) -> str:
        """``"Llama-3.1-8B · dolly"`` — the name of this pair on a shared axis."""
        return f"{self.model_label} · {self.corpus}"

    def scores_path(self, figures_root: Path) -> Path:
        return Path(figures_root) / self.figure_dir / "crosslevel_scores.csv"


def _dir_for(base_model: str, corpus: str) -> str:
    if corpus == "yahoo":
        return _YAHOO_DIRS[base_model]
    return f"simplex3_{corpus}_{_TAGS[base_model]}"


#: Every run, **model-major**: all three corpora of the smallest model, then of
#: the next, and so on. That grouping is what lets a figure put the corpora of
#: one model beside each other and label the model once.
RUNS: list[Run] = [
    Run(base_model, corpus, _dir_for(base_model, corpus))
    for base_model in MODELS
    for corpus in CORPORA
]


def runs_for_corpus(corpus: str) -> list[Run]:
    """The four models' runs on one corpus, in :data:`MODELS` order."""
    return [r for r in RUNS if r.corpus == corpus]


def runs_for_model(base_model: str) -> list[Run]:
    """The three corpora's runs on one model, in :data:`CORPORA` order."""
    return [r for r in RUNS if r.base_model == base_model]


# ── Perspectives ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Perspective:
    """One taxonomy level's representation, as a row of a ``crosslevel_scores.csv``.

    *surrogate* matches the CSV's surrogate column as a case-insensitive
    substring; *pattern* matches it as a regex. Exactly one of the two is set,
    or neither when the level has a single surrogate. The regex form is for a
    label that carries a number the architecture decides — the last adapter
    layer is ``layer 15`` on OLMo and ``layer 39`` on Nemo — which no substring
    can span.
    """

    label: str
    level: str
    metric: str
    surrogate: str | None = None
    pattern: str | None = None
    color: str = "#000000"

    def resolve(self, rows: list[dict]) -> str | None:
        """The exact surrogate string this perspective names in *rows*.

        Raises if a *pattern* matches anything other than exactly one surrogate,
        for the same reason :func:`~src.plots.simplex_suite.select_score` raises
        on an ambiguous substring: a perspective that silently picked the first
        of two matching rows would put a different measurement on each model's
        bar and nothing in the figure would say so.
        """
        if self.pattern is None:
            return self.surrogate
        rx = re.compile(self.pattern, re.IGNORECASE)
        hits = sorted({r["surrogate"] for r in rows
                       if r["level"] == self.level and rx.search(r["surrogate"])})
        if len(hits) != 1:
            raise LookupError(
                f"perspective {self.label!r}: pattern {self.pattern!r} matched "
                f"{len(hits)} surrogate(s) at level {self.level!r}: {hits}"
            )
        return hits[0]


#: The standing per-level default set, in reading order: from what the model was
#: trained on, through what its weights became and what its activations do, to
#: what it says. Structural takes three rows because it is the level with a
#: choice of scope — every projection, the output projections alone, and the
#: output projection of the last layer alone — and the point of carrying all
#: three is to see whether the recipe is legible at one layer or only across the
#: stack.
#:
#: Colours are per level and are used where the *level* is the series, which is
#: the radar; the bar charts encode model or corpus instead and ignore them. The
#: three structural rows share a blue-green family so they read as one level at
#: three scopes rather than as three unrelated levels.
LEVELS: list[Perspective] = [
    Perspective("Data", "dataset_embedding", "euclidean",
                surrogate="dataset text · mean", color="#000000"),
    Perspective("Structural\nall proj.", "structural", "cosine",
                surrogate="all layers · all projections", color="#0072B2"),
    Perspective("Structural\no-proj.", "structural", "cosine",
                surrogate="output projections", color="#56B4E9"),
    Perspective("Structural\nlast layer o-proj.", "structural", "cosine",
                pattern=r"layer \d+ · o_proj", color="#009E73"),
    Perspective("Functional\nall layers", "functional", "cosine",
                surrogate="layers (reference)", color="#E69F00"),
    Perspective("Behavioral\nper query", "behavioral", "cosine",
                surrogate="R=16 · per query", color="#CC79A7"),
]


# ── Scores ────────────────────────────────────────────────────────────────────

#: Which ``procrustes_d*`` column to read, per convention.
#:
#: ``"d2"`` reads ``procrustes_d2`` for every corpus: one measurement across all
#: twelve runs, at the price of asking a four-vertex truth to fit in a plane it
#: cannot. ``"dK1"`` reads each corpus's own ``K-1`` — d2 for yahoo's three
#: groups, d3 for the four-group corpora — which is each run's honest disparity
#: but is not one measurement across the axis.
PROCRUSTES_DIMS = ("d2", "dK1")


def truth_dim(rows: list[dict]) -> int:
    """``K-1`` for the corpus these rows came from, read off its own columns.

    The suite writes one ``procrustes_d<d>`` column per dimension it fitted, up
    to the truth's own, so the largest of them is ``K-1``. Derived rather than
    looked up from the corpus name, so a corpus added later needs no entry here.
    """
    dims = [int(k.split("_d")[1]) for k in rows[0] if k.startswith("procrustes_d")]
    if not dims:
        raise LookupError("no procrustes_d* column in these rows")
    return max(dims)


def score_field(score: str, rows: list[dict], dims: str = "dK1") -> str:
    """The CSV column *score* reads, for the corpus described by *rows*.

    ``"dcor"`` is one column whatever the corpus. ``"procrustes"`` is a family,
    and *dims* picks between the two conventions in :data:`PROCRUSTES_DIMS`.
    """
    if score == "dcor":
        return "dcor"
    if score != "procrustes":
        raise ValueError(f"unknown score {score!r}")
    if dims == "d2":
        return "procrustes_d2"
    if dims == "dK1":
        return f"procrustes_d{truth_dim(rows)}"
    raise ValueError(f"dims must be one of {PROCRUSTES_DIMS}, got {dims!r}")


def read_run(figures_root: Path, run: Run) -> list[dict]:
    """This run's ``crosslevel_scores.csv``, as row dicts."""
    return read_scores_csv(run.scores_path(figures_root))


def level_scores(rows: list[dict], score: str, dims: str = "dK1",
                 levels: list[Perspective] | None = None) -> dict[str, float]:
    """``{perspective label: value}`` for one run's rows.

    Nothing is recomputed: the suites wrote these numbers, and this reads the
    one cell per level that :data:`LEVELS` names.
    """
    field = score_field(score, rows, dims)
    return {
        p.label: select_score(rows, p.level, p.metric, p.resolve(rows), field=field)
        for p in (levels if levels is not None else LEVELS)
    }
