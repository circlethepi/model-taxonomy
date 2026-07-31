"""The one place upstream HuggingFace datasets are loaded and described.

The sampled-row cache stores *source indices* rather than row text, so it depends on
the upstream dataset still being there and still being in the same order.  That
dependency needs exactly one managed home rather than a ``load_dataset`` call
scattered through the sampler and the cache.

Three jobs:

- **Load, once per process.**  ``get`` memoises, so a sweep over one mixture's 19
  draws loads the 1.4M-row dataset once instead of 19 times.  Loading is otherwise
  ordinary: a cold HuggingFace cache downloads, exactly as the sampler did before.
- **Describe.**  ``describe`` builds the source descriptor that goes into a draw
  manifest, so writer and reader agree on its shape by construction.
- **Validate.**  ``validate`` compares a stored descriptor against what is on this
  machine now.

The three descriptor fields are deliberately *not* equal in authority:

``revision``, ``num_rows``
    Portable.  The revision is the Hub commit SHA, so passing it back to
    ``load_dataset`` gets the same data on a machine that has never seen this
    dataset.  Mismatches here are hard failures.
``hf_fingerprint``
    Derived from the *local* Arrow cache.  Stable across processes on one machine,
    but not guaranteed across machines or ``datasets`` versions, so a mismatch is a
    warning and never a failure.  It is recorded because when it does differ it is
    a useful hint about *why*.

The authoritative check on a rehydrated draw is neither of these — it is the
``rows_sha256`` that :class:`~src.cache.sampled_dataset_cache.SampledDatasetCache`
computes over the rows themselves.
"""

from __future__ import annotations

import json
import re
import warnings

#: Column injected to carry each row's position in the unshuffled source split.
#: Added *before* shuffling; verified not to perturb which rows ``shuffle(seed)``
#: selects, so index capture is semantics-preserving.
ROW_INDEX_COLUMN = "__row_idx"

# download_checksums keys look like
#   hf://datasets/{id}@{40-hex-sha}/{config}/train-00000-of-00002.parquet
_REVISION_RE = re.compile(r"^hf://datasets/[^@]+@([0-9a-f]{40})/")

# (dataset_id, subset, split, revision) -> Dataset
_datasets: dict[tuple, object] = {}
# id(Dataset) -> the same Dataset with ROW_INDEX_COLUMN attached
_indexed: dict[int, object] = {}
# (dataset_id, subset, split, revision) -> {row digest: row indices}
_digest_maps: dict[tuple, dict] = {}


class SourceMismatch(RuntimeError):
    """An upstream dataset no longer matches the descriptor a draw was recorded against."""


def get(
    dataset_id: str,
    subset: str | None = None,
    split: str = "train",
    revision: str | None = None,
    token: str | None = None,
):
    """Return the upstream split, memoised for the life of the process.

    *revision* pins a Hub commit.  Pass the one recorded in a draw manifest when
    rehydrating: on a machine with a cold cache that downloads the data the draw was
    actually taken from rather than whatever is currently latest.
    """
    key = (dataset_id, subset, split, revision)
    if key not in _datasets:
        from datasets import load_dataset  # type: ignore[import]

        _datasets[key] = load_dataset(
            dataset_id, subset, split=split, revision=revision, token=token
        )
    return _datasets[key]


def with_row_index(ds):
    """*ds* plus :data:`ROW_INDEX_COLUMN`, memoised alongside the bare dataset.

    Must be called before any ``shuffle``/``filter``, so the recorded index is a
    position in the original split.  Both survive downstream filtering, which is what
    lets the class-aware path record indices too.
    """
    cached = _indexed.get(id(ds))
    if cached is None:
        if ROW_INDEX_COLUMN in ds.column_names:
            cached = ds
        else:
            cached = ds.add_column(ROW_INDEX_COLUMN, list(range(len(ds))))
        _indexed[id(ds)] = cached
    return cached


def describe(ds, dataset_id: str, subset: str | None = None, split: str = "train") -> dict:
    """The descriptor recorded in a draw manifest's ``sources`` list."""
    return {
        "dataset_id": dataset_id,
        "subset": subset,
        "split": split,
        "revision": revision_of(ds),
        "num_rows": len(ds),
        "hf_fingerprint": getattr(ds, "_fingerprint", None),
    }


def revision_of(ds) -> str | None:
    """The Hub commit SHA this split was built from, or None if it cannot be determined.

    Read from ``info.download_checksums``, whose keys carry ``@{sha}``.  A dataset
    built from local files has no revision; that is recorded as None and simply not
    checked, rather than being treated as an error.
    """
    checksums = getattr(getattr(ds, "info", None), "download_checksums", None) or {}
    revisions = set()
    for url in checksums:
        m = _REVISION_RE.match(url)
        if m:
            revisions.add(m.group(1))
    # More than one revision means the split was assembled from mixed sources; refusing
    # to guess is better than pinning an arbitrary half of it.
    return revisions.pop() if len(revisions) == 1 else None


def validate(expected: dict, actual: dict) -> None:
    """Raise if the portable fields drifted; warn if only the local fingerprint did.

    Called before rehydrating a draw, so the failure names the dataset rather than
    surfacing later as mysteriously wrong text.
    """
    where = f"{expected.get('dataset_id')} [{expected.get('split')}]"

    exp_rev, act_rev = expected.get("revision"), actual.get("revision")
    if exp_rev and act_rev and exp_rev != act_rev:
        raise SourceMismatch(
            f"{where}: recorded revision {exp_rev} but this machine has {act_rev}. "
            f"The draw cannot be reproduced from this data. Re-run with "
            f"revision={exp_rev!r}, or re-sample and accept new rows."
        )

    exp_rows, act_rows = expected.get("num_rows"), actual.get("num_rows")
    if exp_rows is not None and act_rows is not None and exp_rows != act_rows:
        raise SourceMismatch(
            f"{where}: recorded {exp_rows} rows but this machine has {act_rows}. "
            f"Row indices no longer mean the same thing."
        )

    exp_fp, act_fp = expected.get("hf_fingerprint"), actual.get("hf_fingerprint")
    if exp_fp and act_fp and exp_fp != act_fp:
        warnings.warn(
            f"{where}: local Arrow fingerprint differs ({exp_fp} → {act_fp}). "
            f"Expected across machines or datasets-library versions; the rows_sha256 "
            f"check on the rehydrated draw is what actually decides.",
            UserWarning,
            stacklevel=2,
        )


def locate_rows(
    rows: list[dict], sources: list[dict], token: str | None = None
) -> list[tuple[int, int]] | None:
    """Recover ``(source_index, row_index)`` for rows whose origin was not recorded.

    Converting a stored draw to the index format by *re-running the sampler* only works
    if the sampler still produces that draw.  It often does not: sampling logic evolves,
    and several large draws in the cache predate the proportional class scale-down in
    ``ClassMixedDataset._load_entry``, so re-running yields different rows entirely.
    Matching on content instead recovers the indices of the rows that are actually
    there, which is what preserves a historical draw rather than replacing it.

    Returns None if any row cannot be found in any source — the caller then knows the
    draw cannot be represented as indices and must keep its rows.

    Rows are matched by a digest of their canonical JSON.  Where a source contains
    duplicate rows the indices are handed out in ascending order, one per occurrence;
    where several sources could supply a row the earliest is preferred.  Both are
    arbitrary but deterministic, and both are invisible downstream because every
    candidate index yields identical text.
    """
    from collections import defaultdict

    maps = [
        _row_digests(
            desc["dataset_id"], desc.get("subset"), desc.get("split", "train"),
            desc.get("revision"), token,
        )
        for desc in sources
    ]

    # Hand out each occurrence at most once, so a draw containing a row twice maps to
    # two distinct source positions rather than the same one twice.
    cursor: dict[tuple[int, bytes], int] = defaultdict(int)
    located: list[tuple[int, int]] = []
    for row in rows:
        digest = _digest(row)
        for source_index, digests in enumerate(maps):
            candidates = digests.get(digest)
            if not candidates:
                continue
            position = cursor[(source_index, digest)]
            if position >= len(candidates):
                position = 0  # drawn more times than it occurs; reuse is harmless
            cursor[(source_index, digest)] = position + 1
            located.append((source_index, candidates[position]))
            break
        else:
            return None
    return located


def _digest(row: dict) -> bytes:
    import hashlib

    return hashlib.blake2b(json.dumps(row).encode(), digest_size=16).digest()


def _row_digests(
    dataset_id: str, subset: str | None, split: str, revision: str | None, token: str | None
) -> dict[bytes, list[int]]:
    """``digest -> row indices`` for a whole split, memoised.

    Building this means hashing every row of the source, which for a 1.4M-row split is
    ~100 s.  Memoising it is the difference between one such pass for a whole migration
    and one per draw.
    """
    key = (dataset_id, subset, split, revision)
    cached = _digest_maps.get(key)
    if cached is None:
        ds = get(dataset_id, subset, split, revision=revision, token=token)
        cached = {}
        for row_index, row in enumerate(ds):
            cached.setdefault(_digest(dict(row)), []).append(row_index)
        _digest_maps[key] = cached
    return cached


def clear_cache() -> None:
    """Drop the memoised datasets.  For tests that need a cold load."""
    _datasets.clear()
    _indexed.clear()
    _digest_maps.clear()
