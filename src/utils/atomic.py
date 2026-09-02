"""Atomic file writes that are safe against concurrent writers.

Every cache and recipe write in this repo is "write a temp file, then
``os.replace`` it into place" -- correct against a crash, but historically the
temp name was derived from the destination alone (``path.with_suffix(".tmp")``).
Two processes writing the *same* destination therefore shared one temp path, and
the loser's ``os.replace`` raised ``FileNotFoundError`` because the winner had
already renamed the file out from under it.  That is not hypothetical: it killed
two jobs of the simplex3_nemo suite when ~55 SLURM jobs started in the same
second and all tried to materialize the same dataset recipe.

The fix is a temp name unique per process *and* per call, so concurrent writers
never contend for the temp path.  They still race on the destination, but
``os.replace`` is atomic -- last writer wins, and every reader sees one complete
file or the other, never a partial one.  For content-addressed paths (most of
this cache) the racing writers are producing identical bytes anyway, so "last
writer wins" is the correct semantics.

Temp names end in ``.tmp`` and never in a real extension, which keeps them out
of the artifact globs that scan these directories (see ``_ARTIFACT_GLOB`` in
``src/cache/_draw_keyed.py``, which matches ``*.safetensors`` precisely so an
interrupted write is not mistaken for a complete artifact).
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

__all__ = [
    "atomic_path",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
]


def _tmp_path(path: Path, suffix: str = "") -> Path:
    """A temp sibling of ``path``, unique to this process and this call.

    ``suffix`` is for writers that append an extension of their own -- notably
    ``np.savez``, which adds ``.npz`` unless the name already ends in it.  Pass
    the extension so the temp name already carries it and nothing is appended.
    """
    unique = f"{os.getpid()}.{uuid.uuid4().hex[:8]}"
    return path.with_name(f"{path.name}.{unique}.tmp{suffix}")


@contextmanager
def atomic_path(path: Path | str, suffix: str = "") -> Iterator[Path]:
    """Yield a temp path to write, then atomically move it onto ``path``.

    For writers that take a filename rather than bytes (``safetensors.save_file``,
    ``torch.save``, ``np.savez``).  The temp file is removed if the body raises,
    so a failed write leaves no litter behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path, suffix)
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically."""
    with atomic_path(path) as tmp:
        tmp.write_bytes(data)


def atomic_write_text(path: Path | str, text: str) -> None:
    """Write ``text`` to ``path`` atomically."""
    atomic_write_bytes(path, text.encode())


def atomic_write_json(path: Path | str, payload, **dumps_kwargs) -> None:
    """Serialize ``payload`` as JSON and write it to ``path`` atomically.

    ``dumps_kwargs`` passes through to :func:`json.dumps`.  Callers differ on
    ``indent`` / ``sort_keys`` / ``default`` and those choices are load-bearing
    wherever the file is hashed or diffed, so only ``indent=2`` is defaulted --
    every other flag stays the caller's decision.
    """
    dumps_kwargs.setdefault("indent", 2)
    atomic_write_text(path, json.dumps(payload, **dumps_kwargs))
