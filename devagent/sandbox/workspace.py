"""A throwaway copy of the ingested checkout, optionally patched.

Tools must never read or write the ingested checkout directly: it is shared
state that ingestion and every other request depend on, and a patch applied in
place would leak into unrelated answers.
"""

import logging
import os
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)


class PatchError(RuntimeError):
    """The supplied patch could not be applied to the checkout."""


@contextmanager
def workspace(repo_dir: Path, patch: str | None) -> Iterator[Path]:
    """Yield a temp copy of `repo_dir` with `patch` applied, then delete it."""
    root = Path(tempfile.mkdtemp(prefix="devagent-ws-"))
    target = root / repo_dir.name
    try:
        # `.git` is deliberately left behind. Nothing in the copy reads history
        # -- blame runs against the source checkout, and `git apply` does not
        # need a repository -- while git's loose objects are read-only, which is
        # enough to defeat the cleanup below and leak a copy per request.
        shutil.copytree(
            repo_dir, target, symlinks=True, ignore=shutil.ignore_patterns(".git")
        )
        if patch:
            _apply(target, patch)
        yield target
    finally:
        _remove(root)


def _apply(target: Path, patch: str) -> None:
    completed = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=target,
        input=patch,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        # git's own stderr names the file and hunk that failed; a generic
        # message would leave the caller guessing at their own patch.
        raise PatchError(completed.stderr.strip() or "git apply failed")


def _remove(root: Path) -> None:
    """Delete the workspace, complaining rather than failing if it will not go.

    This runs in a `finally`, so it must not raise and mask the real error. It
    must not go quiet either: `ignore_errors=True` is how a leaked copy per
    request stays invisible until the disk fills.
    """

    def retry(func, path, exc):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except OSError:
            log.warning("could not delete %s from the workspace: %s", path, exc)

    shutil.rmtree(root, onexc=retry)
