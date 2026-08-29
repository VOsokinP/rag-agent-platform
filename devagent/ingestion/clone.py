"""Acquiring the target repository and deciding which files to ingest."""

import logging
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

INCLUDE_CODE = "fastapi/**/*.py"
INCLUDE_DOCS = "docs/en/docs/**/*.md"
_INCLUDE_PATTERNS = (INCLUDE_CODE, INCLUDE_DOCS)


def ensure_repo(repo_url: str, repo_dir: Path) -> Path:
    """Return a local checkout of the repo, shallow-cloning it if absent.

    An existing checkout is reused as-is rather than pulled: ingestion should be
    reproducible across runs, and refreshing is an explicit action. A directory
    that exists but holds no `.git` is treated as an error rather than reused —
    that is what an interrupted clone leaves behind, and silently ingesting it
    produces an empty index with no failure anywhere.
    """
    repo_dir = Path(repo_dir)

    if (repo_dir / ".git").exists():
        logger.info("Reusing existing checkout at %s", repo_dir)
        return repo_dir

    if repo_dir.exists() and any(repo_dir.iterdir()):
        raise RuntimeError(
            f"{repo_dir} exists but is not a git checkout (no .git). "
            "It may be a partial clone from an interrupted run. "
            "Remove it and try again."
        )

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning %s into %s", repo_url, repo_dir)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        # Don't leave a partial checkout that the next call would accept.
        shutil.rmtree(repo_dir, ignore_errors=True)
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"git clone of {repo_url} failed: {stderr}") from exc

    return repo_dir


def iter_source_files(repo_dir: Path) -> Iterator[tuple[Path, str]]:
    """Yield (absolute path, repo-relative posix path) for each file to ingest.

    Paths are de-duplicated by their *resolved* location and anything resolving
    outside the repository is skipped, so a symlink cycle cannot yield the same
    file many times under different citation paths, and a citation can never
    claim content lives in the checkout when it physically does not.
    """
    repo_dir = Path(repo_dir)
    repo_root = repo_dir.resolve()
    seen: set[Path] = set()

    for pattern in _INCLUDE_PATTERNS:
        for path in sorted(repo_dir.glob(pattern)):
            if not path.is_file():
                continue

            resolved = path.resolve()
            if not resolved.is_relative_to(repo_root):
                logger.warning(
                    "Skipping %s: resolves outside the repository", path
                )
                continue
            if resolved in seen:
                continue

            seen.add(resolved)
            yield resolved, resolved.relative_to(repo_root).as_posix()
