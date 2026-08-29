"""Acquiring the target repository and deciding which files to ingest."""

import logging
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
    reproducible across runs, and refreshing is an explicit action.
    """
    repo_dir = Path(repo_dir)
    if repo_dir.exists():
        logger.info("Reusing existing checkout at %s", repo_dir)
        return repo_dir

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning %s into %s", repo_url, repo_dir)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    return repo_dir


def iter_source_files(repo_dir: Path) -> Iterator[tuple[Path, str]]:
    """Yield (absolute path, repo-relative posix path) for each file to ingest."""
    repo_dir = Path(repo_dir)
    seen: set[Path] = set()

    for pattern in _INCLUDE_PATTERNS:
        for path in sorted(repo_dir.glob(pattern)):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            yield path.resolve(), path.relative_to(repo_dir).as_posix()
