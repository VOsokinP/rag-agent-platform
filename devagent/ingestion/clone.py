"""Acquiring the target repository and deciding which files to ingest."""

import logging
import re
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path

logger = logging.getLogger(__name__)


def _remote_identity(url: str) -> str:
    """Reduce a git URL to `host/path`, so its ssh and https spellings match."""
    text = url.strip().lower().removesuffix(".git").rstrip("/")
    text = re.sub(r"^[a-z][a-z0-9+.-]*://", "", text)
    text = re.sub(r"^[^/@]+@", "", text)
    return re.sub(r"[:/]+", "/", text)


def _origin_url(repo_dir: Path) -> str | None:
    """The checkout's origin remote, or None if it has none we can read."""
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _empty_directory(directory: Path) -> None:
    """Remove a directory's contents, leaving the directory itself in place."""
    for child in directory.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def ensure_repo(repo_url: str, repo_dir: Path) -> Path:
    """Return a local checkout of the repo, shallow-cloning it if absent.

    An existing checkout is reused as-is rather than pulled: ingestion should be
    reproducible across runs, and refreshing is an explicit action. A directory
    that exists but holds no `.git` is treated as an error rather than reused —
    that is what an interrupted clone leaves behind, and silently ingesting it
    produces an empty index with no failure anywhere.

    Reuse is refused when the checkout's origin is a different repository:
    changing `REPO_URL` without also changing `REPO_DIR` would otherwise
    re-ingest the old checkout under the new repository's name.
    """
    repo_dir = Path(repo_dir)

    if (repo_dir / ".git").exists():
        origin = _origin_url(repo_dir)
        if origin is not None and _remote_identity(origin) != _remote_identity(
            repo_url
        ):
            raise RuntimeError(
                f"{repo_dir} is a checkout of {origin}, not {repo_url}. "
                "Point REPO_DIR at a different directory, or remove it."
            )
        logger.info("Reusing existing checkout at %s", repo_dir)
        return repo_dir

    if repo_dir.exists() and any(repo_dir.iterdir()):
        raise RuntimeError(
            f"{repo_dir} exists but is not a git checkout (no .git). "
            "It may be a partial clone from an interrupted run. "
            "Remove it and try again."
        )

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    existed_before = repo_dir.exists()
    logger.info("Cloning %s into %s", repo_url, repo_dir)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        # Clean up the partial clone so the next call doesn't accept it, but
        # never delete a directory the caller created — only what we put in it.
        if existed_before:
            _empty_directory(repo_dir)
        else:
            shutil.rmtree(repo_dir, ignore_errors=True)
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"git clone of {repo_url} failed: {stderr}") from exc

    return repo_dir


def iter_source_files(
    repo_dir: Path, include_globs: Sequence[str]
) -> Iterator[tuple[Path, str]]:
    """Yield (absolute path, repo-relative posix path) for each file to ingest.

    `include_globs` is passed in rather than known here: which files matter is a
    property of the target repository, and so belongs to configuration.

    Paths are de-duplicated by their *resolved* location and anything resolving
    outside the repository is skipped, so a symlink cycle cannot yield the same
    file many times under different citation paths, and a citation can never
    claim content lives in the checkout when it physically does not.
    """
    repo_dir = Path(repo_dir)
    repo_root = repo_dir.resolve()
    seen: set[Path] = set()

    for pattern in include_globs:
        for path in sorted(repo_dir.glob(pattern)):
            if not path.is_file():
                continue

            resolved = path.resolve()
            if not resolved.is_relative_to(repo_root):
                logger.warning("Skipping %s: resolves outside the repository", path)
                continue
            if resolved in seen:
                continue

            seen.add(resolved)
            yield resolved, resolved.relative_to(repo_root).as_posix()
