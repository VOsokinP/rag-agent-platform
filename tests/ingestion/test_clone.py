from pathlib import Path

import pytest

from devagent.ingestion.clone import ensure_repo, iter_source_files


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """A directory shaped like the FastAPI repo's relevant subset."""
    (tmp_path / "fastapi").mkdir()
    (tmp_path / "fastapi" / "routing.py").write_text("x = 1", encoding="utf-8")
    (tmp_path / "fastapi" / "params.py").write_text("y = 2", encoding="utf-8")
    (tmp_path / "fastapi" / "nested").mkdir()
    (tmp_path / "fastapi" / "nested" / "deep.py").write_text("z = 3", encoding="utf-8")

    docs_en = tmp_path / "docs" / "en" / "docs"
    docs_en.mkdir(parents=True)
    (docs_en / "index.md").write_text("# Index", encoding="utf-8")

    docs_es = tmp_path / "docs" / "es" / "docs"
    docs_es.mkdir(parents=True)
    (docs_es / "index.md").write_text("# Indice", encoding="utf-8")

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_routing.py").write_text("pass", encoding="utf-8")

    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "build.py").write_text("pass", encoding="utf-8")

    (tmp_path / "README.md").write_text("# Readme", encoding="utf-8")
    return tmp_path


def relative_paths(repo: Path) -> set[str]:
    return {rel for _, rel in iter_source_files(repo)}


def test_includes_package_python_files(fake_repo):
    found = relative_paths(fake_repo)
    assert "fastapi/routing.py" in found
    assert "fastapi/params.py" in found


def test_includes_nested_python_files(fake_repo):
    assert "fastapi/nested/deep.py" in relative_paths(fake_repo)


def test_includes_english_docs(fake_repo):
    assert "docs/en/docs/index.md" in relative_paths(fake_repo)


def test_excludes_non_english_docs(fake_repo):
    assert "docs/es/docs/index.md" not in relative_paths(fake_repo)


def test_excludes_tests_and_scripts(fake_repo):
    found = relative_paths(fake_repo)
    assert "tests/test_routing.py" not in found
    assert "scripts/build.py" not in found


def test_excludes_top_level_readme(fake_repo):
    assert "README.md" not in relative_paths(fake_repo)


def test_yields_absolute_paths_that_exist(fake_repo):
    for absolute, _ in iter_source_files(fake_repo):
        assert absolute.is_absolute()
        assert absolute.exists()


def test_relative_paths_use_forward_slashes(fake_repo):
    # Citations must look the same on Windows and Linux.
    assert all("\\" not in rel for rel in relative_paths(fake_repo))


def test_ensure_repo_returns_existing_checkout_without_cloning(fake_repo):
    # No network: an existing directory is returned as-is.
    assert ensure_repo("https://example.invalid/nope", fake_repo) == fake_repo
