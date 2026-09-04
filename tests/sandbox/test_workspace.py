"""The throwaway copy of the ingested checkout.

These run offline: the fixture builds a real but tiny git repository in tmp_path
rather than touching `data/repos/fastapi`.
"""

import subprocess
import tempfile

import pytest

from devagent.sandbox.workspace import PatchError, workspace


@pytest.fixture
def source_repo(tmp_path):
    """A minimal git repo standing in for the ingested checkout."""
    repo = tmp_path / "src"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=repo,
        check=True,
    )
    return repo


PATCH = """--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 2
"""


def test_workspace_copies_the_source(source_repo):
    with workspace(source_repo, None) as root:
        assert (root / "pkg" / "mod.py").read_text(encoding="utf-8") == "VALUE = 1\n"
        assert root != source_repo


def test_workspace_applies_a_patch(source_repo):
    with workspace(source_repo, PATCH) as root:
        assert (root / "pkg" / "mod.py").read_text(encoding="utf-8") == "VALUE = 2\n"


def test_the_original_checkout_is_never_modified(source_repo):
    """The ingested repo is shared state; a patch must not leak into it."""
    with workspace(source_repo, PATCH):
        pass
    assert (source_repo / "pkg" / "mod.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_the_copy_carries_no_git_history(source_repo):
    """Nothing in the copy reads history, and `.git` is what makes cleanup fail."""
    with workspace(source_repo, None) as root:
        assert not (root / ".git").exists()


def test_workspace_is_deleted_on_exit(source_repo):
    with workspace(source_repo, None) as root:
        captured = root
    assert not captured.exists()


def test_nothing_is_left_behind_in_the_temp_directory(source_repo):
    """`rmtree(ignore_errors=True)` over a copied `.git` silently leaks the lot.

    Git's loose objects are read-only, Windows refuses to unlink them, and the
    swallowed error left a directory per request behind.
    """
    with workspace(source_repo, None) as root:
        captured = root
    assert not captured.parent.exists()


def test_workspace_is_deleted_even_when_the_body_raises(source_repo):
    captured = None
    with pytest.raises(RuntimeError):
        with workspace(source_repo, None) as root:
            captured = root
            raise RuntimeError("boom")
    assert captured is not None and not captured.exists()


def test_a_patch_that_does_not_apply_raises_with_git_stderr(source_repo):
    bad = PATCH.replace("VALUE = 1", "VALUE = 99")
    with pytest.raises(PatchError) as exc:
        with workspace(source_repo, bad):
            pass
    assert "pkg/mod.py" in str(exc.value)


def test_a_failed_patch_still_deletes_the_workspace(source_repo, tmp_path, monkeypatch):
    """A rejected patch must not leak the copy it was rejected against.

    `tempfile.tempdir` is redirected so the assertion looks at exactly the
    directories this test created, not at whatever else sits in the system temp.
    """
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))

    with pytest.raises(PatchError):
        with workspace(source_repo, PATCH.replace("VALUE = 1", "VALUE = 99")):
            pass
    assert list(scratch.iterdir()) == []
