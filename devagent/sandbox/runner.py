"""Running a repository's tests without trusting them.

The tests being executed are not ours: they come from a cloned repository and
may have been altered by a user-supplied patch. Everything here exists to bound
what that code can do -- no network, capped memory and processes, a read-only
view of the workspace, a non-root user, and a wall-clock timeout.
"""

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

MAX_TAIL = 4000


def parse_pytest_summary(output: str) -> tuple[int, int]:
    """Return `(passed, failed)` from pytest's summary line.

    Collection errors count as failures. The caller is answering "did this
    change break any tests", and a module that will not import has broken
    every test in the file -- reporting that as zero failures is the one
    answer that is actively misleading.
    """
    counts = {"passed": 0, "failed": 0, "error": 0}
    for match in re.finditer(r"(\d+) (passed|failed|errors?)", output):
        # Assignment, not accumulation: pytest repeats the same counts across
        # the short summary and the final line.
        counts[match.group(2).rstrip("s")] = int(match.group(1))
    return counts["passed"], counts["failed"] + counts["error"]


@dataclass(frozen=True)
class TestResult:
    """The outcome of one test run.

    `ok` answers "did the run happen", not "did the tests pass". A suite with
    failures is a successful run reporting a real finding; a missing image or a
    timeout is not, and the agent must not confuse the two.
    """

    ok: bool
    exit_code: int
    passed: int
    failed: int
    duration: float
    output_tail: str
    error: str | None = None

    @classmethod
    def from_output(cls, exit_code: int, output: str, duration: float) -> "TestResult":
        passed, failed = parse_pytest_summary(output)
        # Exit 2 is pytest's "interrupted", which is what a patch breaking a
        # module at import time produces: nothing collected, one error. That
        # is a real finding about the code, so it counts as a run -- but only
        # when the output names something, or an interruption with nothing to
        # report would read as "nothing broke".
        return cls(
            ok=exit_code in (0, 1) or (exit_code == 2 and failed > 0),
            exit_code=exit_code,
            passed=passed,
            failed=failed,
            duration=duration,
            output_tail=output[-MAX_TAIL:],
        )

    @classmethod
    def failure(cls, error: str) -> "TestResult":
        return cls(False, -1, 0, 0, 0.0, "", error)


class TestRunner(Protocol):
    def run(self, workspace_root: Path, target: str) -> TestResult: ...


class DockerRunner:
    """Runs pytest in a throwaway container."""

    def __init__(self, image: str, timeout: float = 300.0) -> None:
        self.image = image
        self.timeout = timeout

    def build_command(self, workspace_root: Path, target: str) -> list[str]:
        """Return the argv for one run, or raise ValueError on a bad target.

        Split out from `run` so the isolation flags and the path check are
        assertable without Docker installed.
        """
        candidate = Path(target)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"test target must be a relative path inside tests/: {target}")
        if candidate.parts[:1] != ("tests",):
            raise ValueError(f"test target must live under tests/: {target}")
        # The tutorial tests write into the work tree, which is mounted
        # read-only. Refusing them beats running them and reporting the
        # resulting OSErrors as tests the patch broke.
        if candidate.parts[:2] == ("tests", "test_tutorial"):
            raise ValueError(
                f"tutorial tests write into the work tree, which is read-only: {target}"
            )

        return [
            "docker", "run", "--rm",
            "--network=none",
            "--memory=2g",
            "--cpus=2",
            "--pids-limit=512",
            "--user", "1000:1000",
            "-v", f"{workspace_root}:/work:ro",
            "--tmpfs", "/tmp",
            "-w", "/work",
            # PYTHONPATH puts the patched copy ahead of the package installed
            # in the image. Without it pytest imports site-packages and every
            # run reports on unpatched code -- passing, always, wrongly.
            "-e", "PYTHONPATH=/work",
            # /work is mounted read-only and the run user owns no home, so
            # bytecode writes and anything resolving ~ would fail for reasons
            # that have nothing to do with the tests.
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-e", "HOME=/tmp",
            self.image,
            "python", "-m", "pytest", "-p", "no:cacheprovider", "--tb=short",
            f"/work/{candidate.as_posix()}",
        ]

    def run(self, workspace_root: Path, target: str) -> TestResult:
        try:
            argv = self.build_command(workspace_root, target)
        except ValueError as exc:
            return TestResult.failure(str(exc))

        started = time.monotonic()
        try:
            completed = subprocess.run(
                argv, capture_output=True, text=True, timeout=self.timeout
            )
        except subprocess.TimeoutExpired:
            return TestResult.failure(f"test run exceeded {self.timeout:.0f}s and was killed")
        except FileNotFoundError:
            return TestResult.failure("docker is not installed or not on PATH")

        duration = time.monotonic() - started
        output = completed.stdout + completed.stderr
        result = TestResult.from_output(completed.returncode, output, duration)
        if not result.ok:
            return TestResult.failure(
                f"the test container exited {completed.returncode}: {output[-MAX_TAIL:]}"
            )
        return result
