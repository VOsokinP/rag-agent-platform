import pytest

from devagent.sandbox.runner import DockerRunner, TestResult, parse_pytest_summary


def test_parses_a_passing_summary():
    passed, failed = parse_pytest_summary("======= 42 passed in 3.10s =======")
    assert (passed, failed) == (42, 0)


def test_parses_a_mixed_summary():
    passed, failed = parse_pytest_summary("=== 3 failed, 39 passed in 4.02s ===")
    assert (passed, failed) == (39, 3)


def test_parses_an_empty_summary_as_zeroes():
    assert parse_pytest_summary("collected 0 items") == (0, 0)


def test_docker_command_pins_the_isolation_flags(tmp_path):
    runner = DockerRunner(image="devagent-runner:fastapi", timeout=300)
    argv = runner.build_command(tmp_path, "tests/test_params_repr.py")
    joined = " ".join(argv)
    assert "--network=none" in argv
    assert "--rm" in argv
    assert "--memory=2g" in argv
    assert "--pids-limit=512" in argv
    # The patched copy must shadow the image's installed package, or every run
    # silently reports on unpatched code.
    assert "PYTHONPATH=/work" in joined
    # The mount is read-only and HOME would otherwise be unset under --user,
    # so anything writing bytecode or touching ~ fails for the wrong reason.
    assert "PYTHONDONTWRITEBYTECODE=1" in joined
    assert "HOME=/tmp" in joined
    assert "-p" in argv and "no:cacheprovider" in argv
    assert argv[-1] == "/work/tests/test_params_repr.py"


def test_docker_command_refuses_a_target_outside_tests(tmp_path):
    runner = DockerRunner(image="x", timeout=1)
    with pytest.raises(ValueError):
        runner.build_command(tmp_path, "../../etc/passwd")


def test_docker_command_refuses_an_absolute_target(tmp_path):
    runner = DockerRunner(image="x", timeout=1)
    with pytest.raises(ValueError):
        runner.build_command(tmp_path, "/etc/passwd")


def test_docker_command_refuses_a_target_outside_the_tests_directory(tmp_path):
    runner = DockerRunner(image="x", timeout=1)
    with pytest.raises(ValueError):
        runner.build_command(tmp_path, "fastapi/params.py")


def test_result_truncates_a_long_tail():
    result = TestResult.from_output(exit_code=1, output="x" * 10_000, duration=1.0)
    assert len(result.output_tail) <= 4000


def test_counts_collection_errors_as_failures():
    assert parse_pytest_summary("=========== 1 error in 8.48s ===========") == (0, 1)


def test_counts_failures_and_errors_together():
    passed, failed = parse_pytest_summary("=== 3 failed, 2 errors, 39 passed in 4.02s ===")
    assert (passed, failed) == (39, 5)


def test_a_collection_error_is_a_real_run():
    """A patch that breaks a module at import time exits 2, and that is an answer."""
    result = TestResult.from_output(
        exit_code=2, output="ERROR tests/test_x.py\n1 error in 8.48s", duration=1.0
    )
    assert result.ok
    assert result.failed == 1


def test_an_interruption_with_nothing_to_report_is_not_a_run():
    result = TestResult.from_output(exit_code=2, output="collected 0 items", duration=1.0)
    assert not result.ok


def test_docker_command_refuses_the_tutorial_tests(tmp_path):
    """They write into the work tree, which is mounted read-only.

    A clear refusal beats a run that fails on OSError and reads as "your patch
    broke 21 tests".
    """
    runner = DockerRunner(image="x", timeout=1)
    with pytest.raises(ValueError):
        runner.build_command(tmp_path, "tests/test_tutorial/test_templates/test_tutorial001.py")
