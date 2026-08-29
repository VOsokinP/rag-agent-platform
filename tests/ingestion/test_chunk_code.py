from pathlib import Path

import pytest

from devagent.ingestion.chunk_code import chunk_python_source

FIXTURES = Path(__file__).parent.parent / "fixtures" / "mini_repo"


def load(name: str) -> str:
    return (FIXTURES / "pkg" / name).read_text(encoding="utf-8")


@pytest.fixture
def sample_chunks():
    return chunk_python_source(load("sample.py"), "pkg/sample.py")


def symbols(chunks) -> list[str | None]:
    return [c.symbol for c in chunks]


def test_extracts_top_level_function(sample_chunks):
    assert "top_level_function" in symbols(sample_chunks)


def test_extracts_async_function(sample_chunks):
    assert "async_function" in symbols(sample_chunks)


def test_extracts_class(sample_chunks):
    assert "Greeter" in symbols(sample_chunks)


def test_extracts_methods_with_qualified_names(sample_chunks):
    names = symbols(sample_chunks)
    assert "Greeter.greet" in names
    assert "Greeter.greet_async" in names
    assert "Greeter.__init__" in names


def test_emits_module_chunk_for_module_level_code(sample_chunks):
    assert "<module>" in symbols(sample_chunks)


def test_module_chunk_contains_imports_and_constants(sample_chunks):
    module_chunk = next(c for c in sample_chunks if c.symbol == "<module>")
    assert "import os" in module_chunk.text
    assert "CONSTANT = 42" in module_chunk.text
    # It must NOT contain the bodies of the definitions.
    assert "return x * 2" not in module_chunk.text


def test_chunk_text_is_real_source(sample_chunks):
    fn = next(c for c in sample_chunks if c.symbol == "top_level_function")
    assert "def top_level_function(x: int) -> int:" in fn.text
    assert "return x * 2" in fn.text


def test_chunk_records_file_path_and_kind(sample_chunks):
    for chunk in sample_chunks:
        assert chunk.file_path == "pkg/sample.py"
        assert chunk.kind == "code"


def test_chunk_line_numbers_are_sane(sample_chunks):
    fn = next(c for c in sample_chunks if c.symbol == "top_level_function")
    assert fn.start_line > 0
    assert fn.end_line >= fn.start_line


def test_unparseable_file_returns_empty_list():
    assert chunk_python_source(load("broken.py"), "pkg/broken.py") == []


def test_empty_file_returns_no_chunks():
    assert chunk_python_source(load("empty.py"), "pkg/empty.py") == []
