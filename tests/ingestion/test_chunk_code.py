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


@pytest.fixture
def decorated_chunks():
    return chunk_python_source(load("decorated.py"), "pkg/decorated.py")


def test_function_chunk_includes_its_decorators(decorated_chunks):
    handler = next(c for c in decorated_chunks if c.symbol == "handler")
    assert '@app.get("/foo")' in handler.text
    assert "@functools.lru_cache" in handler.text
    assert "def handler(x: int) -> int:" in handler.text


def test_class_chunk_includes_its_decorator(decorated_chunks):
    decorated = next(c for c in decorated_chunks if c.symbol == "Decorated")
    assert "@functools.total_ordering" in decorated.text


def test_method_chunk_includes_its_decorator(decorated_chunks):
    value = next(c for c in decorated_chunks if c.symbol == "Decorated.value")
    assert "@property" in value.text


def test_decorators_do_not_leak_into_the_module_chunk(decorated_chunks):
    module_chunk = next(c for c in decorated_chunks if c.symbol == "<module>")
    assert "@app.get" not in module_chunk.text
    assert "@functools.total_ordering" not in module_chunk.text
    assert "@property" not in module_chunk.text


def test_decorated_chunk_start_line_points_at_the_first_decorator(decorated_chunks):
    handler = next(c for c in decorated_chunks if c.symbol == "handler")
    lines = load("decorated.py").splitlines()
    assert lines[handler.start_line - 1].strip() == '@app.get("/foo")'


def test_module_chunk_range_covers_only_retained_lines(decorated_chunks):
    module_chunk = next(c for c in decorated_chunks if c.symbol == "<module>")
    source_lines = load("decorated.py").splitlines()
    assert module_chunk.end_line < len(source_lines)


def test_no_chunk_exceeds_the_size_cap():
    source = "def big():\n" + "\n".join(f"    x{i} = {i}" for i in range(3000))
    chunks = chunk_python_source(source, "pkg/big.py", max_chars=2000)
    assert chunks
    assert all(len(c.text) <= 2000 for c in chunks), [len(c.text) for c in chunks]


def test_split_pieces_have_distinct_line_ranges():
    """(start_line, end_line) is part of the DB uniqueness key — pieces must differ."""
    source = "def big():\n" + "\n".join(f"    x{i} = {i}" for i in range(3000))
    chunks = chunk_python_source(source, "pkg/big.py", max_chars=2000)
    ranges = [(c.start_line, c.end_line) for c in chunks]
    assert len(ranges) == len(set(ranges)), ranges


def test_split_line_numbers_match_the_source():
    source = "def big():\n" + "\n".join(f"    x{i} = {i}" for i in range(3000))
    lines = source.split("\n")
    for chunk in chunk_python_source(source, "pkg/big.py", max_chars=2000):
        assert chunk.text.split("\n")[0] == lines[chunk.start_line - 1]


def test_oversized_class_becomes_a_header_chunk_not_a_split_body():
    body = "\n".join(f"    def m{i}(self):\n        return {i}" for i in range(400))
    source = f'class Huge:\n    """Docs."""\n{body}\n'
    chunks = chunk_python_source(source, "pkg/huge.py", max_chars=2000)
    class_chunk = next(c for c in chunks if c.symbol == "Huge")
    assert "class Huge:" in class_chunk.text
    assert '"""Docs."""' in class_chunk.text
    assert "def m399" not in class_chunk.text, "body must not be in the header chunk"
    assert any(c.symbol == "Huge.m399" for c in chunks), "methods still chunked"
