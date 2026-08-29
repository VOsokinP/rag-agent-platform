from pathlib import Path

import pytest

from devagent.ingestion.chunk_docs import chunk_markdown

FIXTURES = Path(__file__).parent.parent / "fixtures" / "mini_repo"


def load(name: str) -> str:
    return (FIXTURES / "docs" / name).read_text(encoding="utf-8")


@pytest.fixture
def guide_chunks():
    return chunk_markdown(load("guide.md"), "docs/guide.md")


def symbols(chunks) -> list[str | None]:
    return [c.symbol for c in chunks]


def test_splits_on_headings(guide_chunks):
    assert len(guide_chunks) == 4


def test_top_level_heading_is_its_own_symbol(guide_chunks):
    assert "Dependencies" in symbols(guide_chunks)


def test_nested_headings_use_a_heading_path(guide_chunks):
    names = symbols(guide_chunks)
    assert "Dependencies > First Steps" in names
    assert "Dependencies > First Steps > Deeper Detail" in names


def test_sibling_heading_pops_the_path(guide_chunks):
    # "Second Topic" is an h2, so it must not inherit "First Steps".
    assert "Dependencies > Second Topic" in symbols(guide_chunks)


def test_chunk_text_includes_the_section_body(guide_chunks):
    chunk = next(c for c in guide_chunks if c.symbol == "Dependencies > Second Topic")
    assert "Content about the second topic." in chunk.text


def test_chunk_kind_and_path(guide_chunks):
    for chunk in guide_chunks:
        assert chunk.kind == "doc"
        assert chunk.file_path == "docs/guide.md"


def test_line_numbers_are_recorded(guide_chunks):
    for chunk in guide_chunks:
        assert chunk.start_line > 0
        assert chunk.end_line >= chunk.start_line


def test_file_with_no_headings_yields_one_chunk():
    chunks = chunk_markdown(load("no_headings.md"), "docs/no_headings.md")
    assert len(chunks) == 1
    assert chunks[0].symbol is None
    assert "Just a paragraph" in chunks[0].text


def test_empty_file_yields_no_chunks():
    assert chunk_markdown("", "docs/empty.md") == []


def test_oversized_section_is_split_and_keeps_the_heading_path():
    source = "# Big\n\n" + "\n\n".join(f"Paragraph {i}." for i in range(400))
    chunks = chunk_markdown(source, "docs/big.md", max_chars=500)
    assert len(chunks) > 1
    assert all(c.symbol == "Big" for c in chunks)
    assert all(len(c.text) <= 500 for c in chunks)


FENCED_DOC = """# Tutorial

Intro paragraph.

```Python
# Create the app
from fastapi import FastAPI

# Define a path operation
@app.get("/")
def read_root():
    return {"hello": "world"}
```

Closing paragraph.
"""


def test_comments_inside_a_code_fence_are_not_headings():
    chunks = chunk_markdown(FENCED_DOC, "docs/tutorial.md")
    assert symbols(chunks) == ["Tutorial"]


def test_fenced_code_stays_in_one_chunk():
    chunk = chunk_markdown(FENCED_DOC, "docs/tutorial.md")[0]
    assert "# Create the app" in chunk.text
    assert "# Define a path operation" in chunk.text
    assert "Closing paragraph." in chunk.text


def test_tilde_fences_are_tracked_too():
    source = "# Title\n\n~~~python\n# not a heading\n~~~\n\nAfter.\n"
    assert symbols(chunk_markdown(source, "docs/t.md")) == ["Title"]


def test_a_backtick_fence_does_not_close_a_tilde_fence():
    source = "# Title\n\n~~~\n```\n# still not a heading\n~~~\n\nAfter.\n"
    assert symbols(chunk_markdown(source, "docs/t.md")) == ["Title"]


def test_headings_after_a_closed_fence_still_split():
    source = "# One\n\n```\n# fake\n```\n\n## Two\n\nBody.\n"
    assert symbols(chunk_markdown(source, "docs/t.md")) == ["One", "One > Two"]


def test_single_oversized_paragraph_is_split_to_the_budget():
    source = "# Big\n\n" + ("word " * 2000)
    chunks = chunk_markdown(source, "docs/big.md", max_chars=500)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)
    assert all(c.symbol == "Big" for c in chunks)


def test_single_oversized_line_is_hard_split():
    source = "# Big\n\n" + ("x" * 3000)
    chunks = chunk_markdown(source, "docs/big.md", max_chars=500)
    assert len(chunks) >= 6
    assert all(len(c.text) <= 500 for c in chunks)


def test_long_lines_do_not_produce_colliding_line_ranges():
    """(start_line, end_line) is a DB uniqueness key; hard-split pieces must differ."""
    source = "# Title\n\n" + ("x" * 9000) + "\n\n## Next\n\nBody.\n"
    chunks = chunk_markdown(source, "docs/wide.md", max_chars=4000)
    ranges = [(c.start_line, c.end_line) for c in chunks]
    assert len(ranges) == len(set(ranges)), ranges


def test_chunk_line_ranges_stay_within_the_file():
    source = "# Title\n\n" + ("x" * 9000) + "\n\n## Next\n\nBody.\n"
    total = len(source.split("\n"))
    for chunk in chunk_markdown(source, "docs/wide.md", max_chars=4000):
        assert chunk.end_line <= total, (chunk.symbol, chunk.start_line, chunk.end_line)
