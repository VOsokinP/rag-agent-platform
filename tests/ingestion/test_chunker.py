"""The greedy packer shared by the code and docs chunkers."""

from devagent.ingestion.chunker import pack


def ranges(items, max_chars, separator_len=1):
    return list(pack(items, max_chars, separator_len))


def test_everything_fitting_is_one_group():
    assert ranges(["ab", "cd"], 10) == [(0, 2)]


def test_a_group_is_closed_before_it_would_overflow():
    # "abc" + separator + "def" + separator is 8, over the limit of 7.
    assert ranges(["abc", "def"], 7) == [(0, 1), (1, 2)]


def test_the_separator_counts_towards_the_limit():
    """The off-by-one this function exists to hold in one place."""
    assert ranges(["abc", "def"], 8) == [(0, 2)]
    assert ranges(["abc", "def"], 8, separator_len=2) == [(0, 1), (1, 2)]


def test_an_item_over_the_limit_gets_its_own_group():
    """Packing never drops an item; splitting one is the caller's problem."""
    assert ranges(["x" * 20, "y"], 5) == [(0, 1), (1, 2)]


def test_no_items_produces_no_groups():
    assert ranges([], 10) == []


def test_groups_cover_every_item_exactly_once():
    items = [str(i) * i for i in range(1, 12)]
    covered = [i for start, stop in ranges(items, 15) for i in range(start, stop)]
    assert covered == list(range(len(items)))
