import pytest

from src.retriever.RM3 import _escape_query


def test_escape_query_no_change():
    assert _escape_query("hello world") == "hello world"


def test_escape_query_with_apostrophe():
    assert _escape_query("medicare's definition") == "medicare\\'s definition"

