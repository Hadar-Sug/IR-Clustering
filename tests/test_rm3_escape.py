import pytest

from src.retriever.RM3 import _sanitize_query


def test_sanitize_query_no_change():
    assert _sanitize_query("hello world") == "hello world"


def test_sanitize_query_with_apostrophe():
    assert _sanitize_query("medicare's definition") == "medicares definition"

