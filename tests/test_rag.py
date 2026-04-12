# rag.py test
#
# Author
# ------
# Kishanthan Kingston
#
# Copyright
# ---------
# © 2026 Kishanthan Kingston
#
# License
# -------
# MIT
#
# Description
# -----------
# Unit tests for the RAG pipeline.

from unittest.mock import MagicMock
from src.rag import get_context


def test_get_context_returns_string():
    """get_context() should return a non-empty string when results are found."""
    # Create a mock vectorstore that returns fake documents
    mock_doc = MagicMock()
    mock_doc.page_content = "Diabetes is a chronic disease."

    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search.return_value = [mock_doc, mock_doc]

    result = get_context(mock_vectorstore, "What is diabetes?")

    assert isinstance(result, str)
    assert "Diabetes" in result


def test_get_context_joins_chunks():
    """get_context() should join multiple chunks with double newlines."""
    mock_doc1 = MagicMock()
    mock_doc1.page_content = "Chunk 1"

    mock_doc2 = MagicMock()
    mock_doc2.page_content = "Chunk 2"

    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search.return_value = [mock_doc1, mock_doc2]

    result = get_context(mock_vectorstore, "question")

    assert result == "Chunk 1\n\nChunk 2"


def test_get_context_empty_results():
    """get_context() should return an empty string when no results are found."""
    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search.return_value = []

    result = get_context(mock_vectorstore, "unknown question")

    assert result == ""


def test_get_context_calls_similarity_search():
    """get_context() should call similarity_search with the correct arguments."""
    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search.return_value = []

    get_context(mock_vectorstore, "What is hypertension?", k=5)

    mock_vectorstore.similarity_search.assert_called_once_with(
        "What is hypertension?", k=5
    )
