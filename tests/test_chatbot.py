# chatbot.py test
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
# Unit tests for the chatbot logic.

from unittest.mock import MagicMock, patch
from src.chatbot import chat
from src.providers import mistral_provider, gemini_provider


def test_chat_appends_user_message():
    """chat() should append the user message to the history."""
    mock_client = MagicMock()
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "This is a reply."

    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search.return_value = []

    history = []
    chat(mock_client, mock_provider, history, "What is diabetes?", mock_vectorstore)

    # History should contain user message + assistant reply
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_chat_returns_reply():
    """chat() should return the assistant's reply as a string."""
    mock_client = MagicMock()
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "Diabetes is a chronic disease."

    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search.return_value = []

    history = []
    reply = chat(
        mock_client, mock_provider, history, "What is diabetes?", mock_vectorstore
    )

    assert reply == "Diabetes is a chronic disease."


def test_chat_injects_context():
    """chat() should inject RAG context into the user message."""
    mock_client = MagicMock()
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "Reply."

    mock_doc = MagicMock()
    mock_doc.page_content = "Relevant medical context."
    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search.return_value = [mock_doc]

    history = []
    chat(mock_client, mock_provider, history, "What is diabetes?", mock_vectorstore)

    # The user message sent to the API should contain the RAG context
    sent_message = history[0]["content"]
    assert "Relevant medical context." in sent_message


def test_mistral_provider_exits_without_api_key():
    """mistral_provider.create_client() should exit if MISTRAL_API_KEY is not set."""
    with patch.dict("os.environ", {}, clear=True):
        import pytest

        with pytest.raises(SystemExit):
            mistral_provider.create_client()


def test_gemini_provider_exits_without_api_key():
    """gemini_provider.create_client() should exit if GOOGLE_API_KEY is not set."""
    with patch.dict("os.environ", {}, clear=True):
        import pytest

        with pytest.raises(SystemExit):
            gemini_provider.create_client()
