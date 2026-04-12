# src/providers/mistral_provider.py
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
# Mistral AI provider for the medical chatbot.
# Handles client initialization and message generation.

import os
import sys
from mistralai import Mistral
from rich.console import Console

console = Console()


def create_client():
    """
    Initialize and return an authenticated Mistral client.

    Returns
    -------
    Mistral
        An authenticated Mistral client instance.

    Raises
    ------
    SystemExit
        If the MISTRAL_API_KEY environment variable is not set.
    """
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        console.print("Error: MISTRAL_API_KEY is missing.", style="bold red")
        console.print("Add it to your .env file.", style="red")
        sys.exit(1)
    return Mistral(api_key=api_key)


def generate(client, messages):
    """
    Send messages to the Mistral API and return the reply.

    Parameters
    ----------
    client : Mistral
        An authenticated Mistral client instance.
    messages : list of dict
        The full conversation history.

    Returns
    -------
    str
        The assistant's reply.
    """
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=messages,
    )
    return response.choices[0].message.content
