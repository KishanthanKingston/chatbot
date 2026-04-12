# src/providers/gemini_provider.py
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
# Google Gemini provider for the medical chatbot.
# Handles client initialization and message generation.

import os
import sys
from google import genai
from google.genai import types
from rich.console import Console

console = Console()


def create_client():
    """
    Initialize and return an authenticated Gemini client.

    Returns
    -------
    genai.Client
        An authenticated Gemini client instance.

    Raises
    ------
    SystemExit
        If the GOOGLE_API_KEY environment variable is not set.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        console.print("Error: GOOGLE_API_KEY is missing.", style="bold red")
        console.print("Add it to your .env file.", style="red")
        sys.exit(1)

    return genai.Client(api_key=api_key)


def generate(client, messages):
    """
    Send messages to the Gemini API and return the reply.

    Parameters
    ----------
    client : genai.Client
        An authenticated Gemini client instance.
    messages : list of dict
        The full conversation history in Mistral format.

    Returns
    -------
    str
        The assistant's reply.
    """
    # Extract system message and convert history to Gemini format
    system_message = ""
    conversation = []

    for message in messages:
        if message["role"] == "system":
            # Gemini handles system messages separately
            system_message = message["content"]
        elif message["role"] == "user":
            conversation.append(
                types.Content(role="user", parts=[types.Part(text=message["content"])])
            )
        elif message["role"] == "assistant":
            # Gemini uses 'model' instead of 'assistant'
            conversation.append(
                types.Content(role="model", parts=[types.Part(text=message["content"])])
            )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(system_instruction=system_message),
        contents=conversation,
    )

    return response.text
