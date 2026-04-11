# Mistral Chatbot CLI.
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
# Interactive command-line chatbot powered by the Mistral API with
# conversation memory support.


import os
import sys
from mistralai import Mistral

from dotenv import load_dotenv

# Load environment variables from the .env file into os.environ,
# so os.environ.get("MISTRAL_API_KEY") can find the key automatically
# without needing to export it manually in the terminal.
load_dotenv()


def create_client():
    """
    Reads the API key from the environment and returns an authenticated Mistral client.

    This function retrieves the API key from the ``MISTRAL_API_KEY`` environment
    variable and initializes a Mistral client. If the API key is not found, the
    program exits immediately with an error message.

    Returns
    -------
    Mistral
        An authenticated Mistral client instance.

    Raises
    ------
    SystemExit
        If the ``MISTRAL_API_KEY`` environment variable is not set.
    """

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable is missing.")
        print("Export your key: export MISTRAL_API_KEY='your_key'")
        sys.exit(1)
    return Mistral(api_key=api_key)


def chat(client, history, user_input):
    """
    Send a conversation history to the Mistral API and return the assistant's reply.

    The function appends the user's input to the conversation history, sends the
    full history to the API, and then appends the assistant's response to maintain
    conversational context across multiple turns.

    Parameters
    ----------
    client : Mistral
        An authenticated Mistral client instance.
    history : list of dict
        The conversation history. Each message is a dictionary with the keys
        "role" (e.g., "system", "user", "assistant") and
        "content" (str).
    user_input : str
        The user's input message.

    Returns
    -------
    str
        The assistant's reply extracted from the API response.

    Raises
    ------
    Exception
        Propagates any exception raised by the API call.
    """
    # Add the user's message to the history before sending
    history.append({"role": "user", "content": user_input})

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=history,
    )

    # Extract the text reply from the API response
    reply = response.choices[0].message.content

    # Store the assistant's reply so the model can reference it in future turns
    history.append({"role": "assistant", "content": reply})

    return reply


def main():
    """
    Run an interactive command-line chatbot using the Mistral API.

    This function initializes the client, sets up the initial system prompt,
    and enters a loop to handle user input. Special commands allow the user to
    exit the program or reset the conversation history.

    Notes
    -----
    Supported commands:
    - ``exit`` or ``quit``: Exit the program.
    - ``reset``: Clear conversation history while preserving the system prompt.

    The chatbot maintains context by sending the full conversation history
    with each API request.
    """

    client = create_client()

    # The system message sets the assistant's behavior for the whole conversation.
    # It is always kept as the first element of the history (see 'reset' below).
    history = [
        {
            "role": "system",
            "content": """You are a specialized medical assistant with expertise in general medicine.

        Your role:
        - Answer questions about symptoms, diseases, medications, and healthcare
        - Provide clear and accurate information based on established medical knowledge
        - Always recommend consulting a healthcare professional for diagnosis or treatment
        - Refuse to answer non-medical questions politely

        Important rules:
        - Never diagnose a patient
        - Never recommend a specific dosage
        - Always add a disclaimer when discussing serious conditions
        - If the user describes an emergency, tell them to call 911 immediately
        """,
        }
    ]

    print("=== Mistral Chatbot ===")
    print("Type 'exit' or 'quit' to quit, 'reset' to clear history.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            # Gracefully handle Ctrl+C and Ctrl+D
            print("\nGoodbye!")
            break

        if not user_input:
            # Ignore empty input instead of sending a blank message to the API
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if user_input.lower() == "reset":
            # Keep only the system message (index 0) to start a fresh conversation
            history = history[:1]
            print("History cleared.\n")
            continue

        try:
            reply = chat(client, history, user_input)
            print(f"\nAssistant: {reply}\n")
        except Exception as e:
            # Catch API errors (network issues, invalid key, rate limits…)
            # without crashing the whole program
            print(f"API error: {e}\n")


if __name__ == "__main__":
    # Only run main() when the script is executed directly,
    # not when imported as a module
    main()
