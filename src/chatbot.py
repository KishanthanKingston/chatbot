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
# conversation memory support and RAG pipeline.


import os
import sys
from mistralai import Mistral
from dotenv import load_dotenv
from src.rag import get_context
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from langdetect import detect, DetectorFactory

# Make language detection deterministic across all calls
DetectorFactory.seed = 0

# Initialize the Rich console for styled terminal output
console = Console()

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
        console.print(
            "Error: MISTRAL_API_KEY environment variable is missing.", style="bold red"
        )
        console.print("Export your key: export MISTRAL_API_KEY='your_key'", style="red")
        sys.exit(1)
    return Mistral(api_key=api_key)


def chat(client, history, user_input, vectorstore):
    """
    Send a conversation history to the Mistral API and return the assistant's reply.

    The function detects the language of the user's input, retrieves relevant
    medical context from the vector store, injects both into the user message,
    and maintains conversational context across multiple turns.

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
    vectorstore : Chroma
        The vector store used to retrieve relevant medical context.

    Returns
    -------
    str
        The assistant's reply extracted from the API response.

    Raises
    ------
    Exception
        Propagates any exception raised by the API call.
    """
    # Retrieve relevant medical context from the vector store
    context = get_context(vectorstore, user_input)

    # Detect the language of the user's message for each turn independently.
    # For short inputs or unreliable detection, let the model handle language matching.
    try:
        if len(user_input) >= 30:
            detected_lang = detect(user_input)
            lang_instruction = f"You must respond in language code: {detected_lang}."
        else:
            lang_instruction = "Identify the language of the question and respond in that same language."
    except Exception:
        lang_instruction = (
            "Identify the language of the question and respond in that same language."
        )

    # Inject context and language instruction into the user message
    augmented_input = f"""{lang_instruction}
If the question is written in transliterated Tamil (Tamil words written in Latin script), respond in Tamil script.

Use the following medical information to answer the question.
If the context is not relevant, rely on your general medical knowledge.

Context:
{context}

Question: {user_input}

Important: Your entire response must match the language of the question."""

    # Add the augmented message to the history before sending
    history.append({"role": "user", "content": augmented_input})

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=history,
    )

    # Extract the text reply from the API response
    reply = response.choices[0].message.content

    # Store the assistant's reply so the model can reference it in future turns
    history.append({"role": "assistant", "content": reply})

    return reply


def main(vectorstore):
    """
    Run an interactive command-line chatbot using the Mistral API.

    This function initializes the client, sets up the initial system prompt,
    and enters a loop to handle user input. Special commands allow the user to
    exit the program or reset the conversation history.

    Parameters
    ----------
    vectorstore : Chroma
        The vector store used to retrieve relevant medical context.

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
        - Detect the language of each user message and always reply in that exact same language.
          Default to English if the language cannot be determined.

        Important rules:
        - Never diagnose a patient
        - Never recommend a specific dosage
        - Always add a disclaimer when discussing serious conditions
        - If the user describes an emergency, tell them to call the local emergency number:
          15 (SAMU) or 15 in France, 112 in Europe, 911 in North America
        """,
        }
    ]

    console.print("=== Mistral Medical Chatbot ===", style="bold cyan")
    console.print(
        "Type 'exit' or 'quit' to quit, 'reset' to clear history.\n", style="dim"
    )

    while True:
        try:
            # Rich prompt for styled user input
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            # Gracefully handle Ctrl+C and Ctrl+D
            console.print("\nGoodbye!", style="bold")
            break

        if not user_input:
            # Ignore empty input instead of sending a blank message to the API
            continue

        if user_input.lower() in ("exit", "quit"):
            console.print("Goodbye!", style="bold")
            break

        if user_input.lower() == "reset":
            # Keep only the system message (index 0) to start a fresh conversation
            history = history[:1]
            console.print("History cleared.\n", style="green")
            continue

        try:
            reply = chat(client, history, user_input, vectorstore)
            console.print("\n[bold green]Assistant:[/bold green]")
            console.print(Markdown(reply))
            console.print()
        except Exception as e:
            # Catch API errors (network issues, invalid key, rate limits…)
            # without crashing the whole program
            console.print(f"API error: {e}", style="bold red")


if __name__ == "__main__":
    # Only run main() when the script is executed directly,
    # not when imported as a module
    from src.rag import load_vectorstore

    main(load_vectorstore())
