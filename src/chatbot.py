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
    Exits immediately if the key is missing to avoid cryptic errors later.
    """
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable is missing.")
        print("Export your key: export MISTRAL_API_KEY='your_key'")
        sys.exit(1)
    return Mistral(api_key=api_key)


def chat(client, history, user_input):
    """
    Sends the full conversation history to the Mistral API and returns the reply.

    The history is a list of messages with a 'role' (system / user / assistant)
    and a 'content'. Sending the full history on every call is what gives the
    model its memory of the conversation.
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
    client = create_client()

    # The system message sets the assistant's behavior for the whole conversation.
    # It is always kept as the first element of the history (see 'reset' below).
    history = [
        {"role": "system", "content": "You are a helpful and concise assistant."}
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
