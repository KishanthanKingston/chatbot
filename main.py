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
# Entry point for the Mistral Medical Chatbot CLI.
# Handles Python version check, banner display,
# and vector store initialization before starting the chatbot.

import os
import sys
from src.chatbot import main
from src.rag import build_vectorstore, load_vectorstore

# Path where the ChromaDB vector database is stored
DB_PATH = "data/chroma_db"


def check_python_version():
    """Ensure the script is running with Python 3.10 or higher."""
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required.")
        sys.exit(1)


def print_banner():
    """Display a welcome banner when the chatbot starts."""
    print("=" * 40)
    print("       Mistral Medical Chatbot")
    print("=" * 40)
    print("  Your AI-powered medical assistant")
    print("  Type 'exit' to quit, 'reset' to")
    print("  clear history.")
    print("=" * 40)
    print()


if __name__ == "__main__":
    check_python_version()
    print_banner()

    # Build the vector store if it doesn't exist yet, otherwise load it
    if not os.path.exists(DB_PATH):
        print("Building vector database from medical documents...")
        vectorstore = build_vectorstore()
        print("Done.\n")
    else:
        vectorstore = load_vectorstore()

    main(vectorstore)
