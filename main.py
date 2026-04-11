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

# Path where the ChromaDB vector database is stored on disk
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

    # Build the vector store if it doesn't exist yet (first run),
    # otherwise load the existing one from disk to save time.
    if not os.path.exists(DB_PATH):
        print("Building vector database from medical documents...")
        print("This may take several minutes on the first run.\n")
        vectorstore = build_vectorstore()
        print("Done.\n")
    else:
        # Loading the vector store takes a few seconds because the
        # embedding model (all-MiniLM-L6-v2) must be loaded into memory.
        print("Loading medical knowledge base, please wait...")
        vectorstore = load_vectorstore()
        print("Ready.\n")

    main(vectorstore)
