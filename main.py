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


import sys
from src.chatbot import main


def check_python_version():
    """Ensure the script is running with Python 3.8 or higher."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
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
    main()
