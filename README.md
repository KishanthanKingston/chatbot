Mistral Medical Chatbot CLI

A command-line medical chatbot powered by the Mistral AI API and RAG (Retrieval Augmented Generation) using the MedQuAD dataset.

Requirements

Python 3.8+

uv package manager

A Mistral API key (console.mistral.ai)

Installation
```bash
# Clone the repository
git clone https://github.com/KishanthanKingston/chatbot.git
cd chatbot

# Install dependencies
uv sync
```

Data Setup

This project uses the MedQuAD dataset for RAG. Clone it and convert the XML files to plain text:
```bash
# Clone the MedQuAD dataset
cd data
git clone https://github.com/abachaa/MedQuAD.git
cd ..

# Convert XML files to plain text
uv run src/prepare_data.py
```

Configuration

Create a `.env` file at the root of the project:
```
MISTRAL_API_KEY=your_key_here
```

Usage
```bash
uv run main.py
```

Available commands

Command	Action

`reset`	Clear the conversation history

`exit` / `quit`	Quit the program

`Ctrl+C`	Quit the program

Project Structure
```
chatbot/
├── data/
│   ├── MedQuAD/          # MedQuAD dataset
│   ├── medical_docs/     # Converted .txt files
├── src/
│   ├── chatbot.py        # Main chatbot logic
│   ├── rag.py            # RAG pipeline
│   └── prepare_data.py   # MedQuAD XML to .txt converter
├── main.py
├── .gitignore
├── pyproject.toml        # Project dependencies
├── uv.lock
└── README.md
```

Data

This project uses the MedQuAD dataset
by Asma Ben Abacha and Dina Demner-Fushman, licensed under CC BY 4.0.
