# 🩺 Mistral Medical Chatbot CLI

A powerful **command-line medical chatbot** built with the Mistral AI API and enhanced with **RAG (Retrieval-Augmented Generation)** using the MedQuAD dataset.

---

## Features

- Interactive CLI chatbot
- Context-aware conversations (chat history)
- RAG pipeline powered by medical data
- Fast setup with `uv`
- Focused on general medical information

---

## Requirements

- Python 3.8+
- [`uv`](https://github.com/astral-sh/uv) package manager
- A Mistral API key → https://console.mistral.ai

---

## Installation

```bash
# Clone the repository
git clone https://github.com/KishanthanKingston/chatbot.git
cd chatbot

# Install dependencies
uv sync
```

---

## Data Setup

This project uses the **MedQuAD dataset** for the RAG pipeline.

```bash
# Clone the dataset
cd data
git clone https://github.com/abachaa/MedQuAD.git
cd ..

# Convert XML files to plain text
uv run src/prepare_data.py
```

---

## Configuration

Create a `.env` file at the root of the project:

```env
MISTRAL_API_KEY=your_key_here
```

---

## Usage

```bash
uv run main.py
```

---

## Available Commands

| Command         | Description                      |
|-----------------|----------------------------------|
| `reset`         | Clear conversation history       |
| `exit` / `quit` | Exit the chatbot                 |
| `Ctrl+C`        | Force exit                       |

---

## Project Structure

```text
chatbot/
├── data/
│   ├── MedQuAD/          # MedQuAD dataset
│   ├── medical_docs/     # Processed text files
├── src/
│   ├── chatbot.py        # Chat logic
│   ├── rag.py            # RAG pipeline
│   └── prepare_data.py   # XML (text converter)
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Data Source

This project uses the **MedQuAD dataset**
by *Asma Ben Abacha* and *Dina Demner-Fushman*,
licensed under **CC BY 4.0**.

---

## License

This project is licensed under the **MIT**.

---

## Author

**Kishanthan Kingston**
