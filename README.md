# 🩺 Medical Chatbot CLI

A **command-line medical chatbot** built with Mistral AI or Google Gemini, enhanced with **RAG (Retrieval-Augmented Generation)** using the MedQuAD dataset.

---

## Features

- Interactive CLI chatbot with Rich terminal styling
- Context-aware conversations (chat history)
- RAG pipeline powered by MedQuAD medical data
- Automatic language detection (English, French, Tamil, and more)
- Multi-provider support: Mistral and Gemini
- Fast setup with `uv`
- Focused on general medical information
- Unit tested with `pytest`
- CI/CD pipeline with GitHub Actions

---

## Requirements

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) package manager
- A Mistral API key → https://console.mistral.ai
- A Google Gemini API key → https://aistudio.google.com

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

# Build the vector database (recommended: use a GPU)
uv run src/rag.py
```

---

## Configuration

Create a `.env` file at the root of the project:

```env
MISTRAL_API_KEY=your_mistral_key_here
GOOGLE_API_KEY=your_gemini_key_here
```

---

## Usage

```bash
# Run with Gemini (default)
uv run main.py

# Run with Mistral
uv run main.py --model mistral
```

---

## Available Commands

| Command         | Description                      |
|-----------------|----------------------------------|
| `reset`         | Clear conversation history       |
| `exit` / `quit` | Exit the chatbot                 |
| `Ctrl+C`        | Force exit                       |

---

## Language Support

The chatbot automatically detects the language of your message and responds accordingly.

**Supported languages:** English, French, Spanish, German, Tamil, and more.

**Note:** For best results in Tamil, use Tamil script with Gemini.
Gemini 2.5 Flash also handles transliterated Tamil better than Mistral Small.

---

## Running Tests

```bash
uv run pytest tests/ -v
```

---

## Project Structure

```text
chatbot/
├── data/
│   ├── MedQuAD/              # MedQuAD dataset
│   ├── medical_docs/         # Processed text files
│   └── chroma_db/            # Vector database
├── src/
│   ├── providers/
│   │   ├── mistral_provider.py   # Mistral AI provider
│   │   └── gemini_provider.py    # Google Gemini provider
│   ├── chatbot.py            # Chat logic with RAG and language detection
│   ├── rag.py                # RAG pipeline
│   └── prepare_data.py       # MedQuAD XML to text converter
├── tests/
│   ├── test_chatbot.py       # Unit tests for chatbot logic
│   └── test_rag.py           # Unit tests for RAG pipeline
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI pipeline
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Data Source

This project uses the [MedQuAD dataset](https://github.com/abachaa/MedQuAD)
by *Asma Ben Abacha* and *Dina Demner-Fushman*,
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## License

This project is licensed under the **MIT**.

---

## Author

**Kishanthan Kingston**
