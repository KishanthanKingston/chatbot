Mistral CLI Chatbot
A simple command-line chatbot powered by the Mistral AI API.
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
Configuration
Create a `.env` file at the root of the project:
```
MISTRAL_API_KEY=your_key_here
```
Usage
```bash
uv run src/chatbot.py
```
Available commands
Command	Action
`reset`	Clear the conversation history
`exit` / `quit`	Quit the program
`Ctrl+C`	Quit the program
Project Structure
```
chatbot/
├── src/
│   └── chatbot.py   # Main chatbot logic
├── .env.example     # Example environment file
├── .gitignore
├── pyproject.toml   # Project dependencies
├── uv.lock
└── README.md
```
