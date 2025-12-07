# Vault - Personal Knowledge Base

> **100% Local & Private AI-powered document search and Q&A**

Vault is a personal knowledge base that lets you index your documents and ask questions about them using local AI inference powered by [Parallax](https://github.com/GradientHQ/parallax).

## Key Features

- **Multi-format ingestion**: PDFs, DOCX, Markdown, Text, Emails (.eml), CSV, JSON
- **Semantic search**: Find information across all your documents
- **Multi-turn conversations**: Ask follow-up questions with context
- **Natural language Q&A**: Chat with your documents using local LLMs
- **Beautiful interfaces**: CLI and Web UI
- **100% Private**: Your data never leaves your machine

## Use Cases

- **Client management**: Index emails, CRM exports, project notes
- **Research**: Search across papers, notes, and references
- **Knowledge management**: Build a searchable second brain
- **Legal/Medical**: Query sensitive documents privately

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Parallax (for Mac)
git clone https://github.com/GradientHQ/parallax.git
cd parallax && pip install -e '.[mac]' && cd ..

# Install Vault dependencies
pip install chromadb pypdf python-docx rich
```

### 2. Start Parallax

In a separate terminal:
```bash
source venv/bin/activate
parallax run
```

### 3. Index Your Documents

```bash
# Index a directory
python main.py ingest ~/Documents/client-files

# Supported file types
python main.py ingest emails/         # .eml email files
python main.py ingest crm-export.csv  # CRM data
python main.py ingest notes.json      # JSON records
python main.py ingest contracts/      # PDF, DOCX, etc.
```

### 4. Start Chatting

**CLI Mode:**
```bash
python main.py chat
```

**Web Mode:**
```bash
python main.py web
# Open http://localhost:8080
```

## Commands

| Command | Description |
|---------|-------------|
| `python main.py ingest <path>` | Index documents from file or directory |
| `python main.py chat` | Interactive chat with conversation memory |
| `python main.py web` | Start web interface |
| `python main.py search <query>` | Search indexed documents |
| `python main.py stats` | Show knowledge base statistics |
| `python main.py clear` | Clear all indexed documents |

## Chat Commands

When in chat mode, you can use:
- `sources` - List indexed documents
- `history` - View conversation history
- `clear` - Reset conversation
- `help` - Show available commands
- `quit` - Exit chat

## Supported File Types

| Type | Extension | Description |
|------|-----------|-------------|
| PDF | `.pdf` | Documents, contracts, papers |
| Word | `.docx` | Microsoft Word documents |
| Markdown | `.md` | Notes, documentation |
| Text | `.txt` | Plain text files |
| Email | `.eml` | Email messages with headers |
| CSV | `.csv` | CRM exports, spreadsheets |
| JSON | `.json` | Structured data, API exports |

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│   Documents     │────▶│   Ingestion  │────▶│  ChromaDB   │
│ PDF/Email/CSV   │     │  + Chunking  │     │  Vectors    │
└─────────────────┘     └──────────────┘     └──────┬──────┘
                                                     │
                                                     ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│    Response     │◀────│   Parallax   │◀────│  RAG Query  │
│   + Sources     │     │  (Local LLM) │     │   Engine    │
└─────────────────┘     └──────────────┘     └─────────────┘
```

## Why Vault + Parallax?

- **Privacy**: Your documents and queries never leave your machine
- **Cost**: No API fees - run inference on your own hardware
- **Speed**: Local inference with optimized MLX backend for Apple Silicon
- **Control**: Choose your own models, customize everything

## Requirements

- Python 3.11+
- macOS with Apple Silicon (for MLX backend)
- 16GB+ RAM recommended


## License

MIT
