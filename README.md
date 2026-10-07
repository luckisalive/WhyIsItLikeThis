# 🔍 Why The Code Is Like This

> **Answer the question code cannot answer: *why is it like this?***

An Applied AI system that ingests a GitHub repository's history — commits, pull request discussions, issues, and design documents — constructs a **knowledge graph** connecting code entities to decisions, people, and incidents, and answers natural-language "why" questions with a chain of evidence.

---

## ✨ Features

- **Evidence-Backed Answers** — Every answer cites specific PRs, commits, issues, and people
- **Supersession Detection** — Knows when a newer decision overrode an older one
- **Confidence Signals** — Rates every answer HIGH / MEDIUM / LOW and refuses to invent when evidence is absent
- **Knowledge Graph** — Connects code entities (files, functions, config keys) to decisions, people, and events in Neo4j
- **Interactive Dashboard** — Streamlit UI with chat interface, graph visualization, and decision timeline
- **CLI Interface** — Ask questions from the command line

## 🏗️ Architecture

```
GitHub API → SQLite Cache → Groq LLM (extraction) + tree-sitter (code parsing)
    → Neo4j AuraDB (graph) → Hybrid Vector+Cypher Retriever
    → Gemini (reasoning) → Streamlit UI
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys

# 3. Initialize the graph schema
python main.py setup

# 4. Run a quick demo ingestion
python main.py ingest --quick

# 5. Ask a question
python main.py query "Why does FastAPI use Starlette?"

# 6. Launch the dashboard
streamlit run src/ui/app.py
```

## 🔧 Prerequisites

- Python 3.10+
- [Neo4j AuraDB Free](https://console.neo4j.io) instance
- API keys: GitHub PAT, Groq Cloud, Google Gemini

## 📦 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Graph Database | Neo4j AuraDB | Knowledge graph storage with vector search |
| Extraction LLM | Groq (Llama 3.1 70B) | Fast decision extraction from PR discussions |
| Reasoning LLM | Google Gemini 2.0 Flash | Evidence-based answer generation |
| Code Parsing | tree-sitter | Extract functions, classes, config keys |
| Data Source | GitHub API (PyGithub) | Commits, PRs, issues, design docs |
| Frontend | Streamlit | Interactive dashboard |
| Caching | SQLite | Local GitHub API response cache |

## 📁 Project Structure

```
├── config.py               # Pydantic settings
├── main.py                 # CLI entry point
├── src/
│   ├── models.py           # Shared data models
│   ├── pipeline.py         # Ingestion orchestration
│   ├── ingestion/          # GitHub API + caching
│   ├── extraction/         # LLM decisions + tree-sitter code parsing
│   ├── graph/              # Neo4j operations + embeddings
│   ├── supersession/       # Supersession detection
│   ├── query/              # Hybrid retrieval + answer generation
│   └── ui/                 # Streamlit dashboard
└── tests/                  # Unit tests
```

## 💬 Example Questions

```
"Why does FastAPI use Starlette as its foundation?"
"Who decided to use Pydantic for data validation?"
"Why was the dependency injection system designed this way?"
"Has the testing approach changed over time?"
"Why is the documentation built with MkDocs?"
```

## 📄 License

MIT
