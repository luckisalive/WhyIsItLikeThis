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
GitHub GraphQL API → SQLite Cache → Gemini 3.5 Flash-Lite (extraction) + tree-sitter (code parsing)
    → Neo4j AuraDB (graph) → Hybrid Vector+Cypher Retriever
    → Groq openai/gpt-oss-120b (reasoning) → Streamlit UI
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
- API keys: GitHub PAT, Google Gemini (`gemini-3.5-flash-lite`, `gemini-embedding-2`), Groq Cloud (`openai/gpt-oss-120b`)

## 📦 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Graph Database | Neo4j AuraDB | Knowledge graph storage with vector search (768d cosine) |
| Extraction LLM | Google Gemini 3.5 Flash-Lite | Fast, structured JSON extraction from PRs, commits, & issues |
| Reasoning LLM | Groq (`openai/gpt-oss-120b`) | High-reasoning (117B MoE) synthesis of multi-hop graph evidence |
| Embeddings | Gemini Embedding 2 (768d) | 768-dimensional semantic embeddings (`output_dimensionality=768`) |
| Code Parsing | tree-sitter / Python AST | Extract functions, classes, config keys with zero-dependency fallback |
| Data Source | GitHub GraphQL API v4 + Git Trees | Rate-efficient batched ingestion (99% fewer API calls than REST) |
| Frontend | Streamlit | Interactive dashboard with graph visualizer & timeline |
| Caching | SQLite | Local GitHub API response cache (`data/github_cache.db`) |

## 🔮 Proposed Enhancements for v2

Based on an architectural audit of GitHub API's isolated rate-limit resource pools, the following high-value enhancements are planned for v2:

1. **Software Bill of Materials (SBOM) Ingestion (`dependency_sbom` bucket - 100 req/hr)**:
   - Ingest repository SPDX JSON dependency graphs (`/dependency-graph/sbom`).
   - Create `(d:Dependency)` nodes in Neo4j connected via `(dec:Decision)-[:GOVERNS_DEPENDENCY]->(d)` to answer questions like *"Why is Pydantic pinned to v2.x?"* or *"What dependencies govern asynchronous routing?"*.
2. **Release Version Milestones (`core` bucket - 5,000 req/hr)**:
   - Ingest GitHub releases (`/releases`) and tags to populate `(r:Release)` milestone nodes.
   - Link merged PRs and commits via `[:SHIPPED_IN]->(r)` so the system can answer which release version shipped a specific architectural decision or breaking change.
3. **Targeted Architectural Keyword Mining (`search` bucket - 30 req/min)**:
   - Utilize GitHub's Search API to execute targeted historical queries (e.g. `is:pr "breaking change"`, `is:pr label:breaking`, `is:closed "supersedes"`).
   - Ensures pivotal architectural decisions made years ago are indexed even during quick demo runs without crawling the entire repository history.


## 📁 Project Structure

```
├── settings.py             # Pydantic application settings
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
