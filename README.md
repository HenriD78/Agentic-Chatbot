# Agentic RAG with LangGraph and Ollama

Local, multi-agent RAG pipeline orchestrated with LangGraph. The system routes user questions through a supervisor agent to a coding agent that generates and executes DuckDB SQL over Excel/Parquet-style data, then synthesizes the response.

## Setup

1) Prereqs: Python 3.10+, `pip`, and [Ollama](https://ollama.com/) with the supervisor and coding models pulled locally.  
2) Install deps:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3) Configure environment:
   - Copy `.env.example` to `.env`
   - Update model names/paths as needed (see below)
4) Add data: place Excel files in `./data/`. Each sheet will become a DuckDB table.

## Running

### Prerequisites

1. **Install Ollama**: Download and install from [ollama.com](https://ollama.com/)
2. **Pull required models**:
   ```bash
   ollama pull ministral-3:3b
   ollama pull qwen2.5-coder:7b
   ```
3. **Start Ollama** (if not running as a service):
   ```bash
   ollama serve
   ```

### Usage

CLI entry point:
```bash
python -m src.main --question "How has the number of sessions changed since 2024, by market?"
```

With verbose output (shows SQL queries and detailed results):
```bash
python -m src.main --question "Which pages generate the most traffic?" --verbose
```

### Example Questions

The system can answer analytical questions such as:
- "How has the number of sessions changed since 2024, by market or by device?"
- "Which pages generate the most traffic and which convert best into requests?"
- "Which agencies receive the most requests and how has their conversion rate changed since 2024?"
- "What is the median number of sessions per market?"
- "Show me the average conversion rate by device type"

## Configuration

Environment variables (see `.env.example`):
- `OLLAMA_HOST`: Ollama base URL (e.g., `http://localhost:11434`)
- `OLLAMA_SUPERVISOR_MODEL`: supervisor LLM (e.g., `ministral-3:3b`)
- `OLLAMA_CODER_MODEL`: coding LLM (e.g., `qwen2.5-coder:7b`)
- `DUCKDB_PATH`: path to DuckDB database file (or `:memory:`)
- `DATA_DIR`: directory for Excel files
- `AGENT_TEMPERATURE`: default temperature for agents

## Project layout

```
.
├── requirements.txt
├── README.md
├── .env.example
├── config.py
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── data_loader.py
│   ├── schema_manager.py
│   ├── db_manager.py
│   ├── agents.py
│   └── graph.py
└── data/
```

## Architecture

The system uses a multi-agent architecture orchestrated with LangGraph:

1. **Supervisor Agent** (ministral-3:3b): Analyzes questions and determines if SQL queries are needed
2. **Coding Agent** (qwen2.5-coder:7b): Generates SQL queries based on questions and schema
3. **DuckDB**: Executes SQL queries on Excel data loaded as tables
4. **LangGraph**: Orchestrates the workflow with conditional routing

### Workflow

```
User Question → Supervisor → [Needs Query?] → Coding Agent → Execute SQL → Synthesize Answer
                                    ↓
                                  [No] → Direct Answer
```

## Features

- **100% Local**: All processing happens locally with Ollama and DuckDB
- **Schema-Aware**: Agents receive schema information for accurate SQL generation
- **Analytical Capabilities**: Supports aggregations (COUNT, SUM, AVG, MIN, MAX, MEDIAN), grouping, date filtering
- **Error Handling**: Graceful error recovery and informative error messages
- **Extensible**: Easy to add more agents or data sources

## Data Format

- Place Excel files (`.xlsx`) in the `./data/` directory
- Each sheet in an Excel file becomes a separate DuckDB table
- Table names are auto-generated as `{workbook_name}_{sheet_name}` (lowercase, sanitized)
- The system automatically detects schema and makes it available to agents
