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
   ollama pull ministral3:8b
   ollama pull qwen2.5-coder:7b
   ```
3. **Start Ollama** (if not running as a service):
   ```bash
   ollama serve
   ```

### Usage

**1. Initialize the Database (Fast Start)**  
Run this once (or whenever you add new Excel files) to parse data and build the persistent DuckDB database:
```bash
python -m src.main --database
```

**2. Ask a Question (One-off)**  
Uses the pre-built database for faster response:
```bash
python -m src.main --question "How has the number of sessions changed since 2024, by market?"
```

**3. Interactive Chat Mode**  
Start a conversation with memory support:
```bash
python -m src.main --chat
```

**Options**:
- `--database` / `-d`: Re-scan Excel files and rebuild the database
- `--verbose` / `-v`: Show SQL queries and debugging info
- `--chat` / `-c`: Enter interactive chat mode
- `--user [ID]`: Specify user ID for memory isolation (default: "default_user")
- `--thread [ID]`: Specify thread ID to resume a conversation

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

## Configuration

Environment variables (see `.env.example`):
- `OLLAMA_HOST`: Ollama base URL (e.g., `http://localhost:11434`)
- `OLLAMA_SUPERVISOR_MODEL`: supervisor LLM (default: `ministral3:8b`)
- `OLLAMA_CODER_MODEL`: coding LLM (default: `qwen2.5-coder:7b`)
- `DUCKDB_PATH`: path to persistent DuckDB database file (default: `./duckdb.db`)
- `DATA_DIR`: directory for Excel files
- `AGENT_TEMPERATURE`: default temperature for agents (default: 0.1)

### Memory Configuration
The system supports persistent conversation memory. By default, it uses in-memory storage (lost on restart). To enable PostgreSQL persistence:
1. Set `USE_POSTGRES_MEMORY=true`
2. Set `POSTGRES_CONNECTION_STRING=postgresql://user:password@localhost:5432/dbname`
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

1. **Supervisor Agent** (ministral3:8b): Analyzes questions, decomposes complex queries, and synthesizes answers
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
