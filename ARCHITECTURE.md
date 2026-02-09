# Architecture Documentation

## Overview

This document describes the architecture of the Agentic RAG system, explaining how all components work together from the moment Excel files are uploaded to when questions are answered.

## System Architecture

The system uses a multi-agent architecture orchestrated by LangGraph:

```
┌─────────────┐
│ Excel Files │
│  (./data/)  │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  Data Loader     │ ──► Converts Excel sheets to DuckDB tables
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Schema Manager   │ ──► Tracks table schemas for agents
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  DuckDB Manager  │ ──► Executes SQL queries on loaded data
└──────┬───────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│         LangGraph Workflow              │
│  ┌──────────────────────────────────┐  │
│  │  1. Supervisor Agent              │  │ ──► Analyzes question
│  │     (ministral-3:3b)              │  │
│  └───────────┬──────────────────────┘  │
│              │                           │
│              ▼                           │
│  ┌──────────────────────────────────┐  │
│  │  2. Coding Agent                 │  │ ──► Generates SQL
│  │     (qwen2.5-coder:7b)           │  │
│  └───────────┬──────────────────────┘  │
│              │                           │
│              ▼                           │
│  ┌──────────────────────────────────┐  │
│  │  3. Execute Node                 │  │ ──► Runs SQL query
│  └───────────┬──────────────────────┘  │
│              │                           │
│              ▼                           │
│  ┌──────────────────────────────────┐  │
│  │  4. Synthesize Node              │  │ ──► Creates answer
│  │     (Supervisor Agent)           │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│   Answer    │
└─────────────┘
```

## Component Overview

### 1. Configuration (`config.py`)

**Purpose**: Centralized configuration management

**Key Components**:
- `Settings` dataclass: Holds all configuration parameters
- Loads from environment variables (via `.env` file)
- Provides defaults for local development

**Configuration Parameters**:
- `ollama_host`: Ollama API endpoint
- `supervisor_model`: Model name for supervisor agent
- `coder_model`: Model name for coding agent
- `agent_temperature`: LLM sampling temperature
- `data_dir`: Directory containing Excel files
- `duckdb_path`: DuckDB database path

### 2. Data Loading (`src/data_loader.py`)

**Purpose**: Load Excel files into DuckDB tables

**Key Function**:
- `load_excel_files(con)`: Main function that processes Excel files

**Process Flow**:
1. Scans `./data/` directory for `.xlsx` files
2. For each Excel file:
   - Opens the workbook
   - Iterates through all sheets
   - Parses each sheet into a pandas DataFrame
   - Creates a DuckDB table from the DataFrame
   - Sanitizes table names (lowercase, underscores)
3. Returns schema dictionary mapping table names to column lists

**Table Naming**: `{workbook_name}_{sheet_name}` (e.g., `sales_data_q1`)

### 3. Schema Management (`src/schema_manager.py`)

**Purpose**: Track and provide schema information to agents

**Key Class**: `SchemaManager`

**Methods**:
- `update(schemas)`: Update registry with new schemas
- `describe()`: Generate human-readable schema summary for agents
- `get()`: Get full schema dictionary
- `get_table_names()`: List all table names
- `get_columns(table_name)`: Get columns for a specific table

**Schema Format**: `"table1(col1, col2); table2(col1, col2)"`

### 4. Database Management (`src/db_manager.py`)

**Purpose**: Execute SQL queries safely on DuckDB

**Key Class**: `DuckDBManager`

**Methods**:
- `__init__(database_path)`: Initialize connection and load extensions
- `validate_query(sql)`: Ensure query is a SELECT statement (safety)
- `query(sql)`: Execute SQL and return results as list of dicts
- `get_table_info(table_name)`: Get table metadata
- `list_tables()`: List all tables in database
- `close()`: Close database connection

**Safety Features**:
- Only allows SELECT/WITH statements
- Validates queries before execution
- Comprehensive error handling

### 5. Agents (`src/agents.py`)

**Purpose**: Define LLM-powered agents for question analysis and SQL generation

#### SupervisorAgent

**Model**: ministral-3:3b

**Responsibilities**:
1. Analyze questions to determine if SQL query is needed
2. Synthesize natural language answers from query results

**Key Methods**:
- `analyze_question(question, schema_info)`: Determines if query needed
- `synthesize_answer(question, query, results, schema_info)`: Creates final answer

#### CodingAgent

**Model**: qwen2.5-coder:7b

**Responsibilities**:
1. Generate SQL queries from natural language questions
2. Use schema information to create accurate queries

**Key Methods**:
- `generate_query(question, schema_info)`: Translates question to SQL

### 6. Graph Orchestration (`src/graph.py`)

**Purpose**: Coordinate agents using LangGraph state machine

**Key Class**: `AgenticGraph`

**State Schema** (`GraphState`):
- `question`: User's question
- `schema_info`: Schema description
- `query`: Generated SQL query
- `results`: Query execution results
- `answer`: Final synthesized answer
- `error`: Error message (if any)
- `needs_query`: Boolean flag for routing

**Graph Nodes**:
1. **supervisor_node**: Analyzes question, sets `needs_query` flag
2. **coding_node**: Generates SQL query
3. **execute_node**: Executes query in DuckDB
4. **synthesize_node**: Creates final answer

**Graph Flow**:
```
supervisor → [conditional] → coding → execute → synthesize → END
              ↓ (if no query)
            END
```

**Key Methods**:
- `_build_graph()`: Constructs the LangGraph workflow
- `run(question)`: Main entry point to process a question

### 7. Main Application (`src/main.py`)

**Purpose**: CLI interface and system initialization

**Key Functions**:
- `parse_args()`: Parse command-line arguments
- `initialize_system()`: Load data and create managers
- `main()`: Main entry point

## Complete Execution Flow

### Phase 1: System Initialization

**File**: `src/main.py` → `initialize_system()`

1. **Create DuckDB Connection**
   - `DuckDBManager()` is instantiated
   - Database connection is established (in-memory or file-based)
   - Median extension is installed for statistical functions

2. **Load Excel Files**
   - `load_excel_files(db.con)` is called
   - Scans `./data/` directory for `.xlsx` files
   - For each file:
     - Opens with `pandas.ExcelFile()`
     - Parses each sheet into DataFrame
     - Registers DataFrame as temporary DuckDB view
     - Creates permanent table from view
     - Unregisters temporary view
   - Returns schema dictionary

3. **Initialize Schema Manager**
   - `SchemaManager()` is created
   - `schema_manager.update(schemas)` populates the registry

### Phase 2: Graph Initialization

**File**: `src/main.py` → `AgenticGraph()`

1. **Create Agents**
   - `SupervisorAgent(supervisor_cfg)` is instantiated
   - `CodingAgent(coding_cfg)` is instantiated
   - Both connect to Ollama via LangChain

2. **Build Graph**
   - `_build_graph()` creates the LangGraph workflow
   - Adds all nodes (supervisor, coding, execute, synthesize)
   - Sets entry point (supervisor)
   - Adds conditional edge from supervisor
   - Adds linear edges for query path
   - Compiles the graph

### Phase 3: Question Processing

**File**: `src/main.py` → `graph.run(question)`

1. **Create Initial State**
   - Gets schema summary from `schema_manager.describe()`
   - Creates `GraphState` with question and schema info

2. **Supervisor Node** (`_supervisor_node`)
   - Calls `supervisor.analyze_question(question, schema_info)`
   - LLM analyzes question and determines if query needed
   - Sets `needs_query` flag in state
   - If no query needed, sets `answer` directly

3. **Conditional Routing** (`_should_query`)
   - If `needs_query == True` → route to coding node
   - If `needs_query == False` → route to END

4. **Coding Node** (`_coding_node`) - *if query needed*
   - Calls `coder.generate_query(question, schema_info)`
   - LLM generates SQL query from question
   - Stores query in state

5. **Execute Node** (`_execute_node`)
   - Calls `db.query(state["query"])`
   - Validates query is SELECT statement
   - Executes query in DuckDB
   - Converts results to list of dictionaries
   - Stores results in state

6. **Synthesize Node** (`_synthesize_node`)
   - Calls `supervisor.synthesize_answer(question, query, results, schema_info)`
   - LLM creates natural language answer from results
   - Stores answer in state

7. **Return Final State**
   - Graph returns final state with answer (or error)

### Phase 4: Display Results

**File**: `src/main.py` → `main()`

1. **Verbose Mode** (if `--verbose` flag)
   - Displays schema information
   - Shows generated SQL query
   - Shows query results (first 10 rows)
   - Shows any errors

2. **Display Answer**
   - Prints the synthesized answer
   - Handles errors gracefully

3. **Cleanup**
   - Closes database connection

## File Execution Guide

### Which File to Run

**Main Entry Point**: `src/main.py`

**Command**:
```bash
python -m src.main --question "Your question here"
```

**With Verbose Output**:
```bash
python -m src.main --question "Your question" --verbose
```

### Execution Order

When you run `python -m src.main`, here's what happens:

1. **`main()` function** is called
2. **`parse_args()`** parses command-line arguments
3. **`initialize_system()`** is called:
   - Creates `DuckDBManager()`
   - Calls `load_excel_files(db.con)` from `src/data_loader.py`
   - Creates `SchemaManager()` and updates with schemas
4. **`AgenticGraph()`** is instantiated:
   - Creates agents from `src/agents.py`
   - Builds graph from `src/graph.py`
5. **`graph.run(question)`** processes the question:
   - Routes through LangGraph nodes
   - Agents use Ollama LLMs
   - Database executes queries
6. **Results are displayed** and database is closed

## Data Flow Diagram

```
User Question
    │
    ▼
┌─────────────────┐
│  main.py        │
│  - parse_args() │
│  - initialize() │
└────────┬────────┘
         │
         ├──► data_loader.py ──► Excel Files ──► DuckDB Tables
         │
         ├──► schema_manager.py ──► Schema Registry
         │
         └──► graph.py ──► AgenticGraph
                │
                ├──► agents.py ──► SupervisorAgent ──► Ollama (ministral-3:3b)
                │                  CodingAgent ──► Ollama (qwen2.5-coder:7b)
                │
                └──► db_manager.py ──► DuckDB ──► Query Results
                         │
                         └──► Final Answer
```

## Key Design Decisions

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **State Management**: LangGraph TypedDict ensures type safety and clear data flow
3. **Error Handling**: Errors are caught at each node and propagated through state
4. **Safety**: Query validation prevents destructive operations
5. **Extensibility**: Easy to add new agents or modify the workflow
6. **Local Execution**: 100% local - no external API dependencies (except Ollama)

## Dependencies

- **LangGraph**: Workflow orchestration
- **LangChain**: LLM integration (Ollama)
- **DuckDB**: In-memory SQL database
- **Pandas**: Excel file reading
- **OpenPyXL**: Excel file parsing

## Configuration Files

- **`.env`**: Environment variables (copy from `.env.example`)
- **`config.py`**: Centralized settings management
- **`requirements.txt`**: Python dependencies

## Directory Structure

```
.
├── config.py              # Configuration management
├── src/
│   ├── __init__.py        # Package marker
│   ├── main.py            # CLI entry point ⭐ RUN THIS FILE
│   ├── data_loader.py      # Excel → DuckDB conversion
│   ├── schema_manager.py   # Schema tracking
│   ├── db_manager.py       # Database operations
│   ├── agents.py           # LLM agent definitions
│   └── graph.py            # LangGraph workflow
├── data/                   # Excel files go here
├── .env                    # Environment variables
└── requirements.txt        # Dependencies
```

## Summary

The system follows a clear pipeline:
1. **Data Loading**: Excel files → DuckDB tables
2. **Schema Tracking**: Table structures → Agent context
3. **Question Processing**: Natural language → SQL → Results → Answer
4. **Agent Coordination**: LangGraph orchestrates supervisor and coding agents
5. **Answer Synthesis**: Query results → Natural language response

All components work together seamlessly, with each module handling a specific aspect of the system. The LangGraph workflow ensures proper sequencing and error handling throughout the process.
