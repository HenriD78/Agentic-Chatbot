"""
Main entry point for the Agentic RAG system.

This module provides a CLI interface for interacting with the system.
Users can ask questions in natural language, and the system will:
1. Load Excel data into DuckDB
2. Route questions through the LangGraph workflow
3. Return synthesized answers

Usage:
    python -m src.main --question "Your question here"
    python -m src.main --question "Your question" --verbose
    python -m src.main --chat --user "alice" --thread "thread1"  # Interactive mode with memory
"""

import argparse
import sys
import uuid
from pathlib import Path

from config import settings
from src.data_loader import load_excel_files
from src.db_manager import DuckDBManager
from src.graph import AgenticGraph
from src.schema_manager import SchemaManager
from src.memory import MemoryManager


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Namespace object with parsed arguments:
        - question: User's question (optional for chat mode)
        - verbose: Whether to show detailed output (optional)
        - chat: Enable interactive chat mode with memory
        - user: User ID for memory isolation
        - thread: Thread ID for conversation continuity
        - history: Show conversation history
    """
    parser = argparse.ArgumentParser(
        description="Agentic RAG with LangGraph and Ollama - Query Excel data using natural language"
    )
    parser.add_argument(
        "--question",
        "-q",
        required=False,
        help="User question to route through the graph (not required in chat mode)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information including SQL queries",
    )
    parser.add_argument(
        "--database",
        "-d",
        action="store_true",
        help="Initialize database: scan /data folder for Excel files and load into DuckDB",
    )
    parser.add_argument(
        "--chat",
        "-c",
        action="store_true",
        help="Enable interactive chat mode with conversation memory",
    )
    parser.add_argument(
        "--user",
        "-u",
        default="default_user",
        help="User ID for memory isolation (default: default_user)",
    )
    parser.add_argument(
        "--thread",
        "-t",
        default=None,
        help="Thread ID for conversation continuity (default: auto-generated)",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show conversation history for the specified user/thread",
    )
    return parser.parse_args()


def initialize_database() -> tuple[SchemaManager, DuckDBManager]:
    """
    Initialize database by loading Excel files (slow path, used with --database flag).
    
    This function:
    1. Creates a DuckDB connection (overwrites existing DB)
    2. Loads all Excel files from the data directory
    3. Creates a schema manager with the loaded schemas
    4. Exports schema to JSON file for fast subsequent loads
    5. Returns both for use in the graph
    
    Returns:
        Tuple of (SchemaManager, DuckDBManager) ready for use
    """
    print("Initializing database (loading Excel files)...")
    
    # Step 1: Initialize DuckDB connection
    db = DuckDBManager()
    print(f"DuckDB initialized (path: {settings.duckdb_path})")
    
    # Step 2: Load Excel files into DuckDB tables
    print(f"Loading Excel files from {settings.data_dir}...")
    schemas = load_excel_files(db.con)
    
    # Step 3: Check if any tables were loaded
    if not schemas:
        print("Warning: No Excel files found or loaded. Please add Excel files to the data directory.")
        return SchemaManager(), db
    
    # Step 4: Display loaded tables
    print(f"Loaded {len(schemas)} table(s)")
    for table_name, columns in schemas.items():
        print(f"  - {table_name}: {len(columns)} columns")
    
    # Step 5: Initialize schema manager and export to file
    schema_manager = SchemaManager()
    schema_manager.update(schemas)
    schema_manager.save_to_file(settings.schema_path)
    
    return schema_manager, db


def initialize_system() -> tuple[SchemaManager, DuckDBManager]:
    """
    Initialize system using existing database and schema (fast path).
    
    This function:
    1. Loads schema from existing JSON file
    2. Connects to existing DuckDB database
    3. Returns both for use in the graph
    
    Returns:
        Tuple of (SchemaManager, DuckDBManager) ready for use
        
    Raises:
        FileNotFoundError: If schema.json or duckdb.db doesn't exist
    """
    from pathlib import Path
    
    print("Loading existing database (fast path)...")
    
    # Check if database file exists
    db_path = Path(settings.duckdb_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {settings.duckdb_path}. "
            f"Run with --database flag first to initialize the database."
        )
    
    # Check if schema file exists
    schema_path = Path(settings.schema_path)
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema file not found at {settings.schema_path}. "
            f"Run with --database flag first to initialize the database."
        )
    
    # Load schema from file
    schema_manager = SchemaManager()
    if not schema_manager.load_from_file(settings.schema_path):
        raise ValueError(f"Failed to load schema from {settings.schema_path}")
    
    # Connect to existing database
    db = DuckDBManager()
    print(f"Connected to DuckDB (path: {settings.duckdb_path})")
    
    return schema_manager, db


def display_result(result: dict, verbose: bool) -> None:
    """Display the result from graph execution."""
    if verbose:
        print("=" * 80)
        print("DETAILED RESULTS")
        print("=" * 80)
        print(f"\nSchema Info:\n{result.get('schema_info', 'N/A')}\n")
        
        # Show generated SQL query if available
        if result.get("query"):
            print(f"Generated SQL Query:\n{result['query']}\n")
        
        # Show query results if available
        if result.get("results") is not None:
            print(f"Query Results ({len(result['results'])} rows):")
            if result["results"]:
                # Show first 10 rows for readability
                for i, row in enumerate(result["results"][:10], 1):
                    print(f"  Row {i}: {row}")
                if len(result["results"]) > 10:
                    print(f"  ... ({len(result['results']) - 10} more rows)")
            else:
                print("  (No results)")
            print()
        
        # Show errors if any occurred
        if result.get("error"):
            print(f"Error: {result['error']}\n")
        print("=" * 80)
        print()
    
    # Display the final answer
    if result.get("answer"):
        print("Answer:")
        print("-" * 80)
        print(result["answer"])
        print("-" * 80)
    elif result.get("error"):
        # If there's an error but no answer, show the error
        print(f"Error: {result['error']}")


def run_interactive_chat(args: argparse.Namespace, schema_manager: SchemaManager, db: DuckDBManager) -> None:
    """
    Run interactive chat mode with conversation memory.
    
    Args:
        args: Parsed command-line arguments
        schema_manager: SchemaManager instance
        db: DuckDBManager instance
    """
    # Initialize memory manager
    memory = MemoryManager(use_postgres=settings.use_postgres_memory)
    
    # Generate thread ID if not provided
    thread_id = args.thread or str(uuid.uuid4())[:8]
    user_id = args.user
    
    print(f"\n🎯 Interactive Chat Mode")
    print(f"   User: {user_id}")
    print(f"   Thread: {thread_id}")
    print(f"   Memory: {'PostgreSQL' if settings.use_postgres_memory else 'In-Memory'}")
    print("\n   Type 'exit' or 'quit' to end the conversation.")
    print("   Type 'history' to see conversation history.")
    print("-" * 80)
    
    # Register this thread
    memory.register_thread(user_id, thread_id)
    
    # Get config for checkpointing
    config = memory.get_config(user_id, thread_id)
    
    # Create the graph with checkpointer
    graph = AgenticGraph(
        schema_manager=schema_manager,
        db=db,
        checkpointer=memory.checkpointer
    )
    
    while True:
        try:
            # Get user input
            question = input("\n📝 You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ("exit", "quit"):
                print("\n👋 Goodbye!")
                break
            
            if question.lower() == "history":
                # Show conversation history from state
                threads = memory.list_user_threads(user_id)
                print(f"\n📚 Your threads:")
                for t in threads:
                    print(f"   - {t.thread_id} (created: {t.created_at})")
                continue
            
            # Process the question with memory
            print("\n🤔 Processing...\n")
            result = graph.run(question, config)
            
            # Display the answer
            if result.get("answer"):
                print(f"🤖 Assistant: {result['answer']}")
            elif result.get("error"):
                print(f"❌ Error: {result['error']}")
            else:
                print("❌ No answer generated.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Cleanup
    memory.close()


def main() -> None:
    """
    Main function: Entry point for the CLI application.
    
    This function:
    1. Parses command-line arguments
    2. Initializes the system (loads data, creates managers)
    3. Creates the LangGraph workflow
    4. Processes the user's question
    5. Displays the answer
    
    The workflow handles all agent coordination and query execution.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Validate arguments
    if not args.chat and not args.question:
        print("Error: Either --question or --chat is required.", file=sys.stderr)
        print("Use --chat for interactive mode or --question for single query mode.", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Step 1: Initialize system components
        # Use --database flag to determine initialization path
        if args.database:
            schema_manager, db = initialize_database()
        else:
            schema_manager, db = initialize_system()
        
        # Step 2: Handle interactive chat mode
        if args.chat:
            run_interactive_chat(args, schema_manager, db)
            return
        
        # Step 3: Single question mode (original behavior)
        print("\nInitializing agents and graph...")
        graph = AgenticGraph(schema_manager=schema_manager, db=db)
        print("System ready!\n")
        
        # Step 4: Process the user's question
        print(f"Question: {args.question}\n")
        print("Processing...\n")
        
        # Run the graph workflow with the question
        result = graph.run(args.question)
        
        # Step 5: Display results
        display_result(result, args.verbose)
    
    except KeyboardInterrupt:
        # Handle user interruption (Ctrl+C)
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        # Handle any unexpected errors
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            # Show full traceback in verbose mode
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup: Close database connection
        if 'db' in locals():
            db.close()


if __name__ == "__main__":
    # Entry point when running as a script
    main()
