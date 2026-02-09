"""
Configuration module for the Agentic RAG system.

This module loads environment variables from a .env file and provides a centralized
Settings dataclass that contains all configuration parameters needed by the system.
Configuration includes Ollama connection details, model names, and data paths.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


@dataclass
class Settings:
    """
    Centralized configuration settings for the Agentic RAG system.
    
    All settings can be overridden via environment variables. Default values
    are provided for local development.
    
    Attributes:
        ollama_host: Base URL for Ollama API (default: http://localhost:11434)
        supervisor_model: Name of the Ollama model for the supervisor agent
        coder_model: Name of the Ollama model for the coding agent
        agent_temperature: Temperature setting for LLM generation (0.0-1.0)
        data_dir: Directory path where Excel files are stored
        duckdb_path: Path to DuckDB database file (use :memory: for in-memory DB)
    """
    # Ollama connection settings
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    # Model names (must be pulled locally via `ollama pull <model>`)
    supervisor_model: str = os.getenv("OLLAMA_SUPERVISOR_MODEL", "ministral3:8b")
    coder_model: str = os.getenv("OLLAMA_CODER_MODEL", "gpt-oss:120b-cloud")
    
    # Agent configuration
    agent_temperature: float = float(os.getenv("AGENT_TEMPERATURE", "0.1"))
    
    # Data and database configuration
    data_dir: str = os.getenv("DATA_DIR", "./data")
    duckdb_path: str = os.getenv("DUCKDB_PATH", "./duckdb.db")
    schema_path: str = os.getenv("SCHEMA_PATH", "./schema.json")
    
    # Memory/checkpoint configuration
    memory_db_path: str = os.getenv("MEMORY_DB_PATH", "./memory.db")
    use_postgres_memory: bool = os.getenv("USE_POSTGRES_MEMORY", "false").lower() == "true"


# Global settings instance - imported by other modules
settings = Settings()
