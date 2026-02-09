"""
Memory management module for LangGraph checkpointing.

This module provides persistent conversation memory using LangGraph's
checkpointer mechanism. It supports multi-user isolation through thread_id
namespacing and can be backed by PostgreSQL for production deployments.

For development/testing, an in-memory checkpointer can be used.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import os

from langgraph.checkpoint.memory import MemorySaver

from config import settings


@dataclass
class ChatThread:
    """
    Represents a conversation thread for a user.
    
    Attributes:
        thread_id: Unique identifier for this thread
        user_id: User who owns this thread
        created_at: Timestamp when thread was created
        title: Optional human-readable title (derived from first message)
    """
    thread_id: str
    user_id: str
    created_at: datetime = field(default_factory=datetime.now)
    title: Optional[str] = None


class MemoryManager:
    """
    Manages conversation memory with multi-user support.
    
    This class wraps LangGraph's checkpointer mechanism to provide:
    - Thread-scoped conversation history
    - User isolation via namespaced thread IDs
    - Persistent storage (PostgreSQL for production)
    
    Thread ID Format:
        {user_id}__{thread_id}
        
    This ensures that even if two users use the same thread name,
    their conversations are kept separate.
    """
    
    def __init__(self, use_postgres: bool = False) -> None:
        """
        Initialize the memory manager.
        
        Args:
            use_postgres: If True, use PostgreSQL backend (requires connection string).
                         If False, use in-memory storage (for development).
        """
        self.use_postgres = use_postgres
        self._checkpointer = None
        self._threads_cache: Dict[str, List[ChatThread]] = {}
        
        # Initialize the appropriate checkpointer
        if use_postgres:
            self._init_postgres_checkpointer()
        else:
            self._init_memory_checkpointer()
    
    def _init_memory_checkpointer(self) -> None:
        """Initialize in-memory checkpointer for development/testing."""
        self._checkpointer = MemorySaver()
    
    def _init_postgres_checkpointer(self) -> None:
        """
        Initialize PostgreSQL checkpointer for production.
        
        Requires POSTGRES_CONNECTION_STRING environment variable.
        Falls back to memory saver if not configured.
        """
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            import psycopg
            
            conn_string = os.getenv("POSTGRES_CONNECTION_STRING")
            if not conn_string:
                print("Warning: POSTGRES_CONNECTION_STRING not set. Using in-memory storage.")
                self._init_memory_checkpointer()
                return
            
            # Create connection pool for PostgreSQL
            self._pg_connection = psycopg.connect(conn_string)
            self._checkpointer = PostgresSaver(self._pg_connection)
            # Setup tables if they don't exist
            self._checkpointer.setup()
            
        except ImportError:
            print("Warning: langgraph-checkpoint-postgres not installed. Using in-memory storage.")
            self._init_memory_checkpointer()
        except Exception as e:
            print(f"Warning: PostgreSQL connection failed ({e}). Using in-memory storage.")
            self._init_memory_checkpointer()
    
    @property
    def checkpointer(self):
        """Get the underlying LangGraph checkpointer."""
        return self._checkpointer
    
    def make_thread_id(self, user_id: str, thread_id: str) -> str:
        """
        Create a namespaced thread ID for user isolation.
        
        Args:
            user_id: Unique user identifier
            thread_id: Thread name (can be reused across users)
            
        Returns:
            Namespaced thread ID in format: user_id__thread_id
        """
        return f"{user_id}__{thread_id}"
    
    def parse_thread_id(self, namespaced_id: str) -> tuple[str, str]:
        """
        Parse a namespaced thread ID back to components.
        
        Args:
            namespaced_id: Full thread ID in format user_id__thread_id
            
        Returns:
            Tuple of (user_id, thread_id)
        """
        parts = namespaced_id.split("__", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return "unknown", namespaced_id
    
    def get_config(self, user_id: str, thread_id: str) -> Dict[str, Any]:
        """
        Get LangGraph config dict for graph invocation.
        
        This config should be passed to graph.invoke() or graph.stream()
        to enable checkpointing for the specified thread.
        
        Args:
            user_id: User identifier
            thread_id: Thread identifier
            
        Returns:
            Config dict with thread_id in configurable
        """
        return {
            "configurable": {
                "thread_id": self.make_thread_id(user_id, thread_id)
            }
        }
    
    def register_thread(self, user_id: str, thread_id: str, title: Optional[str] = None) -> ChatThread:
        """
        Register a new thread for a user.
        
        Args:
            user_id: User identifier
            thread_id: Thread identifier
            title: Optional title for the thread
            
        Returns:
            ChatThread object
        """
        thread = ChatThread(
            thread_id=thread_id,
            user_id=user_id,
            title=title
        )
        
        if user_id not in self._threads_cache:
            self._threads_cache[user_id] = []
        
        # Avoid duplicates
        existing = [t for t in self._threads_cache[user_id] if t.thread_id == thread_id]
        if not existing:
            self._threads_cache[user_id].append(thread)
        
        return thread
    
    def list_user_threads(self, user_id: str) -> List[ChatThread]:
        """
        List all threads for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of ChatThread objects for this user
        """
        return self._threads_cache.get(user_id, [])
    
    def close(self) -> None:
        """Close any open connections."""
        if hasattr(self, '_pg_connection') and self._pg_connection:
            self._pg_connection.close()
