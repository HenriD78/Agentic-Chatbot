"""
Schema management module for tracking database table structures.

This module provides a lightweight registry that maintains information about
all tables and their columns. This schema information is crucial for agents
to generate accurate SQL queries, as they need to know what tables and columns
are available in the database.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, TypedDict, Optional, Set, Counter, Tuple


class ColumnInfo(TypedDict):
    """Type definition for column metadata."""
    name: str
    type: Optional[str]
    description: Optional[str]


class SchemaManager:
    """
    Lightweight registry of table schemas for agent context.
    
    The SchemaManager maintains a dictionary mapping table names to their
    column lists with rich metadata. It provides methods to:
    - Update schemas when new tables are loaded
    - Generate human-readable schema descriptions for agent prompts
    - Query schema information for specific tables
    """

    def __init__(self) -> None:
        """
        Initialize an empty schema registry.
        
        The internal _schemas dictionary will be populated when Excel files
        are loaded via the update() method.
        """
        # Mapping: table_name -> List[ColumnInfo]
        self._schemas: Dict[str, List[ColumnInfo]] = {}
        # Set of column names that appear in multiple tables
        self._shared_columns: Set[str] = set()

    def update(self, schemas: Dict[str, List[ColumnInfo]]) -> None:
        """
        Update the schema registry with new table schemas.
        
        Args:
            schemas: Dictionary mapping table names to lists of column info objects
        """
        self._schemas.update(schemas)
        self._detect_relationships()

    def _detect_relationships(self) -> None:
        """
        Analyze schemas to identify columns shared across multiple tables.
        
        This method builds a frequency map of all column names and identifies
        those that appear in more than one table. These shared columns are
        likely foreign keys or join keys.
        """
        if not self._schemas:
            self._shared_columns = set()
            return

        # Count occurrences of each column name
        column_counts = Counter()
        for columns in self._schemas.values():
            for col in columns:
                column_counts[col['name']] += 1
        
        # Identify columns that appear in > 1 table
        self._shared_columns = {
            col for col, count in column_counts.items() 
            if count > 1
        }

    def _build_relationship_map(self) -> str:
        """
        Generate a text description of relationships between tables.
        
        This method iterates through all pairs of tables and finds shared columns,
        creating a clear "map" for the agent to understand join possibilities.
        """
        if not self._schemas or len(self._schemas) < 2:
            return ""

        table_names = sorted(list(self._schemas.keys()))
        relationships: List[str] = []
        
        # Iterate over unique pairs of tables
        for i in range(len(table_names)):
            for j in range(i + 1, len(table_names)):
                table_a = table_names[i]
                table_b = table_names[j]
                
                # Get column sets for both tables
                cols_a = {c['name'] for c in self._schemas[table_a]}
                cols_b = {c['name'] for c in self._schemas[table_b]}
                
                # Find intersection
                common = cols_a.intersection(cols_b)
                
                # Filter out common false positives if necessary (e.g. 'is_test')
                # For now we keep it simple and show all shared keys
                valid_links = sorted(list(common))
                
                if valid_links:
                    links_str = ", ".join(valid_links)
                    relationships.append(f"- {table_a} and {table_b} are linked by: {links_str}")

        if not relationships:
            return ""

        return "\n\nDetected Relationships:\n" + "\n".join(relationships)

    def describe(self) -> str:
        """
        Returns a human-readable schema summary for prompting agents.
        
        This method formats the schema information in a way that's easy
        for LLMs to understand when generating SQL queries. The format is:
        "table_name (
            col_name (TYPE) [SHARED KEY]: Description,
            ...
        ); ...
        
        Detected Relationships:
        - table1 and table2 are linked by: id"
        
        Returns:
            String representation of all tables and their columns.
            
        Note:
            Returns "No tables loaded." if the registry is empty.
        """
        if not self._schemas:
            return "No tables loaded."
        
        # Build a list of formatted table descriptions
        parts = []
        for table, columns in sorted(self._schemas.items()):
            # Start table definition
            table_desc = [f"{table} ("]
            
            # Add each column with its metadata
            col_descs = []
            for col in columns:
                name = col['name']
                # Default to 'UNKNOWN' if type is missing, making it explicit
                dtype = col.get('type', 'UNKNOWN')
                desc = col.get('description', '')
                
                # Format: name (TYPE) [SHARED KEY]?: description
                # Check for shared key status
                is_shared = name in self._shared_columns
                shared_tag = " [SHARED KEY]" if is_shared else ""
                
                col_str = f"{name} ({dtype}){shared_tag}"
                if desc:
                    col_str += f": {desc}"
                col_descs.append(col_str)
            
            # Join columns with commas and indentation
            table_desc.append(", ".join(col_descs))
            table_desc.append(")")
            
            # Join parts to form "table (col1...)"
            parts.append("".join(table_desc))
        
        # Join all table descriptions with semicolons
        schema_str = "; ".join(parts)
        
        # Append relationship map
        rel_map = self._build_relationship_map()
        
        return schema_str + rel_map

    def get(self) -> Dict[str, List[ColumnInfo]]:
        """
        Get the full schema dictionary.
        
        Returns:
            Copy of the internal schema dictionary mapping table names to lists of ColumnInfo
        """
        return dict(self._schemas)
    
    def get_table_names(self) -> List[str]:
        """
        Get list of all table names in the registry.
        
        Returns:
            List of table names (keys from the schema dictionary)
        """
        return list(self._schemas.keys())
    
    def get_columns(self, table_name: str) -> List[ColumnInfo]:
        """
        Get columns for a specific table.
        
        Args:
            table_name: Name of the table to query
            
        Returns:
            List of column info objects for the specified table, or empty list if not found
        """
        return self._schemas.get(table_name, [])

    def save_to_file(self, path: str) -> None:
        """
        Export schema with relationships to a JSON file.
        
        The exported file contains:
        - tables: Full schema with column info (name, type, description)
        - relationships: List of table pairs with their shared columns
        - shared_columns: List of all columns appearing in multiple tables
        - generated_at: Timestamp of file generation
        
        Args:
            path: Path to the output JSON file
        """
        # Build relationship list from current schemas
        relationships = []
        table_names = sorted(list(self._schemas.keys()))
        
        for i in range(len(table_names)):
            for j in range(i + 1, len(table_names)):
                table_a = table_names[i]
                table_b = table_names[j]
                
                cols_a = {c['name'] for c in self._schemas[table_a]}
                cols_b = {c['name'] for c in self._schemas[table_b]}
                common = sorted(list(cols_a.intersection(cols_b)))
                
                if common:
                    relationships.append({
                        "table_a": table_a,
                        "table_b": table_b,
                        "shared_columns": common
                    })
        
        # Build output structure
        output = {
            "tables": self._schemas,
            "relationships": relationships,
            "shared_columns": sorted(list(self._shared_columns)),
            "generated_at": datetime.now().isoformat()
        }
        
        # Write to file
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Schema exported to {path}")

    def load_from_file(self, path: str) -> bool:
        """
        Load schema from a JSON file (fast path for query-only mode).
        
        Args:
            path: Path to the schema JSON file
            
        Returns:
            True if loaded successfully, False if file doesn't exist or is invalid
        """
        path_obj = Path(path)
        
        if not path_obj.exists():
            return False
        
        try:
            with open(path_obj, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Restore schemas
            self._schemas = data.get("tables", {})
            
            # Restore shared columns
            self._shared_columns = set(data.get("shared_columns", []))
            
            print(f"Schema loaded from {path} (generated at: {data.get('generated_at', 'unknown')})")
            return True
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load schema from {path}: {e}")
            return False

