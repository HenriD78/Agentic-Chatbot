"""
Data loading module for Excel file processing.

This module handles reading Excel files from the data directory and loading
them into DuckDB tables. Each sheet in an Excel workbook becomes a separate
DuckDB table, allowing SQL queries to be executed across the data.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import duckdb
import pandas as pd

from config import settings
from src.schema_manager import ColumnInfo


def map_excel_type_to_duckdb(excel_type: str) -> str:
    """
    Map Excel/Metadata type strings to DuckDB types.
    
    Args:
        excel_type: Type string from the description sheet (e.g., 'STRING', 'TIMESTAMP')
        
    Returns:
        Corresponding DuckDB type string (e.g., 'VARCHAR', 'TIMESTAMP')
    """
    if not excel_type:
        return 'VARCHAR'
    
    t = excel_type.upper().strip()
    
    mapping = {
        'STRING': 'VARCHAR',
        'TEXT': 'VARCHAR',
        'STR': 'VARCHAR',
        'INTEGER': 'INTEGER',
        'INT': 'INTEGER',
        'FLOAT': 'DOUBLE',
        'DOUBLE': 'DOUBLE',
        'NUMERIC': 'DOUBLE',
        'BOOLEAN': 'BOOLEAN',
        'BOOL': 'BOOLEAN',
        'TIMESTAMP': 'TIMESTAMP',
        'DATETIME': 'TIMESTAMP',
        'DATE': 'DATE'
    }
    
    return mapping.get(t, 'VARCHAR')


def load_excel_files(con: duckdb.DuckDBPyConnection) -> Dict[str, List[ColumnInfo]]:
    """
    Load all Excel workbooks from DATA_DIR into DuckDB tables with strict type enforcement.
    
    This function:
    1. Scans the data directory for .xlsx files
    2. Identifies data sheets vs metadata sheets (ending in '_')
    3. Parses metadata first to understand the desired schema
    4. Loads data sheets, applying explicit type casting (TRY_CAST) based on metadata
    5. Returns a rich schema mapping
    
    Args:
        con: Active DuckDB connection object
        
    Returns:
        Dictionary mapping table names to their rich column definitions.
    """
    # Get the data directory path and ensure it exists
    data_path = Path(settings.data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Find all Excel files in the data directory
    excel_files = list(data_path.glob("*.xlsx"))
    
    # Early return if no Excel files found
    if not excel_files:
        print(f"Warning: No Excel files found in {data_path}")
        return {}

    # Store found tables to process later: (table_name, dataframe)
    data_tables: List[Tuple[str, pd.DataFrame]] = []
    
    # Store metadata dataframes: table_name -> dataframe
    metadata_dfs: Dict[str, pd.DataFrame] = {}

    # Step 1: Read all files and categorize sheets
    print(f"Scanning {len(excel_files)} Excel file(s)...")
    
    for workbook in excel_files:
        try:
            xl = pd.ExcelFile(workbook)
            
            for sheet in xl.sheet_names:
                try:
                    df = xl.parse(sheet)
                    
                    if df.empty:
                        print(f"Warning: Sheet '{sheet}' in '{workbook.name}' is empty, skipping")
                        continue
                    
                    # Sanitize table name
                    raw_name = f"{workbook.stem}_{sheet}".lower().replace(" ", "_").replace("-", "_")
                    table_name = "".join(c if c.isalnum() or c == "_" else "_" for c in raw_name)
                    
                    # Check if it's a metadata sheet (ends in _)
                    if table_name.endswith('_'):
                        metadata_dfs[table_name] = df
                        # Do NOT load into DuckDB
                    else:
                        data_tables.append((table_name, df))
                        
                except Exception as e:
                    print(f"Error loading sheet '{sheet}' from '{workbook.name}': {e}")
                    continue
                    
        except Exception as e:
            print(f"Error reading Excel file '{workbook.name}': {e}")
            continue

    # Step 2: Process data tables with type enforcement
    schema: Dict[str, List[ColumnInfo]] = {}
    
    print(f"Processing {len(data_tables)} data table(s)...")
    
    for table_name, df in data_tables:
        try:
            metadata_name = table_name + "_"
            col_metadata: Dict[str, Dict[str, Any]] = {}
            
            # Parse metadata if available
            if metadata_name in metadata_dfs:
                m_df = metadata_dfs[metadata_name]
                
                # Find column headers
                cols = {c.lower().strip(): c for c in m_df.columns}
                
            # Helper to find a column from candidates
                def get_col_key(candidates: List[str]) -> Optional[str]:
                    for cand in candidates:
                        if cand in cols:
                            return cols[cand]
                    return None
                
                name_col = get_col_key(['field name', 'fieldname', 'name', 'column', 'field'])
                type_col = get_col_key(['type', 'datatype', 'data type'])
                desc_col = get_col_key(['description', 'desc', 'definition'])
                
                if name_col:
                    for _, row in m_df.iterrows():
                        c_name_val = row[name_col]
                        if pd.isna(c_name_val):
                            continue
                        c_name = str(c_name_val).strip()
                        c_type = str(row[type_col]) if type_col and pd.notna(row[type_col]) else 'VARCHAR'
                        c_desc = str(row[desc_col]) if desc_col and pd.notna(row[desc_col]) else None
                        
                        col_metadata[c_name.lower()] = {
                            'type': c_type,
                            'duckdb_type': map_excel_type_to_duckdb(c_type),
                            'description': c_desc,
                            'original_name': c_name
                        }

            # Register raw dataframe as temp view
            tmp_view = f"tmp_{table_name}"
            con.register(view_name=tmp_view, python_object=df)
            
            # Build CREATE TABLE statement with casting
            select_parts = []
            table_columns: List[ColumnInfo] = []
            
            for col in df.columns.tolist():
                col_str = str(col)
                # Look up metadata (case insensitive)
                meta = col_metadata.get(col_str.lower())
                
                if meta:
                    duck_type = meta['duckdb_type']
                    original_type = meta['type']
                    desc = meta['description']
                else:
                    duck_type = 'VARCHAR' # Default fallback
                    original_type = 'STRING'
                    desc = None
                
                # Double quote column name to handle special chars/keywords
                safe_col = f'"{col_str}"'
                
                # CAST logic: TRY_CAST(col AS TYPE)
                select_parts.append(f"TRY_CAST({safe_col} AS {duck_type}) AS {safe_col}")
                
                table_columns.append({
                    'name': col_str,
                    'type': original_type,  # Keep original type for LLM prompt context
                    'description': desc
                })
            
            # Execute CREATE TABLE with explicit types
            query = f"CREATE OR REPLACE TABLE {table_name} AS SELECT {', '.join(select_parts)} FROM {tmp_view}"
            con.execute(query)
            con.unregister(tmp_view)
            
            schema[table_name] = table_columns
            print(f"Loaded table '{table_name}' with type enforcement ({len(df)} rows)")
            
        except Exception as e:
            print(f"Error creating table '{table_name}': {e}")
            continue
    
    return schema
