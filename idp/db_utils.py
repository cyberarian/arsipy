# d:\pandasai\idp\db_utils.py
import streamlit as st
import sqlite3
import json # For handling list serialization/deserialization
import pandas as pd
from datetime import datetime
import time

# --- Constants ---
DB_PATH = "idp_records.db"

# --- Database Initialization ---
def init_db():
    """Initializes the SQLite database and creates the records table if it doesn't exist."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Create table with an ID, filename, timestamp, provider, notes, and keywords (as JSON text)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,    -- User-facing (editable) filename
                base_filename TEXT UNIQUE, -- Internal unique ID linking to processed files (NOT editable)
                processed_at REAL NOT NULL, -- Store as Unix timestamp (REAL)
                provider TEXT,             -- Method used for processing (OCR, Direct, etc.)
                notes TEXT,                -- User-editable notes
                keywords TEXT,             -- Store keywords as JSON string
                chunk_size INTEGER,        -- Chunk size used during processing
                chunk_overlap INTEGER      -- Chunk overlap used during processing
            )
        ''')
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        st.error(f"Database Error (init_db): {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during DB initialization: {e}")

# --- Create Function ---
def insert_record(record_data: dict):
    """Inserts a new record into the database."""
    # Add base_filename, chunk_size, chunk_overlap to required keys
    required_keys = ["filename", "base_filename", "processed_at", "provider", "chunk_size", "chunk_overlap"]
    if not all(key in record_data and record_data[key] is not None for key in required_keys):
        st.error(f"Database Error (insert_record): Missing required keys ({required_keys}).")
        return False

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Convert keywords list to JSON string for storage
        keywords_json = json.dumps(record_data.get("keywords", []))
        cursor.execute(
            """INSERT INTO records
               (filename, base_filename, processed_at, provider, notes, keywords, chunk_size, chunk_overlap)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            # Add None for notes initially
            (
                record_data.get("filename"),
                record_data.get("base_filename"),
                record_data.get("processed_at", time.time()), # Use provided or current time
                record_data.get("provider"),
                None, # Initialize notes as None
                keywords_json,
                record_data.get("chunk_size"),
                record_data.get("chunk_overlap")
            )
        )
        conn.commit()
        conn.close()
        st.toast(f"Record for '{record_data.get('filename')}' added to database.")
        return True # Indicate success
    except sqlite3.Error as e:
        # Handle potential UNIQUE constraint violation for base_filename gracefully
        if "UNIQUE constraint failed: records.base_filename" in str(e):
             st.error(f"Database Error: A record with the internal ID '{record_data.get('base_filename')}' already exists. Cannot insert duplicate.")
        else:
            st.error(f"Database Error (insert_record): {e}")
        return False # Indicate failure
    except Exception as e:
        st.error(f"An unexpected error occurred during record insertion: {e}")
        return False # Indicate failure

# --- Read Function ---
def load_records_to_dataframe() -> pd.DataFrame:
    """Loads all records from the database into a Pandas DataFrame."""
    try:
        # Select all columns including the new ones
        conn = sqlite3.connect(DB_PATH)
        query = """SELECT id, filename, base_filename, processed_at, provider, notes, keywords, chunk_size, chunk_overlap
                   FROM records ORDER BY filename ASC""" # Order alphabetically by display name
        df = pd.read_sql_query(query, conn)
        conn.close()

        # --- Post-processing ---
        # Convert Unix timestamp (REAL) to datetime objects
        # Use errors='coerce' to handle potential invalid timestamp values gracefully
        df['processed_at'] = pd.to_datetime(df['processed_at'], unit='s', errors='coerce')
        # Convert keywords JSON string back to list (handle potential errors)
        def safe_json_loads(x):
            try:
                # Return empty list if x is None or empty string
                if pd.isna(x) or not x:
                    return []
                loaded = json.loads(x)
                # Ensure it's a list
                return loaded if isinstance(loaded, list) else []
            except (json.JSONDecodeError, TypeError):
                return [] # Return empty list on error

        df['keywords'] = df['keywords'].apply(safe_json_loads)

        # Ensure 'notes' column exists and fill NaNs with empty string for display
        if 'notes' not in df.columns:
            df['notes'] = ''
        df['notes'] = df['notes'].fillna('')

        # Ensure chunk settings columns exist and fill NaNs with a default or indicator
        if 'chunk_size' not in df.columns:
            df['chunk_size'] = 0 # Or pd.NA
        df['chunk_size'] = df['chunk_size'].fillna(0) # Or pd.NA

        if 'chunk_overlap' not in df.columns:
            df['chunk_overlap'] = 0 # Or pd.NA
        df['chunk_overlap'] = df['chunk_overlap'].fillna(0) # Or pd.NA


        # --- DEBUG PRINT ---
        print(f"[DEBUG] Fetched {len(df)} records from the database.") # Add this line

        return df

    except sqlite3.Error as e:
        st.error(f"Database Error (load_records): {e}")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        st.error(f"An unexpected error occurred loading records: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

# --- Delete Function ---
def delete_record(record_id: int):
    """Deletes a record from the database by its ID."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM records WHERE id = ?", (record_id,))
        conn.commit()
        conn.close()
        st.toast(f"Record ID {record_id} deleted from database.")
        return True # Indicate success
    except sqlite3.Error as e:
        st.error(f"Database Error (delete_record): {e}")
        return False # Indicate failure
    except Exception as e:
        st.error(f"An unexpected error occurred during record deletion: {e}")
        return False # Indicate failure

# --- Update Function ---
def update_record(record_id: int, updates: dict):
    """Updates specific fields of a record in the database by its ID."""
    if not updates:
        st.warning("No updates provided for record ID {record_id}.")
        return False

    # Prepare SQL update statement dynamically
    set_clauses = []
    values = []
    for key, value in updates.items():
        # Ensure we only try to update valid columns (add more as needed)
        # base_filename, chunk_size, chunk_overlap are NOT updatable here
        if key in ["filename", "provider", "notes", "keywords"]:
            set_clauses.append(f"{key} = ?")
            # Serialize keywords list to JSON if updating that column
            values.append(json.dumps(value) if key == "keywords" else value)

    if not set_clauses:
        st.warning(f"No valid fields to update for record ID {record_id}.")
        return False

    sql = f"UPDATE records SET {', '.join(set_clauses)} WHERE id = ?"
    values.append(record_id) # Add the ID for the WHERE clause

    try:
        # --- DEBUG PRINT ---
        print(f"[DEBUG update_record] Updating ID: {record_id} with SQL: '{sql}' and VALUES: {tuple(values)}")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql, tuple(values))
        conn.commit()
        conn.close()
        # Don't toast on every update, let the calling function handle feedback
        return True # Indicate success
    except sqlite3.Error as e:
        st.error(f"Database Error (update_record ID {record_id}): {e}")
        return False # Indicate failure
    except Exception as e:
        st.error(f"An unexpected error occurred during record update (ID {record_id}): {e}")
        return False # Indicate failure
