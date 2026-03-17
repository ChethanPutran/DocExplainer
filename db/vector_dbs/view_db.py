import sqlite3
import pandas as pd

# Update this path to the actual location of your file
db_path = 'chroma.sqlite3' 

conn = sqlite3.connect(db_path)

# Check all tables, including internal system tables
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)

if tables.empty:
    print("The database is empty. No tables found.")
    print("Tip: Make sure you called 'vector_db.persist()' or allowed the script to finish saving.")
else:
    print("--- Tables Found ---")
    print(tables)
    
    # In recent Chroma versions, look for these specific tables:
    for target in ['segments', 'embeddings', 'collections', 'embedding_fulltext']:
        if target in tables['name'].values:
            print(f"\n--- Top 5 rows of {target} ---")
            print(pd.read_sql_query(f"SELECT * FROM {target} LIMIT 5;", conn))

conn.close()