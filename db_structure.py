import sqlite3

conn = sqlite3.connect('Stock Data.db')
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

structure = ""
for table_name in tables:
    structure += f"Table: {table_name[0]}\n"
    cursor.execute(f"PRAGMA table_info('{table_name[0]}');")
    columns = cursor.fetchall()
    for col in columns:
        structure += f" - {col[1]} ({col[2]})\n"
    structure += "\n"

with open('db_structure.txt', 'w') as f:
    f.write(structure)

conn.close()
