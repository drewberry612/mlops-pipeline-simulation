import sqlite3

def init_db():
    # Connect (creates the file if it doesn’t exist)
    conn = sqlite3.connect("warehouse.db")
    cur = conn.cursor()

    # Read and execute schema
    with open("schema.sql", "r") as f:
        schema = f.read()
    cur.executescript(schema)

    conn.commit()
    conn.close()
    print("Database initialised...")

if __name__ == "__main__":
    init_db()
