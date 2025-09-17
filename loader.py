import pandas as pd
import sqlalchemy

# load CSV
df = pd.read_csv("cleaned_data.csv")

# connect to SQLite
engine = sqlalchemy.create_engine("sqlite:///readmatch.db")

ddl = """
CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title VARCHAR(512) NOT NULL,
    authors VARCHAR(512),
    genres VARCHAR(512),
    description TEXT,
    avg_rating FLOAT,
    review_count INT,
    publish_year INT,
    publisher VARCHAR(512)
);

CREATE TABLE IF NOT EXISTS embeddings (
    book_id INT,
    model VARCHAR(50),
    vector BLOB,
    PRIMARY KEY (book_id, model),
    FOREIGN KEY (book_id) REFERENCES books(book_id)
);

CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255),
    email VARCHAR(255) UNIQUE
);

CREATE TABLE IF NOT EXISTS user_likes (
    user_id INT,
    book_id INT,
    rating FLOAT,
    PRIMARY KEY (user_id, book_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (book_id) REFERENCES books(book_id)
);
"""
with engine.connect() as conn:
    for stmt in ddl.strip().split(";\n"):
        if stmt.strip():
            conn.execute(sqlalchemy.text(stmt))

df_mapped = pd.DataFrame({
    "title": df["Title"],
    "authors": df["Author"],
    "genres": df["Main Genre"] + " - " + df["Sub Genre"],
    "description": df["URLs"],  # store URLs in description
    "avg_rating": df["Rating"],
    "review_count": df["No. of People rated"],
    "publish_year": None,  # not available in CSV
    "publisher": df["Type"]
})

# insert into books table
df_mapped.to_sql("books", engine, if_exists="append", index=False)

print("Books table populated from CSV")
