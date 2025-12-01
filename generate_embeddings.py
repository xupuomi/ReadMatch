import sqlalchemy
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

engine = sqlalchemy.create_engine("sqlite:///readmatch.db")

df = pd.read_sql("SELECT * FROM books", engine)

def build_search_text(row):
    return f"{row['title']}. {row['authors']}. {row['genres']}. {row['description'] or ''}"
    
df["search_text"] = df.apply(build_search_text, axis=1)

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    df["search_text"].tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True,
)

with engine.begin() as conn:
    for book_id, vector in zip(df["book_id"], embeddings):
        conn.execute(sqlalchemy.text("""
            INSERT OR REPLACE INTO embeddings (book_id, model, vector)
            VALUES (:book_id, :model, :vector)
        """), {
            "book_id": int(book_id),
            "model": "sbert_all-MiniLM-L6-v2",
            "vector": vector.tobytes(),
        })
