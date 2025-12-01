from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlalchemy
import numpy as np

import retrieval #retrieval.py 

app = Flask(__name__)
CORS(app)

engine = sqlalchemy.create_engine("sqlite:///readmatch.db")
MODEL_NAME = "sbert_all-MiniLM-L6-v2"

# Load books + embeddings during start 
with engine.connect() as conn:
    rows = conn.execute(sqlalchemy.text("""
        SELECT b.book_id,
               b.title,
               b.authors,
               b.genres,
               b.description,
               b.avg_rating,
               b.review_count,
               e.vector
        FROM books b
        JOIN embeddings e
          ON b.book_id = e.book_id
        WHERE e.model = :model
    """), {"model": MODEL_NAME}).fetchall()

BOOKS = []
emb_list = []
for row in rows:
    BOOKS.append({
        "book_id": row.book_id,
        "title": row.title,
        "authors": row.authors,
        "genres": row.genres,
        "description": row.description,
        "avg_rating": row.avg_rating,
        "review_count": row.review_count,
    })
    emb_list.append(np.frombuffer(row.vector, dtype="float32"))

book_embeddings = np.vstack(emb_list)

# Build BM25 index
bm25_index, tokenized_corpus, corpus_strings = retrieval.build_bm25_index(BOOKS)

# Return top 10 most similar books
@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = (data.get("query") or "").strip()
    top_n = 10

    if not query:
        return jsonify({"results": []})

    order, final_scores, debug = retrieval.rank_books(
        query=query,
        books=BOOKS,
        bm25=bm25_index,
        tokenized_corpus=tokenized_corpus,
        book_embeddings=book_embeddings,
        top_n=top_n,
    )

    results = []
    for idx in order[:top_n]:
        b = BOOKS[idx]
        results.append({
            "book_id": b["book_id"],
            "title": b["title"],
            "authors": b["authors"],
            "genres": b["genres"],
            "description": b["description"],
            "avg_rating": b["avg_rating"],
            "review_count": b["review_count"],
            "score": float(final_scores[idx]),
        })

    return jsonify({"results": results})
