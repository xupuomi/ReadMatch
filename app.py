from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlalchemy
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash

import retrieval

app = Flask(__name__)
# allow origins
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    allow_headers=["Content-Type"],
    methods=["GET", "POST", "DELETE", "PUT", "OPTIONS"],
    origins=["http://localhost:3000"]
)

engine = sqlalchemy.create_engine("sqlite:///readmatch.db")
MODEL_NAME = "sbert_all-MiniLM-L6-v2"

# verify user-related tables exist
with engine.begin() as conn:
    conn.execute(sqlalchemy.text("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(255) UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        );
    """))
    conn.execute(sqlalchemy.text("""
        CREATE TABLE IF NOT EXISTS user_want_to_read (
            user_id INT,
            book_id INT,
            PRIMARY KEY (user_id, book_id),
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (book_id) REFERENCES books(book_id)
        );
    """))
    conn.execute(sqlalchemy.text("""
        CREATE TABLE IF NOT EXISTS user_ratings (
            user_id INT,
            book_id INT,
            rating FLOAT,
            PRIMARY KEY (user_id, book_id),
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (book_id) REFERENCES books(book_id)
        );
    """))

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

if emb_list:
    book_embeddings = np.vstack(emb_list)
else:
    book_embeddings = np.zeros((0, 384), dtype="float32")

# user profile based on past rated books
def get_user_profile_vector(engine, user_id, model_name=MODEL_NAME):
    with engine.begin() as conn:
        result = conn.execute(sqlalchemy.text("""
            SELECT
                e.vector AS vec_blob,
                ur.rating AS rating,
                b.title  AS title
            FROM user_ratings ur
            JOIN embeddings e
                ON ur.book_id = e.book_id
               AND e.model = :model
            JOIN books b
                ON b.book_id = ur.book_id
            WHERE ur.user_id = :user_id
        """), {"user_id": user_id, "model": model_name})

        rows = result.mappings().all() 

    if not rows:
        return None

    vecs = []
    weights = []

    for row in rows:
        blob = row["vec_blob"]
        rating = float(row["rating"])
        title = row["title"]
        print(f"   rated {title!r} with rating={rating}")
        v = np.frombuffer(blob, dtype=np.float32)
        w = max(rating, 1.0)
        vecs.append(v)
        weights.append(w)

    if not vecs:
        return None
    V = np.stack(vecs, axis=0)
    w = np.array(weights, dtype=np.float32)
    user_vec = (w[:, None] * V).sum(axis=0) / w.sum()
    norm = np.linalg.norm(user_vec)
    if norm == 0:
        return None
    user_vec = user_vec / norm
    return user_vec


# Build BM25 index
if BOOKS:
    bm25_index, tokenized_corpus, corpus_strings = retrieval.build_bm25_index(BOOKS)
else:
    bm25_index, tokenized_corpus, corpus_strings = None, [], []

# Return top 10 most similar books
@app.route("/search", methods=["POST"])
def search():
    data = request.get_json() or {}
    query = (data.get("query") or "").strip()
    top_n = 10

    if not query or not BOOKS:
        return jsonify({"results": []})
    
    user_id = data.get("user_id")  
    print("Search user_id:", user_id)
    if user_id is not None:
        user_profile_emb = get_user_profile_vector(engine, user_id)
        print("User profile emb is None?", user_profile_emb is None)
    else:
        user_profile_emb = None

    order, final_scores, debug = retrieval.rank_books(
        query=query,
        books=BOOKS,
        bm25=bm25_index,
        tokenized_corpus=tokenized_corpus,
        book_embeddings=book_embeddings,
        user_profile_emb=user_profile_emb,  
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


@app.route("/book/<int:book_id>", methods=["GET"])
def book_detail(book_id):
    user_id = request.args.get("user_id", type=int)
    with engine.connect() as conn:
        row = conn.execute(sqlalchemy.text("""
            SELECT book_id, title, authors, genres, description, avg_rating, review_count
            FROM books
            WHERE book_id = :book_id
        """), {"book_id": book_id}).fetchone()
        user_rating = None
        if user_id is not None:
            rrow = conn.execute(sqlalchemy.text("""
                SELECT rating FROM user_ratings
                WHERE user_id = :user_id AND book_id = :book_id
            """), {"user_id": user_id, "book_id": book_id}).fetchone()
            if rrow:
                user_rating = rrow.rating
    if not row:
        return jsonify({"error": "not found"}), 404
    payload = dict(row._mapping)
    if user_rating is not None:
        payload["user_rating"] = user_rating
    return jsonify(payload)


@app.route("/users/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not username or not password:
        return jsonify({"error": "username and password required"}), 400
    pwd_hash = generate_password_hash(password, method="pbkdf2:sha256")
    try:
        with engine.begin() as conn:
            res = conn.execute(sqlalchemy.text("""
                INSERT INTO users (username, password_hash)
                VALUES (:username, :password_hash)
            """), {"username": username, "password_hash": pwd_hash})
            user_id = res.lastrowid
    except sqlalchemy.exc.IntegrityError:
        return jsonify({"error": "username already exists"}), 409
    return jsonify({"user_id": user_id, "username": username})


@app.route("/users/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not username or not password:
        return jsonify({"error": "username and password required"}), 400
    with engine.connect() as conn:
        row = conn.execute(sqlalchemy.text("""
            SELECT user_id, password_hash FROM users WHERE username = :username
        """), {"username": username}).fetchone()
    if not row or not check_password_hash(row.password_hash, password):
        return jsonify({"error": "invalid credentials"}), 401
    return jsonify({"user_id": row.user_id, "username": username})


@app.route("/users/<int:user_id>/want_to_read", methods=["POST"])
def add_want_to_read(user_id):
    data = request.get_json() or {}
    book_id = data.get("book_id")
    if book_id is None:
        return jsonify({"error": "book_id required"}), 400
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("""
            INSERT OR IGNORE INTO user_want_to_read (user_id, book_id)
            VALUES (:user_id, :book_id)
        """), {"user_id": user_id, "book_id": book_id})
    return jsonify({"status": "saved", "user_id": user_id, "book_id": book_id})


@app.route("/users/<int:user_id>/want_to_read/<int:book_id>", methods=["DELETE"])
def remove_want_to_read(user_id, book_id):
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("""
            DELETE FROM user_want_to_read
            WHERE user_id = :user_id AND book_id = :book_id
        """), {"user_id": user_id, "book_id": book_id})
    return jsonify({"status": "removed", "user_id": user_id, "book_id": book_id})


@app.route("/users/<int:user_id>/rating", methods=["POST"])
def set_rating(user_id):
    data = request.get_json() or {}
    book_id = data.get("book_id")
    rating = data.get("rating")
    if book_id is None or rating is None:
        return jsonify({"error": "book_id and rating required"}), 400
    try:
        rating_val = float(rating)
    except ValueError:
        return jsonify({"error": "rating must be numeric"}), 400
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("""
            INSERT INTO user_ratings (user_id, book_id, rating)
            VALUES (:user_id, :book_id, :rating)
            ON CONFLICT(user_id, book_id) DO UPDATE SET rating = excluded.rating
        """), {"user_id": user_id, "book_id": book_id, "rating": rating_val})
    return jsonify({"status": "saved", "user_id": user_id, "book_id": book_id, "rating": rating_val})


@app.route("/users/<int:user_id>/rating/<int:book_id>", methods=["DELETE"])
def remove_rating(user_id, book_id):
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("""
            DELETE FROM user_ratings
            WHERE user_id = :user_id AND book_id = :book_id
        """), {"user_id": user_id, "book_id": book_id})
    return jsonify({"status": "removed", "user_id": user_id, "book_id": book_id})


@app.route("/users/<int:user_id>/want_to_read", methods=["GET"])
def list_want_to_read(user_id):
    with engine.connect() as conn:
        rows = conn.execute(sqlalchemy.text("""
            SELECT b.book_id, b.title, b.authors, b.genres, b.description, b.avg_rating, b.review_count
            FROM user_want_to_read uw
            JOIN books b ON b.book_id = uw.book_id
            WHERE uw.user_id = :user_id
        """), {"user_id": user_id}).fetchall()
    return jsonify({"books": [dict(r._mapping) for r in rows]})


@app.route("/users/<int:user_id>/ratings", methods=["GET"])
def list_user_ratings(user_id):
    with engine.connect() as conn:
        rows = conn.execute(sqlalchemy.text("""
            SELECT ur.book_id, ur.rating,
                   b.title, b.authors, b.genres, b.description, b.avg_rating, b.review_count
            FROM user_ratings ur
            JOIN books b ON b.book_id = ur.book_id
            WHERE ur.user_id = :user_id
        """), {"user_id": user_id}).fetchall()
    return jsonify({"ratings": [dict(r._mapping) for r in rows]})


if __name__ == "__main__":
    # allow running with python3 app.py
    app.run(host="127.0.0.1", port=5000, debug=True)
