from flask import Flask, request, jsonify
from flask_cors import CORS
from rank_bm25 import BM25Okapi

app = Flask(__name__)
CORS(app)

# --- Step 1: Load corpus (temporary toy example, later Amazon dataset) ---
corpus = [
    "A thrilling journey through space and time with astronauts.",
    "An introduction to machine learning and artificial intelligence.",
    "The history of ancient Rome and its powerful emperors.",
    "Learn Python programming with hands-on projects and examples.",
    "Exploring the wonders of the deep ocean and marine biology."
]

tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# --- Step 2: Flask route ---
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    query = data.get("query", "").lower()
    mode = data.get("mode", "book")   # you can still keep modes: "book" / "theme"

    if not query:
        return jsonify({"recommendations": []})

    tokenized_query = query.split()
    top_n = bm25.get_top_n(tokenized_query, corpus, n=3)

    return jsonify({"recommendations": top_n})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
