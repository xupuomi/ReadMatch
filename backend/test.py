# eval_retrieval.py
import numpy as np
from collections import OrderedDict

from retrieval import (
    build_bm25_index,
    build_search_text,
    rank_books,
    simple_tokenize,
    lemmatize_tokens,
    proximity_boost,
    negation_penalty,
    compute_semantic_scores,
    sbert_model,
)


# ------------------------------
# Mock data
# ------------------------------

def build_mock_books():
    """Small sample of books to test ranking behavior."""
    books = [
        {
            "book_id": 1,
            "title": "Book Lovers",
            "authors": "Emily Henry",
            "genres": "romance, contemporary",
            "description": "An enemies-to-lovers slow burn romance about a bookish literary agent and an editor.",
            "avg_rating": 4.2,
            "review_count": 250000,
        },
        {
            "book_id": 2,
            "title": "Beach Read",
            "authors": "Emily Henry",
            "genres": "romance, contemporary",
            "description": "Two writers with opposite styles swap genres over the summer and fall in love.",
            "avg_rating": 4.1,
            "review_count": 300000,
        },
        {
            "book_id": 3,
            "title": "The Book Thief",
            "authors": "Markus Zusak",
            "genres": "historical fiction, world war ii",
            "description": "A young girl in Nazi Germany steals books and shares them during the horrors of WWII.",
            "avg_rating": 4.4,
            "review_count": 2500000,
        },
        {
            "book_id": 4,
            "title": "Project Hail Mary",
            "authors": "Andy Weir",
            "genres": "science fiction, space",
            "description": "A lone astronaut must save Earth from an extinction-level threat in a hard science space adventure.",
            "avg_rating": 4.5,
            "review_count": 800000,
        },
        {
            "book_id": 5,
            "title": "The Night Circus",
            "authors": "Erin Morgenstern",
            "genres": "fantasy, magical realism",
            "description": "A lush, atmospheric fantasy about a magical competition set in a mysterious circus.",
            "avg_rating": 4.1,
            "review_count": 900000,
        },
        {
            "book_id": 6,
            "title": "Mistborn: The Final Empire",
            "authors": "Brandon Sanderson",
            "genres": "fantasy, epic",
            "description": "A heist story in a dark, ash-filled world where magic comes from ingesting metals.",
            "avg_rating": 4.4,
            "review_count": 1100000,
        },
        {
            "book_id": 7,
            "title": "Atomic Habits",
            "authors": "James Clear",
            "genres": "nonfiction, self-help",
            "description": "A practical guide to building good habits and breaking bad ones through tiny, consistent changes.",
            "avg_rating": 4.4,
            "review_count": 1500000,
        },
        {
            "book_id": 8,
            "title": "Educated",
            "authors": "Tara Westover",
            "genres": "memoir, nonfiction",
            "description": "A woman who grows up in a strict, survivalist family escapes through education.",
            "avg_rating": 4.5,
            "review_count": 1300000,
        },
        {
            "book_id": 9,
            "title": "Twilight",
            "authors": "Stephanie Meyer",
            "genres": "romance, contemporary",
            "description": "A high school finds herself falling in love with a vampire.",
            "avg_rating": 4.7,
            "review_count": 1230000,
        },
    ]
    return books


def build_book_embeddings(books):
    """Encode each book's combined text into an embedding (using retrieval.build_search_text)."""
    texts = [build_search_text(b) for b in books]
    emb = sbert_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.array(emb, dtype="float32")


# ------------------------------
# Model variants for evaluation
# ------------------------------

def rank_bm25_only(query, books, bm25, tokenized_corpus, book_embeddings=None,
                   user_profile_emb=None, top_n=10):
    """Ranking using only BM25 (plus your proximity/negation tweaks)."""
    qtoks = lemmatize_tokens(simple_tokenize(query))
    scores = bm25.get_scores(qtoks)
    scores = proximity_boost(scores, qtoks, tokenized_corpus, boost=1.0, window=5)
    scores = negation_penalty(scores, qtoks, tokenized_corpus, penalty=1.5, window_before=1)
    scores = np.array(scores, dtype="float32")
    order = np.argsort(scores)[::-1]
    return order, scores, {}


def rank_semantic_only(query, books, bm25, tokenized_corpus, book_embeddings,
                       user_profile_emb=None, top_n=10):
    """Ranking using only semantic similarity between query and book embeddings."""
    scores = compute_semantic_scores(query, book_embeddings)
    scores = np.array(scores, dtype="float32")
    order = np.argsort(scores)[::-1]
    return order, scores, {}


def rank_hybrid(query, books, bm25, tokenized_corpus, book_embeddings,
                user_profile_emb=None, top_n=10):
    """Your full hybrid ranker from retrieval.rank_books."""
    return rank_books(
        query=query,
        books=books,
        bm25=bm25,
        tokenized_corpus=tokenized_corpus,
        book_embeddings=book_embeddings,
        user_profile_emb=user_profile_emb,
        top_n=top_n,
    )


# ------------------------------
# Accuracy / evaluation helpers
# ------------------------------

def accuracy_at_k(rank_fn, test_queries, k, books, bm25, tokenized_corpus,
                  book_embeddings, user_profile_emb=None):
    """
    rank_fn: function(query, books, bm25, tokenized_corpus, book_embeddings, user_profile_emb, top_n)
    test_queries: dict(query -> [expected_titles])
    returns accuracy@k
    """
    total = len(test_queries)
    correct = 0

    for query, expected_titles in test_queries.items():
        order, scores, _ = rank_fn(
            query=query,
            books=books,
            bm25=bm25,
            tokenized_corpus=tokenized_corpus,
            book_embeddings=book_embeddings,
            user_profile_emb=user_profile_emb,
            top_n=k,
        )
        top_titles = [books[i]["title"] for i in order[:k]]
        if any(exp in top_titles for exp in expected_titles):
            correct += 1

    return correct / total if total > 0 else 0.0


def pretty_top_k(query, rank_fn, books, bm25, tokenized_corpus,
                 book_embeddings, user_profile_emb=None, k=5, label="model"):
    print(f"\nQuery: '{query}'  -- {label}")
    order, scores, _ = rank_fn(
        query=query,
        books=books,
        bm25=bm25,
        tokenized_corpus=tokenized_corpus,
        book_embeddings=book_embeddings,
        user_profile_emb=user_profile_emb,
        top_n=k,
    )
    for rank, idx in enumerate(order[:k], start=1):
        b = books[idx]
        print(f"  {rank:>2}. {b['title']} ({b['genres']})  score={scores[idx]:.4f}")


# ------------------------------
# Mock user profile embedding
# ------------------------------

def build_mock_user_profile(book_embeddings, liked_indices, ratings=None):
    """
    Build a fake user profile vector from books they 'like'.
    liked_indices: list of indices into books/book_embeddings
    ratings: optional list aligned with liked_indices, e.g. [5, 4, 5]
    """
    if not liked_indices:
        return None

    liked_vecs = book_embeddings[liked_indices]

    if ratings is None:
        w = np.ones(len(liked_indices), dtype="float32")
    else:
        w = np.array(ratings, dtype="float32")
        # center around 3 and ensure positive
        w = np.maximum(w - 3.0, 0.1)

    user_vec = (w[:, None] * liked_vecs).sum(axis=0) / w.sum()
    norm = np.linalg.norm(user_vec)
    if norm == 0:
        return None
    return user_vec / norm


# ------------------------------
# Main test harness
# ------------------------------

if __name__ == "__main__":
    # 1. Build mock data
    BOOKS = build_mock_books()
    bm25_index, tokenized_corpus, corpus = build_bm25_index(BOOKS)
    book_embeddings = build_book_embeddings(BOOKS)

    # 2. Define small test query set with expected titles
    test_queries = OrderedDict({
        "book lovers": ["Book Lovers"],
        "emily henry romance": ["Book Lovers", "Beach Read"],
        "world war ii historical fiction": ["The Book Thief"],
        "magic circus": ["The Night Circus"],
        "space hard sci fi": ["Project Hail Mary"],
        "habits self help": ["Atomic Habits"],
        "epic fantasy": ["Mistborn: The Final Empire"],
    })

    Ks = [1, 3, 5]

    print("=== EVALUATION WITHOUT USER PERSONALIZATION ===")
    for k in Ks:
        acc_bm25 = accuracy_at_k(
            rank_bm25_only,
            test_queries,
            k,
            BOOKS,
            bm25_index,
            tokenized_corpus,
            book_embeddings,
        )
        acc_sem = accuracy_at_k(
            rank_semantic_only,
            test_queries,
            k,
            BOOKS,
            bm25_index,
            tokenized_corpus,
            book_embeddings,
        )
        acc_hybrid = accuracy_at_k(
            rank_hybrid,
            test_queries,
            k,
            BOOKS,
            bm25_index,
            tokenized_corpus,
            book_embeddings,
        )

        print(f"\nAccuracy@{k}:")
        print(f"  BM25-only      : {acc_bm25:.2f}")
        print(f"  Semantic-only  : {acc_sem:.2f}")
        print(f"  Hybrid (full)  : {acc_hybrid:.2f}")

    # 3. Build a mock user who LOVES romance + fantasy
    liked_indices = [0, 1, 4, 5]              # Book Lovers, Beach Read, Night Circus, Mistborn
    liked_ratings = [5, 4, 5, 5]
    user_profile_emb = build_mock_user_profile(book_embeddings, liked_indices, liked_ratings)

    print("\n=== MOCK USER PROFILE ===")
    print("Liked books:")
    idx = 0
    for i in liked_indices:
        print(" ", BOOKS[i]["title"], "-", BOOKS[i]["genres"], "- Rating: ", liked_ratings[idx])
        idx += 1
    print("User profile embedding norm:", np.linalg.norm(user_profile_emb))

    # 4. Evaluate hybrid with personalization
    print("\n=== EVALUATION WITH USER PERSONALIZATION (Hybrid + user_pref) ===")
    for k in Ks:
        acc_hybrid_user = accuracy_at_k(
            rank_hybrid,
            test_queries,
            k,
            BOOKS,
            bm25_index,
            tokenized_corpus,
            book_embeddings,
            user_profile_emb=user_profile_emb,
        )
        print(f"  Accuracy@{k} (Hybrid + user) @ {k}: {acc_hybrid_user:.2f}")

    # 5. Show qualitative examples with and without personalization
    demo_queries = [
        "slow burn romance",
        "fantasy with magic",
        "character-driven fantasy romance",
    ]

    for q in demo_queries:
        pretty_top_k(
            q,
            rank_hybrid,
            BOOKS,
            bm25_index,
            tokenized_corpus,
            book_embeddings,
            user_profile_emb=None,
            k=5,
            label="Hybrid (no user)",
        )
        pretty_top_k(
            q,
            rank_hybrid,
            BOOKS,
            bm25_index,
            tokenized_corpus,
            book_embeddings,
            user_profile_emb=user_profile_emb,
            k=5,
            label="Hybrid (with user)",
        )
