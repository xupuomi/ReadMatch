from rank_bm25 import BM25Okapi
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

# nltk.download("wordnet")
# nltk.download("omw-1.4")

_WN_LEMMATIZER = WordNetLemmatizer()

BOOKS = [
    {
        "id": 0,
        "Title": "Harry Potter and the Sorcerer's Stone",
        "Author": "J.K. Rowling",
        "Main Genre": "Fantasy",
        "Sub Genre": "Young Adult",
        "Type": "Novel",
        "Price": 9.99,
        "Rating": 4.8,
        "NoOfPeopleRated": 1_200_000,
        "Url": "https://example.com/hp1",
        # scraped from Amazon
        "Description": "A boy discovers he is a wizard and attends a magical boarding school."
    },
    {
        "id": 1,
        "Title": "The Martian",
        "Author": "Andy Weir",
        "Main Genre": "Science Fiction",
        "Sub Genre": "Survival",
        "Type": "Novel",
        "Price": 11.99,
        "Rating": 4.6,
        "NoOfPeopleRated": 900_000,
        "Url": "https://example.com/martian",
        "Description": "An astronaut is stranded alone on Mars and must use science and engineering to survive."
    },
    {
        "id": 2,
        "Title": "Project Hail Mary",
        "Author": "Andy Weir",
        "Main Genre": "Science Fiction",
        "Sub Genre": "Space Opera",
        "Type": "Novel",
        "Price": 13.99,
        "Rating": 4.7,
        "NoOfPeopleRated": 500_000,
        "Url": "https://example.com/phm",
        "Description": "A lone astronaut wakes up on a ship and must save Earth from an extinction-level threat."
    },
    {
        "id": 3,
        "Title": "A Court of Thorns and Roses",
        "Author": "Sarah J. Maas",
        "Main Genre": "Fantasy",
        "Sub Genre": "Romance",
        "Type": "Novel",
        "Price": 10.99,
        "Rating": 4.5,
        "NoOfPeopleRated": 800_000,
        "Url": "https://example.com/acotar",
        "Description": "A human huntress is pulled into faerie politics and dangerous romantic intrigue."
    },
    {
        "id": 4,
        "Title": "Deep Learning with Python",
        "Author": "François Chollet",
        "Main Genre": "Nonfiction",
        "Sub Genre": "Machine Learning",
        "Type": "Textbook",
        "Price": 49.99,
        "Rating": 4.6,
        "NoOfPeopleRated": 60_000,
        "Url": "https://example.com/dlwp",
        "Description": "An introduction to deep learning using Python and the Keras library."
    },
]

def build_search_text(book):
    """
    This is what you'll do for your real dataset:
    concatenate the fields you have.
    If Description is missing for a book, just treat it as "".
    """
    desc = book.get("Description", "") or ""
    return (
        f"{book['Title']}. {book['Author']}. "
        f"{book['Main Genre']} {book['Sub Genre']} {book['Type']}. "
        f"{desc}"
    )

for b in BOOKS:
    b["search_text"] = build_search_text(b)


STOPWORDS = set(stopwords.words('english'))

def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    tokens = [t.strip("'") for t in text.split() if t]
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens

def _rule_based_lemmatize(token):
    if token.endswith('ies') and len(token) > 4:
        return token[:-3] + 'y'
    if token.endswith('ing') and len(token) > 4:
        return token[:-3]
    if token.endswith('ed') and len(token) > 3:
        return token[:-2]
    if token.endswith('s') and len(token) > 3:
        return token[:-1]
    return token

def lemmatize_tokens(tokens):
    out = []
    for t in tokens:
        try:
            lv = _WN_LEMMATIZER.lemmatize(t, pos='v')
            ln = _WN_LEMMATIZER.lemmatize(lv, pos='n')
            out.append(ln)
        except Exception:
            out.append(_rule_based_lemmatize(t))
    return out

def proximity_boost(scores, original_query, tokenized_corpus, boost=1.0, window=5):
    if len(original_query) < 2:
        return scores

    t1, t2 = original_query[0], original_query[1]
    for i, doc_tokens in enumerate(tokenized_corpus):
        if t1 in doc_tokens and t2 in doc_tokens:
            pos1 = [idx for idx, tok in enumerate(doc_tokens) if tok == t1]
            pos2 = [idx for idx, tok in enumerate(doc_tokens) if tok == t2]
            if not pos1 or not pos2:
                continue
            min_dist = min(abs(p1 - p2) for p1 in pos1 for p2 in pos2)
            if min_dist <= window:
                scores[i] += boost
    return scores

def negation_penalty(scores, original_query, tokenized_corpus, penalty=1.5, window_before=1):
    negations = {'no', 'not', "n't"}
    for i, doc_tokens in enumerate(tokenized_corpus):
        for term in original_query:
            positions = [idx for idx, tok in enumerate(doc_tokens) if tok == term]
            for pos in positions:
                start = max(0, pos - window_before)
                if any(tok in negations for tok in doc_tokens[start:pos]):
                    scores[i] -= penalty
                    break
    return scores

corpus = [b["search_text"] for b in BOOKS]
tokenized_corpus = [simple_tokenize(doc) for doc in corpus]


sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
book_embeddings = sbert_model.encode(
    [b["search_text"] for b in BOOKS],
    convert_to_numpy=True,
    normalize_embeddings=True,
)



def normalize_scores(values):
    values = np.array(values, dtype="float32")
    v_min, v_max = values.min(), values.max()
    if v_max == v_min:
        return np.zeros_like(values)
    return (values - v_min) / (v_max - v_min)

def compute_popularity_score(rating, num_ratings):
    rating = float(rating) if rating is not None else 0.0
    num_ratings = float(num_ratings) if num_ratings is not None else 0.0
    rating_norm = rating / 5.0
    count_norm = np.log1p(num_ratings) / np.log1p(1e6)  # assume 1M ~= max
    return 0.7 * rating_norm + 0.3 * count_norm

def compute_title_match_score(query, title):
    q = query.lower()
    t = title.lower()
    if q == t:
        return 1.0
    if q in t or t in q:
        return 0.7
    q_tokens = set(q.split())
    t_tokens = set(t.split())
    if not q_tokens:
        return 0.0
    overlap = len(q_tokens & t_tokens)
    return min(0.6, overlap / len(q_tokens))

def compute_genre_match_score(query, main_genre, sub_genre):
    q = query.lower()
    mg = str(main_genre).lower()
    sg = str(sub_genre).lower()
    score = 0.0
    if mg and mg in q:
        score = max(score, 1.0)
    if sg and sg in q:
        score = max(score, 1.0)
    return score

def compute_semantic_scores(query: str):
    q_emb = sbert_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    return np.dot(book_embeddings, q_emb)  # cosine similarity

def combine_scores(query: str, bm25_raw, sem_raw):
    q = query.lower()
    bm25_raw = np.array(bm25_raw, dtype="float32")
    sem_raw = np.array(sem_raw, dtype="float32")

    bm25_norm = normalize_scores(bm25_raw)
    sem_norm = normalize_scores(sem_raw)

    title_scores = []
    genre_scores = []
    pop_scores = []

    for b in BOOKS:
        title_scores.append(compute_title_match_score(q, b["Title"]))
        genre_scores.append(compute_genre_match_score(q, b["Main Genre"], b["Sub Genre"]))
        pop_scores.append(compute_popularity_score(b["Rating"], b["NoOfPeopleRated"]))

    title_norm = normalize_scores(title_scores)
    genre_norm = np.array(genre_scores, dtype="float32")  # 0 or 1
    pop_norm = normalize_scores(pop_scores)

    # weights – tweak these
    w_bm25 = 0.25
    w_sem = 0.35
    w_title = 0.20
    w_genre = 0.05
    w_pop = 0.15

    final_scores = (
        w_bm25 * bm25_norm +
        w_sem * sem_norm +
        w_title * title_norm +
        w_genre * genre_norm +
        w_pop * pop_norm
    )

    debug = {
        "bm25_norm": bm25_norm,
        "sem_norm": sem_norm,
        "title_norm": title_norm,
        "genre_norm": genre_norm,
        "pop_norm": pop_norm,
    }
    return final_scores, debug


def test_hybrid(k1, b, queries, top_n=3):
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
    print(f"\n=== Hybrid search with BM25(k1={k1}, b={b}) ===")
    for query in queries:
        qtoks = lemmatize_tokens(simple_tokenize(query))
        bm25_scores = bm25.get_scores(qtoks)
        bm25_scores = proximity_boost(bm25_scores, qtoks, tokenized_corpus, boost=1.0, window=5)
        bm25_scores = negation_penalty(bm25_scores, qtoks, tokenized_corpus, penalty=1.5, window_before=1)

        sem_scores = compute_semantic_scores(query)

        final_scores, debug = combine_scores(query, bm25_scores, sem_scores)
        order = np.argsort(final_scores)[::-1]

        print(f"\nQuery: '{query}'")
        print("Rank | Final | BM25  | Sem   | Title | Genre | Pop   | Title")
        print("-----+-------+-------+-------+-------+-------+-------+------------------------------")
        for rank, idx in enumerate(order[:top_n]):
            print(
                f"{rank+1:>4} | "
                f"{final_scores[idx]:.3f} | "
                f"{debug['bm25_norm'][idx]:.3f} | "
                f"{debug['sem_norm'][idx]:.3f} | "
                f"{debug['title_norm'][idx]:.3f} | "
                f"{debug['genre_norm'][idx]:.3f} | "
                f"{debug['pop_norm'][idx]:.3f} | "
                f"{BOOKS[idx]['Title'][:28]}"
            )

if __name__ == "__main__":
    queries = [
        "space survival",
        "cozy fantasy romance",
        "machine learning book",
        "wizard boarding school",
    ]
    for k1, b in [(1.2, 0.75), (0.9, 0.5)]:
        test_hybrid(k1, b, queries, top_n=3)
