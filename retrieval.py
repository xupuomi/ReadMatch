from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOPWORDS = set(stopwords.words("english"))
_WN_LEMMATIZER = WordNetLemmatizer()
sbert_model = SentenceTransformer("all-MiniLM-L6-v2") 

def normalize_title(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)  
    s = re.sub(r"\s+", " ", s)    
    return s

# Combines all info about the books for matching
def build_search_text(book):
    return (
        f"{book.get('title','')}. "
        f"{book.get('authors','')}. "
        f"{book.get('genres','')}. "
        f"{book.get('description','')}"
    )

# Process input
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

# Give boosts 
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

# Build BM25 Index
def build_bm25_index(books):
    corpus = [build_search_text(b) for b in books]
    tokenized_corpus = [simple_tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus, corpus

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
    count_norm = np.log1p(num_ratings) / np.log1p(1e6)  
    return 0.7 * rating_norm + 0.3 * count_norm

def compute_title_match_score(query, title):
    nq = normalize_title(query)
    nt = normalize_title(title)

    if not nq or not nt:
        return 0.0
    if nq == nt:
        return 1.0
    
    q_tokens = nq.split()
    t_tokens = nt.split()

    if len(q_tokens) <= len(t_tokens):
        for i in range(len(t_tokens) - len(q_tokens) + 1):
            if t_tokens[i:i+len(q_tokens)] == q_tokens:
                return 0.9

    q_set = set(q_tokens)
    t_set = set(t_tokens)
    overlap = len(q_set & t_set)

    if len(q_tokens) >= 2 and overlap >= len(q_tokens) - 1:
        return 0.8

    if overlap > 0:
        return 0.5 * (overlap / len(q_tokens))

    return 0.0


def compute_genre_match_score(query, genre):
    nq = normalize_title(query)
    q_tokens = set(nq.split())

    g = str(genre).lower().strip()
    if not g:
        return 0.0

    g_tokens = set(re.split(r"[,\s]+", g)) 
    if not g_tokens:
        return 0.0

    overlap = len(q_tokens & g_tokens)
    return overlap / len(g_tokens)


def compute_semantic_scores(query, book_embeddings):
    q_emb = sbert_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]
    return np.dot(book_embeddings, q_emb)  # cosine similarities

def compute_user_pref_scores(book_embeddings, user_profile_emb):
    if user_profile_emb is None:
        return np.zeros(len(book_embeddings), dtype="float32")
    return np.dot(book_embeddings, user_profile_emb)


def combine_scores(query, bm25_raw, sem_raw, books, user_pref_raw=None):
    q = query.lower()
    bm25_raw = np.array(bm25_raw, dtype="float32")
    sem_raw = np.array(sem_raw, dtype="float32")

    bm25_norm = normalize_scores(bm25_raw)
    sem_norm = normalize_scores(sem_raw)

    if user_pref_raw is not None:
        user_pref_raw = np.array(user_pref_raw, dtype="float32")
        user_pref_norm = normalize_scores(user_pref_raw)
    else:
        user_pref_norm = np.zeros_like(bm25_norm)
    
    print("User preference raw scores:", user_pref_raw)

    title_scores = []
    main_genre_scores = []
    sub_genre_scores = []
    pop_scores = []

    for b in books:
        genres = b.get("genres", "")
        main_genre, sub_genres = genres[0], genres[1:]
        title_scores.append(compute_title_match_score(q, b["title"]))
        main_genre_scores.append(compute_genre_match_score(q, main_genre))
        sub_genre_scores.append(compute_genre_match_score(q, sub_genres))
        pop_scores.append(compute_popularity_score(b.get("avg_rating"), b.get("review_count")))

    title_norm = normalize_scores(title_scores)
    main_genre_norm = np.array(main_genre_scores, dtype="float32")
    sub_genre_norm = np.array(sub_genre_scores, dtype="float32")
    pop_norm = normalize_scores(pop_scores)

    w_bm25 = 0.22
    w_sem = 0.32
    w_title = 0.18
    w_main_genre = 0.09
    w_sub_genre = 0.05
    w_pop = 0.13
    w_user = 0.05  

    final_scores = (
        w_bm25 * bm25_norm +
        w_sem * sem_norm +
        w_title * title_norm +
        w_main_genre * main_genre_norm +
        w_sub_genre * sub_genre_norm +
        w_pop * pop_norm +
        w_user * user_pref_norm
    )

    debug = {
        "bm25_norm": bm25_norm,
        "sem_norm": sem_norm,
        "title_norm": title_norm,
        "main_genre_norm": main_genre_norm,
        "sub_genre_norm": sub_genre_norm,
        "pop_norm": pop_norm,
        "user_pref_norm": user_pref_norm,
    }
    return final_scores, debug


def is_title_match_like(query: str, title: str) -> bool:
    nq = normalize_title(query)
    nt = normalize_title(title)

    if not nq or not nt:
        return False

    if nq == nt:
        return True

    q_tokens = nq.split()
    t_tokens = nt.split()

    if len(q_tokens) <= len(t_tokens):
        for i in range(len(t_tokens) - len(q_tokens) + 1):
            if t_tokens[i:i + len(q_tokens)] == q_tokens:
                return True

    q_set = set(q_tokens)
    t_set = set(t_tokens)
    overlap = len(q_set & t_set)

    if len(q_tokens) <= 2:
        if overlap == len(q_tokens):
            return True
        return False 
    
    if overlap >= len(q_tokens) - 1:
        return True

    return False


def rank_books(query, books, bm25, tokenized_corpus, book_embeddings,
               user_profile_emb=None, top_n=10):
    qtoks = lemmatize_tokens(simple_tokenize(query))
    bm25_scores = bm25.get_scores(qtoks)
    bm25_scores = proximity_boost(bm25_scores, qtoks, tokenized_corpus, boost=1.0, window=5)
    bm25_scores = negation_penalty(bm25_scores, qtoks, tokenized_corpus, penalty=1.5, window_before=1)
    sem_scores = compute_semantic_scores(query, book_embeddings)
    user_pref_raw = compute_user_pref_scores(book_embeddings, user_profile_emb)

    final_scores, debug = combine_scores(
        query,
        bm25_scores,
        sem_scores,
        books,
        user_pref_raw=user_pref_raw, 
    )
    final_scores = np.array(final_scores, dtype="float32")

    exact_idxs = []
    other_idxs = []
    exact_flags = []

    for i, b in enumerate(books):
        is_exact_like = is_title_match_like(query, b.get("title", ""))
        exact_flags.append(is_exact_like)
        if is_exact_like:
            exact_idxs.append(i)
        else:
            other_idxs.append(i)

    exact_sorted = sorted(exact_idxs, key=lambda i: final_scores[i], reverse=True)
    others_sorted = sorted(other_idxs, key=lambda i: final_scores[i], reverse=True)

    order_list = exact_sorted + others_sorted
    if top_n is not None:
        order_list = order_list[:top_n]

    order = np.array(order_list, dtype=int)
    debug["exact_title_match"] = np.array(exact_flags, dtype=bool)
    debug["user_pref_raw"] = user_pref_raw  

    return order, final_scores, debug
