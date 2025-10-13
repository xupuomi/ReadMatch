from rank_bm25 import BM25Okapi
import numpy as np
import re
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
_WN_LEMMATIZER = WordNetLemmatizer()
_LEMMATIZER_AVAILABLE = True

corpus = [
    "A thrilling journey through space and time with astronauts.",
    "There is no space on this paper",
    "An introduction to machine learning and artificial intelligence.",
    "I've done lots of learning about machines",
    "The history of ancient Rome and its powerful emperors.",
    "Learn Python programming with hands-on projects and examples.",
    "Exploring the wonders of the deep ocean and marine biology."
]

_STOPWORDS = set([
    "the", "is", "a", "an", "and", "of", "on", "this", "there",
    "with", "its", "i", "ive", "about", "to", "in", "for", "that"
])

def simple_tokenize(text):
    text = text.lower()

    text = re.sub(r"[^a-z0-9\s']", " ", text)
    tokens = [t.strip("'") for t in text.split() if t]
    tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens

tokenized_corpus = [simple_tokenize(doc) for doc in corpus]

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
        lv = _WN_LEMMATIZER.lemmatize(t, pos='v')
        ln = _WN_LEMMATIZER.lemmatize(lv, pos='n')
        out.append(ln)

    return out

# for non exact matches
# next steps: replace with word2vec for semantic similarity 
SYNONYMS = {
    'space': ['astronauts', 'spaceflight', 'cosmos', 'outerspace', 'spacecraft'],
    'exploration': ['explore', 'exploring', 'explorer', 'expedition']
}

def expand_query_tokens(tokens):
    out = []
    for t in tokens:
        out.append(t)
        if t in SYNONYMS:
            out.extend(SYNONYMS[t])
    return out

def proximity_boost(scores, tokenized_query, original_query, tokenized_corpus, boost=1.0, window=5):
    orig_terms = original_query
    if len(orig_terms) < 2:
        return scores

    t1, t2 = orig_terms[0], orig_terms[1]
    for i, doc_tokens in enumerate(tokenized_corpus):
        if t1 in doc_tokens and t2 in doc_tokens:
            pos1 = [idx for idx, tok in enumerate(doc_tokens) if tok == t1]
            pos2 = [idx for idx, tok in enumerate(doc_tokens) if tok == t2]
            min_dist = min(abs(p1 - p2) for p1 in pos1 for p2 in pos2)
            if min_dist <= window:
                scores[i] += boost
    return scores

def negation_penalty(scores, original_query, tokenized_corpus, penalty=1.5, window_before=1):
    # "no" space -> penalize space
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

def test_bm25(k1, b, queries):
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
    print(f"\nBM25 parameters: k1={k1}, b={b}")
    for query in queries:
        original_tokens = simple_tokenize(query)
        original_tokens = lemmatize_tokens(original_tokens)
        tokenized_query = expand_query_tokens(original_tokens)
        scores = bm25.get_scores(tokenized_query)
        scores = proximity_boost(scores, tokenized_query, original_tokens, tokenized_corpus, boost=1.0, window=5)
        scores = negation_penalty(scores, original_tokens, tokenized_corpus, penalty=1.5, window_before=1)

        top_n = bm25.get_top_n(tokenized_query, corpus, n=3)
        print(f"Query: '{query}'")
        print("Scores:", np.round(scores, 3))
        print("Top 3 documents:", top_n)

queries = [
    "space exploration",
    "python programming",
    "machine learning",
    "ancient history",
    "marine biology"
]

for k1, b in [(1.2, 0.75), (0.9, 0.5), (1.5, 0.9)]:
    test_bm25(k1, b, queries)

