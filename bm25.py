# Step 1: Import the BM25 library
from rank_bm25 import BM25Okapi

# Step 2: Create your "corpus" of documents
# Each document is a book description (or title+description combined)
corpus = [
    "A thrilling journey through space and time with astronauts.",
    "An introduction to machine learning and artificial intelligence.",
    "The history of ancient Rome and its powerful emperors.",
    "Learn Python programming with hands-on projects and examples.",
    "Exploring the wonders of the deep ocean and marine biology."
]

# Step 3: Tokenize the corpus
# BM25 expects tokenized text (split into words)
tokenized_corpus = [doc.lower().split() for doc in corpus]

# Step 4: Initialize the BM25 model
bm25 = BM25Okapi(tokenized_corpus)

# Step 5: Define a user query
query = "space exploration"

# Step 6: Tokenize the query
tokenized_query = query.lower().split()

# Step 7: Get scores for all documents
scores = bm25.get_scores(tokenized_query)

# Step 8: Get the top-k results (e.g., top 3)
top_n = bm25.get_top_n(tokenized_query, corpus, n=3)

# Print results
print("Scores:", scores)
print("Top 3 documents:", top_n)

