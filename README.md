# ReadMatch
ReadMatch is a book recommendation system that helps readers discover their next favorite book by understanding what they’re really looking for.

Instead of relying on simple filters or exact word matches, ReadMatch combines classic information retrieval with modern NLP techniques to recommend books based on meaning, context, and user preferences.

Whether you’re searching by a book you love or a genre you’re in the mood for, ReadMatch finds recommendations that actually make sense based on your taste!

### How it works
ReadMatch uses a hybrid recommendation pipeline:
* BM25 (Lexical Retrieval)
    * Captures exact and partial keyword matches across book titles, authors, genres, and descriptions.

* SBERT Semantic Similarity
    * Understands the meaning behind queries and book descriptions, allowing the system to match related ideas even when the wording differs.

* User Profiles & Preferences
    * Learns from user interactions (liked books and ratings) to personalize recommendations over time.



# Set Up
### Frontend
1. Enter the frontend folder using ```cd frontend```
2. Use ```npm i``` to install the dependencies
3. Use ```npm run start``` to launch the frontend

### Backend
1. Enter the backend folder using ```cd backend```
2. Use ```pip install flask flask-cors numpy rank-bm25 nltk sentence-transformers sqlalchemy``` to install the dependencies
3. Use ```python3 loader.py``` to create the books table
4. Use ```python3 generate_embeddings.py``` to generate the book embeddings
5. Use ```python3 app.py``` to run the backend

### Enjoy!