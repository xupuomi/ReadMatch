import pandas as pd
import sqlalchemy
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Make sure to download NLTK resources
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

def preprocess_text(text):
    if not isinstance(text, str):
        return []
    
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token.lower() for token in tokens if token.isalnum() and token.lower() not in stop_words]
    return filtered_tokens

# Load spaCy model (Elizabeth)
nlp = spacy.load('en_core_web_sm')

# Lemmatization using spaCy (Elizabeth)
def lemmatize_tokens(tokens):
    doc = nlp(' '.join(tokens))
    return [token.lemma_ for token in doc]


def process_books_data(db_path="sqlite:///readmatch.db"):
    engine = sqlalchemy.create_engine(db_path)
    columns = ["title", "genres"]
    
    try:
        query = f"SELECT {', '.join(columns)} FROM books;"
        df = pd.read_sql(query, engine)
        
        processed_data = {}
        for col in columns:
            processed_data[col] = [preprocess_text(doc) for doc in df[col]]
            tokenized = [preprocess_text(doc) for doc in df[col]]               # added by Elizabeth
            lemmatized = [lemmatize_tokens(tokens) for tokens in tokenized]     # added by Elizabeth
            processed_data[col] = lemmatized                                    # added by Elizabeth

 
            
        return processed_data
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# get the processed data
processed_data_dict = process_books_data()

# print
if processed_data_dict:
    for key, value in processed_data_dict.items():
        print(f"Processed Tokens for '{key}':")
        for tokens in value[:5]:
            print(tokens)
        print("\n")
