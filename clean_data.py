import pandas as pd
import math
import numpy as np
# from langdetect import detect
import re
import os
from collections import OrderedDict

def clean_title(title):
    # Convert to lowercase
    title = str(title).lower()
    
    # Remove everything after common subtitle separators (:, -, |)
    title = re.split(r'[:—\-|]', title)[0].strip()
    
    # Remove common book descriptions/identifiers (e.g., 'ebook', 'bestseller')
    title = re.sub(r'\b(the|a|an|copy|ebook|bestseller|novel|edition|paperback|hardcover)\b', '', title)
    
    # Remove extra spaces
    title = re.sub(r'\s+', ' ', title).strip()
    
    return title

def standardize_author(name):
    name_parts = str(name).lower().split()
    return ' '.join(sorted(name_parts))

def clean_genre(genre): 
    genre = str(genre)
    genre = genre.replace(' and ', ' ')
    
    genre = re.sub(r'[^a-zA-Z0-9\s]', ' ', genre)
    words = [word.strip() for word in genre.split()]
    
    # genre = re.sub(r'[&,]', ' ', genre).strip()
    # genre = genre.split()
    
    unique_words = list(OrderedDict.fromkeys(words))
    return ", ".join(unique_words) 

    # genre = set(OrderedDict.fromkeys(genre)) 
    # genre = list(genre) 
    # genre = " ".join(genre)
    # return genre




base_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(base_dir, "Books_df.csv")

df = pd.read_csv(csv_file_path)
df.columns = df.columns.str.strip()

# get rid of price column
df.drop(columns=['Price'], inplace=True)

# remove exact duplicate entries 
df.drop_duplicates(subset=['Title'], inplace=True)
df['Clean_Title'] = df['Title'].apply(clean_title)
df['Clean_Author'] = df['Author'].apply(standardize_author)

df.drop_duplicates(subset=['Clean_Title'], inplace=True)
df.drop_duplicates(subset=['Clean_Author'], inplace=True)


# remove unneeded types (not actually books)
df = df[df['Type'] != 'Map']
df = df[df['Type'] != 'Game']
df = df[df['Type'] != 'Calendar']
df = df[df['Type'] != 'Stationery']
df = df[df['Type'] != 'Workbook']
df = df[df['Type'] != 'Diary']
df = df[df['Type'] != 'Misc. Supplies']
df = df[df['Type'] != 'Cards']
df = df[df['Type'] != 'Poster']
df = df[df['Type'] != 'Gift']
df = df[df['Type'] != 'Single Issue Magazine']
df = df[df['Type'] != 'Puzzle']
df = df[df['Type'] != 'Wall Chart']
df = df[df['Type'] != 'Sheet music']
df = df[df['Type'] != 'Card Book']
# df = df[df['Type'] != 'Kindle Edition']


df.sort_values(by='No. of People rated', ascending=False, inplace=True)
df = df.iloc[:300]

# combine main genre + sub genre columns
df['Genre'] = df['Main Genre'] + ' ' + df['Sub Genre']
df = df.drop(columns=['Main Genre'])
df = df.drop(columns=['Sub Genre'])

# properly reorder columns
order = df.columns.tolist()
order.remove('Genre')
order.remove('Clean_Title')
order.remove('Clean_Author')

order.insert(3, 'Genre')
order.insert(1, 'Clean_Title')
order.insert(2, 'Clean_Author')

df = df[order]

df['Genre'] = df['Genre'].apply(clean_genre)

# upload to new csv vile 
df.to_csv('cleaned_data.csv', index=False)
