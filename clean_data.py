import pandas as pd
import math
import numpy as np
# from langdetect import detect
import re
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(base_dir, "Books_df.csv")

df = pd.read_csv(csv_file_path)

# get rid of price column
df.drop(columns=['Price'], inplace=True)

# remove duplicate entries 
df.drop_duplicates()

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

# combine main genre + sub genre columns
df['Genre'] = df['Main Genre'] + ' ' + df['Sub Genre']
df = df.drop(columns=['Main Genre'])
df = df.drop(columns=['Sub Genre'])

# properly reorder columns
order = df.columns.tolist()
order.remove('Genre')
order.insert(3, 'Genre')

df = df[order]

# upload to new csv vile 
df.to_csv('cleaned_data.csv', index=False)
