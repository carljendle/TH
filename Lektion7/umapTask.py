import pandas as pd
import umap


#Ladda in data 
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import matplotlib.pyplot as plt

#Går att ändra test till train eller all baserat på hur mycket data du vill ladda in
dataset = fetch_20newsgroups(subset='test',
                             shuffle=True, random_state=42)

print(f'{len(dataset.data)} documents')
print(f'{len(dataset.target_names)} categories')

#Todo - Gör en Bag of Words-representation av varje dokument med hjälp av count vectorizer
#Använd umap för att skapa en 2D-representation med färgkodade labels för dokumenten och plotta.


#Testa sedan att ta bort alla ord som förekommer färre än 5 gånger i datasetet.