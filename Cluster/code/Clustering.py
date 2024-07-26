# -*- coding: utf-8 -*-
# @Time     : 7/16/2024 20:39
# @Author   : Junyi
# @FileName: Clustering.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import re

# Download necessary NLTK data files
nltk.download('stopwords')


def preprocess_text(text):
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize and remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word.lower() not in stop_words]

    # Stem words
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word) for word in words]

    return ' '.join(stemmed_words)


# Sample data
texts = [
    "A GPT specialized in generating and refining images with a mix of professional and friendly tone.image generator",
    "An application that generates and refines images with a professional tone.image generator",
    "An app for creating and editing images with a professional and friendly tone.image editor",
    "A tool for photo editing and manipulation with an easy-to-use interface.photo editor",
    "Software for designing graphics with a variety of tools.graphic designer",
    "A program to create digital art and illustrations.digital art creator"
]

# Preprocess the texts
processed_texts = [preprocess_text(text) for text in texts]

# Convert texts to TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_texts)


# Elbow Method to determine optimal number of clusters
#  at the point where the inertia (sum of squared distances to the nearest cluster center) starts to diminish at a slower rate.
def elbow_method(tfidf_matrix, max_k=4):
    inertias = []
    K = range(1, max_k + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(tfidf_matrix)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.show()


elbow_method(tfidf_matrix)


# Silhouette Score to determine optimal number of clusters
# The optimal k is the one with the highest Silhouette Score.
def silhouette_method(tfidf_matrix, max_k=4):
    silhouette_scores = []
    K = range(2, max_k + 1)  # Silhouette score is undefined for k=1
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(tfidf_matrix)
        score = silhouette_score(tfidf_matrix, kmeans.labels_)
        silhouette_scores.append(score)

    plt.figure(figsize=(8, 4))
    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Method For Optimal k')
    plt.show()


silhouette_method(tfidf_matrix)

