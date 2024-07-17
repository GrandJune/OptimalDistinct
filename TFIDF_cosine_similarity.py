# -*- coding: utf-8 -*-
# @Time     : 7/16/2024 20:32
# @Author   : Junyi
# @FileName: TextAnalysis.py
# @Software  : PyCharm
# Observing PEP 8 coding style
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Sample corpus including the two texts to compare and additional documents
corpus = [
    "A GPT specialized in generating and refining images with a mix of professional and friendly tone.image generator",
    "An application that generates and refines images with a professional tone.image generator",
    "An app for creating and editing images with a professional and friendly tone.image editor",
    "A tool for photo editing and manipulation with an easy-to-use interface.photo editor",
    "Software for designing graphics with a variety of tools.graphic designer",
    "A program to create digital art and illustrations.digital art creator",
    # Add more documents as needed
]


def calculate_similarity_with_corpus(text1, text2, corpus):
    # Append the two texts to compare to the corpus
    corpus_with_texts = corpus + [text1, text2]

    # Fit the vectorizer on the corpus
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus_with_texts)

    # Extract the vectors for the specific texts
    tfidf_text1 = tfidf_matrix[-2]
    tfidf_text2 = tfidf_matrix[-1]

    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_text1, tfidf_text2)
    return similarity[0][0]


# Example texts to compare
text1 = "A GPT specialized in generating and refining images with a mix of professional and friendly tone.image generator"
text2 = "An application that generates and refines images with a professional tone.image generator"

# Calculate similarity
similarity_score = calculate_similarity_with_corpus(text1, text2, corpus)
print(f"Similarity score: {similarity_score}")