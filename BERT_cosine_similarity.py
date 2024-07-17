# -*- coding: utf-8 -*-
# @Time     : 7/16/2024 21:10
# @Author   : Junyi
# @FileName: BERT_cosine_similarity.py
# @Software  : PyCharm
# Observing PEP 8 coding style
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Load pre-trained BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample corpus
corpus = [
    "A GPT specialized in generating and refining images with a mix of professional and friendly tone.image generator",
    "An application that generates and refines images with a professional tone.image generator",
    "An app for creating and editing images with a professional and friendly tone.image editor",
    "A tool for photo editing and manipulation with an easy-to-use interface.photo editor",
    "Software for designing graphics with a variety of tools.graphic designer",
    "A program to create digital art and illustrations.digital art creator",
    # Add more documents as needed
]

# Encode the corpus to get embeddings
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

def calculate_similarity_bert(text1, text2, model):
    # Encode the individual texts
    embeddings = model.encode([text1, text2], convert_to_tensor=True)

    # Calculate cosine similarity
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])

    return similarity.item()

# Example texts to compare
text1 = "A GPT specialized in generating and refining images with a mix of professional and friendly tone.image generator"
text2 = "An application that generates and refines images with a professional tone.image generator"

# Calculate similarity
similarity_score = calculate_similarity_bert(text1, text2, model)
print(f"Similarity score: {similarity_score}")

# If you want to calculate similarities for all pairs in the corpus:
all_similarities = util.pytorch_cos_sim(corpus_embeddings, corpus_embeddings)

# Convert the similarity matrix to a pandas DataFrame for better visualization
df_similarities = pd.DataFrame(all_similarities.numpy(), index=corpus, columns=corpus)

# Display the similarity DataFrame
print(df_similarities)
