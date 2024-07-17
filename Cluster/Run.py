# -*- coding: utf-8 -*-
# @Time     : 7/17/2024 19:49
# @Author   : Junyi
# @FileName: Run.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


# text_file = r"E:\GPT\Detail_99_CleanData_0625.csv"
text_file = r"Detail_99_CleanData_0625.csv"
df = pd.read_csv(text_file)
df = df[["GPTs_ID", "GPTs_Name", "Description"]]

# Load pre-trained BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example corpus
# corpus = [
#     "A GPT specialized in generating and refining images with a mix of professional and friendly tone.image generator",
#     "An application that generates and refines images with a professional tone.image generator",
#     "An app for creating and editing images with a professional and friendly tone.image editor",
#     "A tool for photo editing and manipulation with an easy-to-use interface.photo editor",
#     "Software for designing graphics with a variety of tools.graphic designer",
#     "A program to create digital art and illustrations.digital art creator",
#     # Add more documents as needed
# ]
corpus = df["Description"].tolist()


# Encode corpus into embeddings
embeddings = model.encode(corpus)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
range_clusters = range(5, 20)  # Test from 1 to 10 clusters

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embeddings)
    inertia.append(kmeans.inertia_)

# Plot the inertia to visualize the Elbow Method
plt.figure(figsize=(8, 4))
plt.plot(range_clusters, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()


from sklearn.metrics import silhouette_score

# Determine the optimal number of clusters using the Silhouette Score
silhouette_scores = []

for k in range_clusters:  # Silhouette score is not defined for 1 cluster
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    silhouette_scores.append(score)

# Plot the Silhouette Scores
plt.figure(figsize=(8, 4))
plt.plot(range_clusters, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal Number of Clusters')
plt.show()

# Optimal number of clusters
optimal_clusters = range_clusters[np.argmax(silhouette_scores)]
print(f'Optimal number of clusters: {optimal_clusters}')