# -*- coding: utf-8 -*-
# @Time     : 7/18/2024 20:34
# @Author   : Junyi
# @FileName: Run_text_with_cluster.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load and preprocess data
text_file = r"Detail_99_CleanData_0625.csv"
df = pd.read_csv(text_file)
df = df[["GPTs_ID", "GPTs_Name", "Description"]]
df = df.dropna()

# Load pre-trained BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode corpus into embeddings
corpus = df["Description"].astype(str).tolist()
embeddings = model.encode(corpus)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
range_clusters = range(5, 40)

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
plt.savefig('Inertia.png')
plt.show()

# Use the Kneedle algorithm to find the elbow point
kneedle = KneeLocator(range_clusters, inertia, curve='convex', direction='decreasing')
elbow_optimal_clusters = kneedle.elbow
print(f'Elbow Method Optimal number of clusters: {elbow_optimal_clusters}')

# Determine the optimal number of clusters using the Silhouette Score
silhouette_scores = []

for k in range_clusters:
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
plt.savefig('SilhouetteScore.png')
plt.show()

# Optimal number of clusters based on Silhouette Score
silhouette_optimal_clusters = range_clusters[np.argmax(silhouette_scores)]
print(f'Silhouette Score Optimal number of clusters: {silhouette_optimal_clusters}')

# Choose the optimal number of clusters (You can decide based on either method or combine them)
optimal_clusters = silhouette_optimal_clusters
print(f'Chosen Optimal number of clusters: {optimal_clusters}')

# Clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)

# Adding the cluster labels to the dataframe
df['Cluster'] = labels

# Save the clustered data to a new CSV file
output_file = 'Clustered_Text_Data.csv'
df.to_csv(output_file, index=False)

print(f'Clustered data saved to {output_file}')
