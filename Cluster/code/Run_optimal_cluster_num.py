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
import warnings
from kneed import KneeLocator
import os
# Suppress NVML warning
warnings.filterwarnings("ignore", message="Can't initialize NVML")

# Set TOKENIZERS_PARALLELISM environment variable to false to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

text_file = r"sbert_similarity_results_allcat_with_language.csv"
# text_file_list = ["Detail_99_CleanData_0507.csv", "Detail_99_CleanData_0514.csv", "Detail_99_CleanData_0521.csv",
#                   "Detail_99_CleanData_0528.csv", "Detail_99_CleanData_0604.csv", "Detail_99_CleanData_0611.csv",
#                   "Detail_99_CleanData_0618.csv", "Detail_99_CleanData_0625.csv"]
# text_file_list = [each[:-4] + "_with_language.csv" for each in text_file_list]
# text_file_list = [r"./data/" + each for each in text_file_list]

df = pd.read_csv(text_file)
# print(df.isna().sum()) # there are many missing value in URL, Website, Linkedin, and so on
# Load pre-trained BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Description  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
corpus_description = df["Description_cleaned"].astype(str).tolist()

embeddings = model.encode(corpus_description)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
range_clusters = range(10, 100)  # Test from 1 to 10 clusters

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embeddings)
    inertia.append(kmeans.inertia_)

# Plot the inertia to visualize the Elbow Method
plt.figure(figsize=(8, 4))
plt.plot(range_clusters, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Description')
plt.savefig('Inertia_Description.png')
# plt.show()

# Use the Kneedle algorithm to find the elbow point
kneedle = KneeLocator(range_clusters, inertia, curve='convex', direction='decreasing')
optimal_clusters = kneedle.elbow
print('Elbow, Optimal number of Description: {0}'.format(optimal_clusters))

from sklearn.metrics import silhouette_score

# Determine the optimal number of clusters using the Silhouette Score
# How similar an object is to its own cluster compared to other clusters.
# Silhouette Score close to +1: The clusters are well separated, and each sample is appropriately clustered.
# Silhouette Score close to 0: The clusters are overlapping, and samples are very close to the decision boundary between clusters. It indicates that the clusters are not well-defined.
# Silhouette Score close to -1: Samples may have been assigned to the wrong clusters.
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
plt.title('Silhouette Score for Description')
plt.savefig('SilhouetteScore_Description.png')
# plt.show()

# Optimal number of clusters
optimal_clusters = range_clusters[np.argmax(silhouette_scores)]
print('Silhouette, Optimal number of Description: {0}'.format(optimal_clusters))

# Features  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
corpus_feature = df["Features_cleaned"].astype(str).tolist()

embeddings = model.encode(corpus_feature)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
range_clusters = range(10, 100)  # Test from 1 to 10 clusters

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embeddings)
    inertia.append(kmeans.inertia_)

# Plot the inertia to visualize the Elbow Method
plt.figure(figsize=(8, 4))
plt.plot(range_clusters, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Features')
plt.savefig('Inertia_Features.png')
# plt.show()

# Use the Kneedle algorithm to find the elbow point
kneedle = KneeLocator(range_clusters, inertia, curve='convex', direction='decreasing')
optimal_clusters = kneedle.elbow
print('Elbow, Optimal number of Features: {0}'.format(optimal_clusters))

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
plt.title('Silhouette Score for Features')
plt.savefig('SilhouetteScore_Features.png')
# plt.show()

# Optimal number of clusters
optimal_clusters = range_clusters[np.argmax(silhouette_scores)]
print('Silhouette, Optimal number of Features: {0}'.format(optimal_clusters))

# Conversion Start !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
corpus_conversion = df["Conversion_start_cleaned"].astype(str).tolist()

embeddings = model.encode(corpus_conversion)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
range_clusters = range(10, 100)  # Test from 1 to 10 clusters

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embeddings)
    inertia.append(kmeans.inertia_)

# Plot the inertia to visualize the Elbow Method
plt.figure(figsize=(8, 4))
plt.plot(range_clusters, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Conversion')
plt.savefig('Inertia_Conversion.png')
plt.show()

# Use the Kneedle algorithm to find the elbow point
kneedle = KneeLocator(range_clusters, inertia, curve='convex', direction='decreasing')
optimal_clusters = kneedle.elbow
print('Elbow, Optimal number of Conversion: {0}'.format(optimal_clusters))

from sklearn.metrics import silhouette_score

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
plt.title('Silhouette Score for Conversion')
plt.savefig('SilhouetteScore_Conversion.png')
# plt.show()

# Optimal number of clusters
optimal_clusters = range_clusters[np.argmax(silhouette_scores)]
print('Silhouette, Optimal number of Conversion: {0}'.format(optimal_clusters))
