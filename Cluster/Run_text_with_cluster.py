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
text_file = r"Detail_99_CleanData_0625_with_language.csv"
df = pd.read_csv(text_file)
df = df[["GPTs_ID", "GPTs_Name", "Description"]]
df = df[df['Primary_Language'] == 'en']
df = df.dropna()

# Load pre-trained BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode corpus into embeddings
corpus = df["Description"].astype(str).tolist()
embeddings = model.encode(corpus)


optimal_clusters = 16

# Clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)

# Adding the cluster labels to the dataframe
df['Cluster'] = labels

# Save the clustered data to a new CSV file
output_file = 'Clustered_Text_Data.csv'
df.to_csv(output_file, index=False)

print(f'Clustered data saved to {output_file}')
