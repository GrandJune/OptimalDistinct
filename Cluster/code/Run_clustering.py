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
import warnings
import os
# Suppress NVML warning
warnings.filterwarnings("ignore", message="Can't initialize NVML")
# Set TOKENIZERS_PARALLELISM environment variable as false to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load and preprocess data
text_file = r"sbert_similarity_results_allcat_with_language.csv"
# text_file_list = ["Detail_99_CleanData_0507.csv", "Detail_99_CleanData_0514.csv", "Detail_99_CleanData_0521.csv",
#                   "Detail_99_CleanData_0528.csv", "Detail_99_CleanData_0604.csv", "Detail_99_CleanData_0611.csv",
#                   "Detail_99_CleanData_0618.csv", "Detail_99_CleanData_0625.csv"]
# text_file_list_2 = [each[:-4] + "_with_language.csv" for each in text_file_list]
# text_file_list_2 = [r"./data/" + each for each in text_file_list_2]

# for index, text_file in enumerate(text_file_list_2):
df = pd.read_csv(text_file)
df = df[df['Primary_Language'] == 'en']
# df = df.dropna()

model = SentenceTransformer('all-MiniLM-L6-v2')
optimal_clusters = 30
# Description
corpus = df["Description"].astype(str).tolist()
embeddings = model.encode(corpus)
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)
df['Description_Cluster'] = labels
# Feature
corpus = df["Features"].astype(str).tolist()
embeddings = model.encode(corpus)
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)
df['Features_Cluster'] = labels
# Conversion
corpus = df["Conversion_start"].astype(str).tolist()
embeddings = model.encode(corpus)
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)
df['Conversion_Cluster'] = labels

# Save the clustered data to a new CSV file
# output_file = r"./data/" + text_file_list[index][:-4] + "_with_cluster.csv"
output_file = text_file[:-4] + "_with_cluster.csv"
df.to_csv(output_file, index=False)
