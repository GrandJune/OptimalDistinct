# -*- coding: utf-8 -*-
# @Time     : 7/28/2024 17:11
# @Author   : Junyi
# @FileName: Neighboring_distance.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer

text_file = r"sbert_similarity_results_allcat_with_cluster.csv"
df = pd.read_csv(text_file)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings for all texts
df['embedding'] = df['Description_cleaned'].apply(lambda x: model.encode(x))


# Function to find superior texts and calculate average distance
def calculate_average_distance(df):
    avg_distances = []
    for i, row in df.iterrows():
        focal_cluster = row['Description_Cluster']
        focal_performance = row['Conversions']
        focal_embedding = row['embedding']

        # Calculate 90th percentile for the current cluster
        cluster_df = df[df['Description_Cluster'] == focal_cluster]
        threshold = cluster_df['Conversions'].quantile(0.90)

        # Find texts in the same cluster with performance in the top 10%
        superior_texts = cluster_df[cluster_df['Conversions'] > threshold]

        if not superior_texts.empty:
            # Calculate distances
            superior_embeddings = list(superior_texts['embedding'])
            distances = cosine_distances([focal_embedding], superior_embeddings)[0]
            avg_distance = distances.mean()
        else:
            avg_distance = None

        avg_distances.append(avg_distance)

    return avg_distances


df['avg_distance'] = calculate_average_distance(df)
df = df.drop(columns=['embedding'])
df.to_csv('data_with_avg_distance.csv', index=False)
# print(df)
