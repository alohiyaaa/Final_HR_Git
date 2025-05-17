# Databricks notebook source
# MAGIC %pip install --upgrade threadpoolctl
# MAGIC %pip install umap-learn
# MAGIC %pip install kneed

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import numpy as np
import pandas as pd
import collections
from collections import Counter
import ast
import re
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import pairwise_distances

# COMMAND ----------

CHUNK_FILE_PATH = ' ' #Initial chunk file path to use for clustering
OUTPUT_PARQUET_FILE_PATH = ' ' #output filepath name to be written to
Best_K = 0          # best K value will be determined as part of the KMeans algorithm
BP_PERCENT = 0.10    # Boundary point percent, take top 10% of boundary points

# COMMAND ----------

#runs the actual KMeans algorithm
def run_kmeans_pipeline(embeddings, input_data):
    df_results = []

    #KMeans cluster range to try
    cluster_range = find_kemeans_range(embeddings)

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        #Always work on a copy, don't modify original data
        datac = input_data.iloc[:len(labels)].copy()
        datac['Cluster_Label'] = labels
        datac['Chunk'] = input_data['chunk']
        datac['Source'] = input_data['source']
        datac['Document'] = input_data['document']

        inertia = kmeans.inertia_
        sil_score = silhouette_score(embeddings, labels)
        dbi_score = davies_bouldin_score(embeddings, labels)
        sil_coeff = silhouette_samples(embeddings, labels)

        filenames_by_label = {
            label: list(zip(sub['source'], sub['chunk']))
            for label, sub in datac.groupby('Cluster_Label')
        }

        # Store one row per k run.
        row = {
            'K': k,
            'Cluster_Label': labels,
            'Source': datac['Source'].tolist(),
            'Chunk': datac['Chunk'].tolist(),
            'Document': datac['Document'].tolist(),
            'Embeddings': embeddings,  # Keep full np.ndarray here
            'Filenames_BY_Label': filenames_by_label,
            'Cluster_Centroids': kmeans.cluster_centers_,
            'Silhouette_Score': sil_score,
            'DB_Index': dbi_score,
            'Inertia': inertia,
            'Silhouette_Coeff': sil_coeff
        }

        df_results.append(row)

    return pd.DataFrame(df_results)

# COMMAND ----------

def analyze_kmeans_results(df_results):

    elbow_found = False
    k_values = df_results['K']
    silhouette = df_results['Silhouette_Score']
    inertia = df_results['Inertia']
    # Elbow method on inertia
    try:
        knee = KneeLocator(k_values, inertia, curve='convex', direction='decreasing')
        best_k_elbow = round(knee.knee)
        #print(f" Elbow found via KneeLocator: k = {best_k_elbow}")
        elbow_found = True
    except:
        best_k_elbow = None

    if not elbow_found:
        inertia_diff = np.diff(inertia)
        second_diff = np.diff(inertia_diff)
        if len(second_diff) > 0:
            idx = np.argmax(second_diff) + 2  
            best_k_elbow = k_values.iloc[idx]
        else:            
            best_k_elbow = None

    total_rows = len(df_results)
    n_top = max(1, int(total_rows * BP_PERCENT))

    if best_k_elbow is None:
        top_silhouette = set(df_results.nlargest(n_top, 'Silhouette_Score')['K'])
    
        best_row_sil = df_results.sort_values('Silhouette_Score', ascending=False).iloc[0]
        best_k_sil = best_row_sil['K']

    summary_df = df_results.sort_values(by='Silhouette_Score', ascending=False)
    k_val = summary_df.iloc[0]['K']

    return summary_df, k_val

# COMMAND ----------


def find_kemeans_range(embeddings):
    data_len = embeddings.shape[0]

    # max_k by formula
    max_k = math.ceil(math.sqrt(data_len / 2))
    max_k = max(2, max_k)  # At least allow up to 2 clusters minimum

    # estimate some min_k
    if max_k <= 20:
        min_k = 2
        k_inc = 1
    else:
        min_k = 9  # If you have a decent amount of data, start from 9 clusters
        k_inc = 3

    # Check and adjust if min_k >= max_k
    if min_k >= max_k:
        min_k = max_k

    # Now define the range safely
    k_range = list(range(min_k, max_k, k_inc))
    return k_range

# COMMAND ----------

# For the selected best k, extract labels, centroids, and construct a 
# chunk-level DataFrame for downstream processing
def prepare_best_k_dataframe(df_results, best_k):
    best_row = df_results[df_results['K'] == Best_K].squeeze()

    # Unpack row values
    labels = best_row['Cluster_Label']
    embeddings = best_row['Embeddings']
    chunks = best_row['Chunk']
    sources = best_row['Source']
    sil_coeffs = best_row['Silhouette_Coeff']
    cluster_label = best_row['K']
    centroids = best_row['Cluster_Centroids']
    filenames_by_label = best_row['Filenames_BY_Label']
    documents = best_row['Document']

    # Convert to per-chunk rows and save in chunk_df
    rows = []
    for i in range(len(labels)):
        rows.append({
            'K': cluster_label,
            'Source': sources[i],
            'Document': documents[i],
            'Chunk': chunks[i],
            'Cluster_Label': labels[i],
            'Silhouette_Coeff': sil_coeffs[i],
            'Embedding': embeddings[i].tolist(), 
            'Cluster_Centroids': centroids.tolist(),
        })

    #per chunk df
    chunk_df = pd.DataFrame(rows)
    return chunk_df, labels, centroids


# COMMAND ----------

#Fix Embeddings column from strings to numpy arrays
def parse_embedding_column(col):
    sample = col.iloc[0]
    if isinstance(sample, (list, np.ndarray)):
        return col.apply(np.array)
    
    def fix_and_parse(s):
        if isinstance(s, (list, np.ndarray)):
            return np.array(s)
        if "array(" in s:
            s = s.replace("array(", "[").replace(")", "]")
        s = re.sub(r'(?<=[0-9])\s+(?=[\-0-9])', ', ', s.strip())
        return np.array(ast.literal_eval(s))
    
    return col.apply(fix_and_parse)

# COMMAND ----------

#The code here in this block is used to compute the top N% boundary points for each cluster.
def compute_centroid_distance(chunk_df, embeddings, labels, centroids):

    best_row = df_results[df_results['K'] == Best_K].iloc[0]
    labels = best_row['Cluster_Label']         # np.array of cluster labels for each embedding
    centroids = best_row['Cluster_Centroids']  # list of np.array vectors
    embeddings = np.array(best_row['Embeddings'].tolist())  # embeddings as np.array

    if isinstance(centroids, list):
        centroids = np.vstack(centroids)  # (k, embedding_dim)

    #Fix embedding first
    chunk_df['Embedding'] = parse_embedding_column(chunk_df['Embedding'])
    chunk_df['Cluster_Label'] = labels
    assigned_distances = pairwise_distances(embeddings, centroids)[range(len(labels)), labels]
    chunk_df['Distance_To_Center'] = assigned_distances

    # Initialize
    chunk_df['Is_Boundary_Point'] = 0
    return chunk_df

#  Utility Function to determine Second-Closest Centroid
def get_second_closest_centroid(embedding, centroids, labels):
    distances = np.linalg.norm(centroids - embedding, axis=1)
    sorted_indices = np.argsort(distances)
    second_idx = sorted_indices[1]  
    return centroids[second_idx], labels[second_idx], distances[second_idx]

# For each boundary point, calculate and store its second-closest cluster 
# centroid and distance
# Adds three new columns: Second_Closest_Centroid, 
# Second_Closest_Cluster, Second_Closest_Cluster_Dist
#
def mark_boundary_points(chunk_df, centroids, labels):
    chunk_df["Second_Closest_Centroid"] = None
    chunk_df["Second_Closest_Cluster"] = None
    chunk_df["Second_Closest_Cluster_Dist"] = None

    for idx, row in chunk_df.iterrows():
        embedding = row['Embedding']
        assert isinstance(embedding, np.ndarray), f"Row {idx}: Embedding is not a numpy array!"
    
        second_centroid, second_label, second_distance = get_second_closest_centroid(
                                        embedding, centroids, np.unique(labels))
        chunk_df.at[idx, "Second_Closest_Centroid"] = second_centroid
        chunk_df.at[idx, "Second_Closest_Cluster"] = second_label
        chunk_df.at[idx, "Second_Closest_Cluster_Dist"] = second_distance

    sort_chunk_df = chunk_df.sort_values(by='Distance_To_Center', ascending=False)
    top_10_percent = int(BP_PERCENT * len(sort_chunk_df))
    top_10_percent_samples = sort_chunk_df.head(top_10_percent)

    return chunk_df, top_10_percent_samples
    

# COMMAND ----------

def save_final_dataframe(chunk_df):
    chunk_df.to_parquet(OUTPUT_PARQUET_FILE_PATH, index=False)
    print(f" Final dataframe saved at: {OUTPUT_PARQUET_FILE_PATH}")


# COMMAND ----------

try:
    data = pd.read_csv(CHUNK_FILE_PATH)
    print(f"Reading file {CHUNK_FILE_PATH}")
except FileNotFoundError:
    print(f"Error: CSV file not found at {CHUNK_FILE_PATH}")
    pass

# Using list comprehension to handle the conversion
# Convert string to list of floats
data['embedding'] = data['embedding'].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))  

# Convert the 'embedding' column to a NumPy array
embeddings = np.array(data['embedding'].tolist())

print("running kmeans pipeline")
df_results = run_kmeans_pipeline(embeddings, data)

print("selecting K value")
candidate_summary_df, Best_K = analyze_kmeans_results(df_results)

print("Marking boundary points")
chunk_df, labels, centroids = prepare_best_k_dataframe(df_results, Best_K)
chunk_df = compute_centroid_distance(chunk_df, embeddings, labels, centroids)
chunk_df, top_10_percent_samples = mark_boundary_points(chunk_df, centroids, labels)

#This is important to save in parquet format for the LLM part
print("Saving final dataframe")
save_final_dataframe(chunk_df)


# COMMAND ----------


