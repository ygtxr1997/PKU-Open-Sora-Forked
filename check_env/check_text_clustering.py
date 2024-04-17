import os
import json

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


json_path = "/public/home/201810101923/datasets/opensora/dataset_v1.0.0_tmptest_sorted/sharegpt4v_path_cap_64x512x512.json"
with open(json_path, "r") as f:
    json_data = json.load(f)

captions = [str(x["cap"][0]) for x in json_data]
print(len(captions), captions[0])


dataset = captions

# Vectorize the dataset
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset)

# Define the number of clusters
k = 2

# Create a k-means model and fit it to the data
km = KMeans(n_clusters=k)
km.fit(X)

# Predict the clusters for each document
y_pred = km.predict(X)

# Print the cluster assignments
for i in range(10):
    print(y_pred[i], dataset[i][:30])
