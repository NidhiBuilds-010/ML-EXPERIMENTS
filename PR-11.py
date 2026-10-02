import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Dataset
X = np.array([
    [1, 2],
    [1, 3],
    [2, 2],
    [8, 8],
    [9, 8],
    [8, 9],
    [4, 5],
    [5, 5],
    [5, 6]
])

# Create K-Means model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# Train the model
kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_

# Get cluster centers
centers = kmeans.cluster_centers_

print("Cluster Labels:")
print(labels)

print("\nCluster Centers:")
print(centers)

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)

# Plot centroids
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker='X',
    s=200
)

plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()