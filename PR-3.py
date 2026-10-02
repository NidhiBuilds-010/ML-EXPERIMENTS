import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# High-dimensional data
data = np.array([
    [10, 20, 30, 40],
    [15, 25, 35, 45],
    [20, 30, 40, 50],
    [25, 35, 45, 55],
    [30, 40, 50, 60]
])

print("Original Data:")
print(data)

# Step 1: Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Step 2: Apply PCA
# Reduce 4 dimensions to 2 dimensions
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data_scaled)

# Display reduced data
print("\nReduced Data after PCA:")
print(reduced_data)

# Display explained variance
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)