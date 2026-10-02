import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

# Two data points
A = np.array([[2, 3, 4]])
B = np.array([[3, 4, 5]])

# Similarity measure - Cosine Similarity
similarity = cosine_similarity(A, B)

# Dissimilarity measure - Euclidean Distance
dissimilarity = euclidean(A[0], B[0])

# Display results
print("Data Point A:", A[0])
print("Data Point B:", B[0])

print("\nSimilarity Measure (Cosine Similarity):")
print(similarity[0][0])

print("\nDissimilarity Measure (Euclidean Distance):")
print(dissimilarity)