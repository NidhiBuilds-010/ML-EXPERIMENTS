import numpy as np

# Dataset with 2 classes
class1 = np.array([
    [1, 2],
    [2, 3],
    [3, 2]
])

class2 = np.array([
    [7, 8],
    [8, 9],
    [9, 8]
])

# Combine the data
X = np.vstack((class1, class2))

# Calculate overall mean
overall_mean = np.mean(X, axis=0)

# Calculate class means
mean1 = np.mean(class1, axis=0)
mean2 = np.mean(class2, axis=0)

# Within-class scatter
S_W = np.zeros((2, 2))

for x in class1:
    diff = (x - mean1).reshape(2, 1)
    S_W += diff @ diff.T

for x in class2:
    diff = (x - mean2).reshape(2, 1)
    S_W += diff @ diff.T

# Between-class scatter
n1 = len(class1)
n2 = len(class2)

diff1 = (mean1 - overall_mean).reshape(2, 1)
diff2 = (mean2 - overall_mean).reshape(2, 1)

S_B = n1 * (diff1 @ diff1.T) + n2 * (diff2 @ diff2.T)

# Total scatter
S_T = S_W + S_B

print("Overall Mean:")
print(overall_mean)

print("\nWithin-Class Scatter:")
print(S_W)

print("\nBetween-Class Scatter:")
print(S_B)

print("\nTotal Scatter:")
print(S_T)