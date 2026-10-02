# Import libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import tensorflow as tf

# -------------------------
# 1. NumPy
# -------------------------

data = np.array([10, 20, 30, 40, 50])

print("NumPy Array:")
print(data)

print("Mean using NumPy:", np.mean(data))
print("Sum using NumPy:", np.sum(data))


# -------------------------
# 2. Pandas
# -------------------------

df = pd.DataFrame({
    "Name": ["A", "B", "C", "D"],
    "Marks": [80, 75, 90, 85]
})

print("\nPandas DataFrame:")
print(df)

print("\nAverage Marks:")
print(df["Marks"].mean())


# -------------------------
# 3. Scikit-learn
# -------------------------

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[6]])

print("\nScikit-learn Prediction:")
print(prediction)


# -------------------------
# 4. TensorFlow
# -------------------------

tensor = tf.constant([10, 20, 30, 40, 50])

print("\nTensorFlow Tensor:")
print(tensor)

print("\nTensorFlow Mean:")
print(tf.reduce_mean(tensor))