#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Feature (e.g., house size)
y = np.array([150, 200, 250, 300, 350])  # Target (e.g., house price)

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

plt.scatter(X, y, color='blue')  # Actual data
plt.plot(X, y_pred, color='red')  # Predicted data
plt.xlabel('House Size')
plt.ylabel('House Price')
plt.show()
