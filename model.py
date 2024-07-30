import numpy as np
from sklearn.linear_model import LinearRegression
import joblib


X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 2, 3, 4, 5])

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'linear_regression_model.pkl')