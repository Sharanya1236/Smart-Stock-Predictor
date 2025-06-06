import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

def predict_stock_price(data, days_ahead):
    # Prepare data
    data = data.reset_index()
    data['Day'] = np.arange(len(data))
    X = data[['Day']]
    y = data['Close']

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Predict future day
    future_day = np.array([[len(data) + days_ahead]])
    pred = model.predict(future_day)
    return pred[0]
