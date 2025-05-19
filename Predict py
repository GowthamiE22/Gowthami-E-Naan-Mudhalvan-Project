import numpy as np
import pandas as pd
from keras.models import load_model
import joblib

# Load model and scaler
model = load_model('model/lstm_model.h5')
scaler = joblib.load('model/scaler.save')

# Load data
df = pd.read_csv('data/stock_data.csv')
data = df[['Close']].values
scaled_data = scaler.transform(data)

last_60_days = scaled_data[-60:]
X_test = np.reshape(last_60_days, (1, 60, 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)

print(f"Predicted Next Day Price: {pred_price[0][0]:.2f}")
