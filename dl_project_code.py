# -*- coding: utf-8 -*-
"""DL_Project_Code.ipynb

# **Problem Statement 1**: Agricultural commodity price prediction

# Importing necessary libraries and modules
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, GRU, Dropout, Input,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam

"""# Data Preprocessing and Feature Engineering


*   Load and prepare the dataset

  * Feature Engineering: Creating Lagged, Rolling, and Technical Indicators emphasizes the core activity of adding meaningful features.

"""

#unzip the csv file and load the dataset
!unzip /content/Price_Agriculture_commodities_Week.csv.zip

df = pd.read_csv('Price_Agriculture_commodities_Week.csv')

# Convert date
df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], format='%d-%m-%Y')

# Select one commodity (example: Onion)
commodity = "Onion"
df = df[df['Commodity'] == commodity].sort_values("Arrival_Date")

# EXISTING FEATURES
df['Price_Spread'] = df['Max Price'] - df['Min Price']
df['MA_4'] = df['Modal Price'].rolling(4).mean()
df['MA_8'] = df['Modal Price'].rolling(8).mean()
df['EMA_4'] = df['Modal Price'].ewm(span=4).mean()
df['Momentum'] = df['Modal Price'] - df['Modal Price'].shift(4)

# NEW: LAG FEATURES (Historical context)
#Lags (1, 2, 3): Added lag_1, lag_2, and lag_3.
# This lets the model "remember" the specific price points from the previous three weeks directly, which is a strong signal for the next week's price.
df['lag_1'] = df['Modal Price'].shift(1)
df['lag_2'] = df['Modal Price'].shift(2)
df['lag_3'] = df['Modal Price'].shift(3)

# NEW: ROLLING STATISTICS (Trend & Volatility)
#Rolling Statistics: Added a 7-week rolling mean and standard deviation.
#The mean helps the model identify the broad trend, while the standard deviation helps it understand market volatility.
df['rolling_mean_7'] = df['Modal Price'].rolling(window=7).mean()
df['rolling_std_7'] = df['Modal Price'].rolling(window=7).std()

# Smooth prediction target
df['Target'] = df['Modal Price'].rolling(4).mean()

# Drop rows with NaNs created by shifting/rolling
df.dropna(inplace=True)

# Log transformation to stabilize variance
df['Target'] = np.log1p(df['Target'])

# Updated features list
features_cols = [
    'Target', 'Min Price', 'Max Price', 'Price_Spread',
    'MA_4', 'MA_8', 'EMA_4', 'Momentum',
    'lag_1', 'lag_2', 'lag_3', 'rolling_mean_7', 'rolling_std_7'
]

target_col = 'Target'

# Handle missing values
df.ffill(inplace=True)

df.head()

"""# Exploratory Data Analysis: Price Trend and Distribution

*   **Data Visualization**


"""

plt.figure(figsize=(10,5))
plt.plot(df['Arrival_Date'], df['Modal Price'], label='Modal Price')
plt.title(f"{commodity} Price Trend")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

sns.boxplot(data=df[['Min Price', 'Max Price', 'Modal Price']])
plt.title("Price Distribution")
plt.show()

"""# Feature Scaling

*   Using MinMaxScaler


"""

# Import the MinMaxScaler class from the sklearn library
from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler for the independent variables (features)
scaler_X = MinMaxScaler()
# Initialize a separate scaler for the dependent variable (target)
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df[features_cols])
y_scaled = scaler_y.fit_transform(df[[target_col]])

"""# **Data Transformation**

*   Sliding Window Sequences Creation
*   Time-Series Splitting


"""

def create_sequences(X, y, window, horizon):
    Xs, ys = [], []
    for i in range(window, len(X) - horizon):
        Xs.append(X[i-window:i])
        ys.append(y[i:i+horizon].mean())
    return np.array(Xs), np.array(ys)

WINDOW = 24   # ~6 months of weekly data
HORIZON=3 # 3 weeks for short term trend
#we can increase the horizon value to learn long term trends
X, y = create_sequences(X_scaled, y_scaled, WINDOW,HORIZON_SHORT)

# Train-test split
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Iterate through the splits to define training and testing indices
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

"""*   The Windowing: Deep learning models like LSTMs don't just look at one row; they look at a "cube" of data. The WINDOW tells the model how many weeks of history to consider for every single prediction.
*   The Horizon Mean: By taking the .mean() of the HORIZON, you are making the model predict a 3-week average trend rather than a single volatile day. This usually leads to a more stable and higher $R^2$ score.

# Model Architecture and Training

*   **Bidirectional LSTM**
"""

from tensorflow.keras.layers import Bidirectional


#model building
lstm_model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True),
                  input_shape=(WINDOW, X.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dense(1)
])

#model compilation
lstm_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse'
)

#Adding early stopping to halt the training when the validation loss stops improving
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

#model training
print("Re-training LSTM model with updated features...")
lstm_model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=32,
    shuffle=False,
    validation_split=0.15,
    callbacks=[early_stop],
    verbose=1
)
print("LSTM model re-training complete.")

"""**Bidirectional LSTM/GRU**: It allows the model to capture patterns from both past and future time steps within the input sequence, improving temporal feature learning.

*   **Bidirectional GRU**
"""

#model building
gru_model = Sequential([
    Bidirectional(GRU(128, return_sequences=True),
                  input_shape=(WINDOW, X.shape[2])),
    Dropout(0.3),
    GRU(64),
    Dense(1)
])

#model compilation
gru_model.compile(
    optimizer=Adam(learning_rate=0.0007),
    loss='mse'
)

#model training
print("Re-training GRU model with updated features...")
gru_model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=32,
    shuffle=False,
    validation_split=0.15,
    callbacks=[early_stop],
    verbose=1
)
print("GRU model re-training complete.")

"""
*   **Hybrid Transformer**



"""

from tensorflow.keras.layers import GlobalAveragePooling1D
# GlobalAveragePooling1D reduces temporal dimensions, captures overall trends and prevents overfitting retaining overall trend information

#model building
def build_transformer(input_shape):
    inputs = Input(shape=input_shape)

    x = MultiHeadAttention(num_heads=2, key_dim=16)(inputs, inputs)
    x = LayerNormalization()(x + inputs)

    x = LSTM(64)(x) # CRITICAL FIX
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)

    #model compilation
    model.compile(
        optimizer=Adam(0.0003),
        loss='mse'
    )
    return model

transformer_model = build_transformer((WINDOW, X.shape[2]))

#model training
print("Re-training Transformer model with updated features...")
transformer_model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=16,
    shuffle=False,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)
print("Transformer model re-training complete.")

"""**Hybrid Transformer**: It combines self-attention with feedforward layers to capture long-range temporal dependencies and complex relationships between features more effectively than standard RNNs.

# Model Evaluation
*   Inverse Transformation
*   Used Performance Metrics like MAE, RMSE, R2 Score
"""

def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)

    # Ensure 2D shape
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    # Inverse scaling
    preds_inv = scaler_y.inverse_transform(preds)
    y_test_inv = scaler_y.inverse_transform(y_test)

    # Reverse log transform
    preds_inv = np.expm1(preds_inv)
    y_test_inv = np.expm1(y_test_inv)

    rmse = np.sqrt(mean_squared_error(y_test_inv, preds_inv))
    mae = mean_absolute_error(y_test_inv, preds_inv)
    r2 = r2_score(y_test_inv, preds_inv)

    print(f"\n{name} Results")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE : {mae:.2f}")
    print(f"R²  : {r2:.2f}")

    return preds_inv.flatten(), y_test_inv.flatten()

"""# Model Inference
*   Generating Predictions
*   Collection of Results


"""

lstm_preds, actual = evaluate(lstm_model, X_test, y_test, "LSTM")
gru_preds, _ = evaluate(gru_model, X_test, y_test, "GRU")
trans_preds, _ = evaluate(transformer_model, X_test, y_test, "Transformer")

"""**LSTM**: Lowest RMSE and high R² (0.66) → best at capturing temporal patterns, but training can be slower.

**GRU**: Moderate performance (R² ~0.52) → simpler and faster than LSTM, but less accurate for long-term dependencies.

**Transformer**: Low RMSE and high R² (0.64) → captures long-range dependencies effectively, but may require more data and careful tuning.

# Results Summary

*   **Model Performance Comparison Table**
"""

results = pd.DataFrame({
    "Model": ["LSTM", "GRU", "Transformer"],
    "RMSE": [
        np.sqrt(mean_squared_error(actual, lstm_preds)),
        np.sqrt(mean_squared_error(actual, gru_preds)),
        np.sqrt(mean_squared_error(actual, trans_preds))
    ],
    "MAE": [
        mean_absolute_error(actual, lstm_preds),
        mean_absolute_error(actual, gru_preds),
        mean_absolute_error(actual, trans_preds)
    ],
    "R2 Score": [
        r2_score(actual, lstm_preds),
        r2_score(actual, gru_preds),
        r2_score(actual, trans_preds)
    ]
})

print(results)

"""# Visual Comparison

*   **Actual vs. Predicted Price Trends**
"""

plt.figure(figsize=(12,6))
plt.plot(actual, label='Actual Price', linewidth=2)
plt.plot(lstm_preds, label='LSTM')
plt.plot(gru_preds, label='GRU')
plt.plot(trans_preds, label='Transformer')
plt.title(f"{commodity} Price Prediction Comparison")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
