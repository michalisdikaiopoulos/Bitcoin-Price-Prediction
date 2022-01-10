# This can work with any feature set

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.callbacks import EarlyStopping


from matplotlib import pyplot as plt

from preprocessing import btc_prices, make_lags, indices

def sort_key(val):
    """
    Search in string all list of integers and return the first of them
    :param val: Given string
    :return: First integer in list
    """
    return [int(s) for s in val.split('_') if s.isdigit()][0]

# ARGS
LAGS = 50
STEPS = 1 # Number of predicted daily close values
N_NODES = 64
EPOCHS = 35 # Optimized
DROPOUT_RATE = 0.2

# Names of Variables used
# variables = ['Open', 'High', 'Low', 'Close', 'Volume']
variables = ['Close']

# Columns of btc_prices used
variables_f = btc_prices[variables].astype(float)

# Normalization
scaler = MinMaxScaler()
scaler = scaler.fit(variables_f)
variables_scaled = pd.DataFrame(scaler.transform(variables_f)).set_index(indices)
variables_scaled.columns = variables
print(variables_scaled)

# Feature Set
X = pd.concat([make_lags(variables_scaled[col], LAGS, col)for col in variables], axis=1)

# Sort feature set, in order to be reshaped properly afterwards
cols = list(X.columns.values)
cols.sort(key = sort_key)
X = X[cols]

# Target Set
y = variables_scaled['Close']

# DROPPING FIRST ROWS OUT (First rows have NaN in all shifted columns)
X.drop(index=indices[0:LAGS], axis=0, inplace=True)
y.drop(index=indices[0:LAGS], axis=0, inplace=True)

# Reshape X and y in appropriate form for LSTM layers
X = X.to_numpy().reshape((X.shape[0], LAGS, len(variables)))[:, ::-1, :]
y = y.to_numpy().reshape((y.shape[0], 1))

# Split Dataset to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05,random_state=None, shuffle=False, stratify=None)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05,random_state=None, shuffle=False, stratify=None)

print(X_train.shape)

# Fix RNN Model
model = Sequential()
model.add(LSTM(N_NODES, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(N_NODES, activation='relu', return_sequences=False))
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(y_train.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()

# Find optimal epochs
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience=20, verbose = 2)

# Train Data
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, validation_data=(X_test, y_test), verbose=2, callbacks=[es])


# ------------ PREDICTION IN TEST SET ------------

# Predict
y_pred = model.predict(X_test)

# Calculate RMSE
test_score = mean_squared_error(y_test, y_pred)
print(f'Test Score  : {test_score}')
