# TODO : Find optimal number of splits
# TODO : In tscv split we drop first row out of train datasets due to close lag NaN value. Check in the end if still
#        in need.

import pandas as pd
from preprocessing import btc_prices, indices, make_lags
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ARGS
LAGS = 1
N_SPLITS = 5

# -------------------------------------- STAGE 1 --------------------------------------

# Store target variable
y = btc_prices['Close']

# Store Features
X = make_lags(y, lags=LAGS) # This is the best X when not using same day data
# X = btc_prices[['Open', 'High', 'Low']] # This is the best X when using same day data

# Normalization
min_max_scaler = MinMaxScaler()
x_scaled =  min_max_scaler.fit_transform(X.values)
X = pd.DataFrame(x_scaled).set_index(indices)

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
linear_model = LinearRegression()

# DROPPING FIRST ROWS OUT (First rows have NaN in all shifted columns)
X.drop(index=indices[0:LAGS], axis=0, inplace=True)
y.drop(index=indices[0:LAGS], axis=0, inplace=True)

r2_scores = []
rms_errors = []

# -------------------------------------- STAGE 2 --------------------------------------
# In this stage we used Time Series cross validation to find the best model with which
# we later train our data

# Cross Validation
for train, test in tscv.split(X, y):

   X_train, X_validation = X.loc[indices[train[LAGS:]]], X.loc[indices[test]]
   y_train, y_validation = y.loc[indices[train[LAGS:]]], y.loc[indices[test]]

   # Train
   linear_model.fit(X_train, y_train)

   # Predict
   y_pred = linear_model.predict(X_validation)

   # Calculate scores and errors
   score = linear_model.score(X_validation, y_validation)
   r2_scores.append(score)

   rmse = mean_squared_error(y_validation, y_pred, squared=False)
   normalized_rmse = rmse / (y.max() - y.min())
   rms_errors.append(normalized_rmse)
   
print("R^2 scores : " + str(r2_scores))
print("Root Mean Squared Errors : " + str(rms_errors))
print("Cross-Validation Average R^2 : " + str(sum(r2_scores) / len(r2_scores)))
print("Cross-Validation Average Normalised RMSE : " + str(sum(rms_errors) / len(rms_errors)))

# -------------------------------------- STAGE 3 --------------------------------------
# We train our data with our chosen features from cross validation stage

# Split data for training
X_train, X_test, y_train, y_test = \
   train_test_split(X, y, test_size=0.2, train_size=0.8, shuffle=False)

# Train with selected model

linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)

# Calculate r2 score and rmse
score = linear_model.score(X_test, y_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
normalized_rmse = rmse / (y.max() - y.min())
rms_errors.append(normalized_rmse)

print(f'\nTest Data R^2 Score : {score}')
print(f'Test Data Normalised RMSE : {normalized_rmse}')

# Plot last iteration results
plt.plot(range(366), y_test, label = "Actual Prices")
plt.plot(range(366), y_pred, label = "Predicted Prices")
plt.legend()
plt.show()







