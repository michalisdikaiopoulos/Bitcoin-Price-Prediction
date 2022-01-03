# TODO : Find optimal number of splits
# TODO : In tscv split we drop first row out of train datasets due to close lag NaN value. Check in the end if still
#        in need.

import pandas as pd
from preprocessing import btc_prices, indices, make_lags
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ARGS
LAGS = 1
N_SPLITS = 5

# Store separately features and target variable
y = btc_prices['Close']
# X = make_lags(y, lags=LAGS) # This is the best X when not using same day data
X = btc_prices[['Open', 'High', 'Low']] # This is the best X when using same day data

# Normalization
min_max_scaler = MinMaxScaler()
x_scaled =  min_max_scaler.fit_transform(X.values)
X = pd.DataFrame(x_scaled).set_index(indices)

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
linear_model = LinearRegression()

# DROPPING FIRST ROWS OUT
X.drop(index=indices[0:LAGS], axis=0, inplace=True)
y.drop(index=indices[0:LAGS], axis=0, inplace=True)

# scores = cross_val_score(linear_model, X, y, scoring='neg_root_mean_squared_error', cv=tscv, n_jobs= -1)

r2_scores = []
rms_errors = []
for train, test in tscv.split(X, y):

   X_train, X_test = X.loc[indices[train[LAGS:]]], X.loc[indices[test]]
   y_train, y_test = y.loc[indices[train[LAGS:]]], y.loc[indices[test]]

   # Train
   linear_model.fit(X_train, y_train)

   # Predict
   y_pred = linear_model.predict(X_test)

   # Calculate scores and errors
   score = linear_model.score(X_test, y_test)
   r2_scores.append(score)

   rmse = mean_squared_error(y_test, y_pred, squared=False)
   normalized_rmse = rmse / (y.max() - y.min())
   rms_errors.append(normalized_rmse)

# Plot last iteration results
plt.plot(range(304), y_test, label = "Actual Prices")
plt.plot(range(304), y_pred, label = "Predicted Prices")
plt.legend()
plt.show()



print("R^2 scores : " + str(r2_scores))
print("Root Mean Squared Errors : " + str(rms_errors))
print("Average RMSE : " + str(sum(rms_errors) / len(rms_errors)))
print("Average R^2 : " + str(sum(r2_scores) / len(r2_scores)))





