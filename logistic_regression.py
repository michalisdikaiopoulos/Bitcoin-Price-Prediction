# TODO : Change the following parameters in logistic model : penalty,

import pandas as pd
from preprocessing import btc_prices, indices, make_lags
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ARGS
LAGS = 1
N_SPLITS = 5

# Features : Increased Lags, Close Lag, Relative Close, Today Potential, Close Difference, Open, High, Low, Volume

# Store separately features and target variable
y = btc_prices['Increased']
# X = pd.concat([make_lags(y, lags=LAGS, name='Increased')], axis=1)
X = btc_prices[['Today Potential']]

# Normalization
min_max_scaler = MinMaxScaler()
x_scaled =  min_max_scaler.fit_transform(X.values)
X = pd.DataFrame(x_scaled).set_index(indices)

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
logistic_model = LogisticRegression(class_weight='balanced')

# DROPPING FIRST ROWS OUT
X.drop(index=indices[0:LAGS], axis=0, inplace=True)
y.drop(index=indices[0:LAGS], axis=0, inplace=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)

average_accuracy = 0

for train, test in tscv.split(X, y):

   X_train, X_test = X.loc[indices[train[LAGS:]]], X.loc[indices[test]]
   y_train, y_test = y.loc[indices[train[LAGS:]]], y.loc[indices[test]]


   """# PLOT TRAIN AND TEST SETS
   plt.title('Bitcoin Prices Train and Test Sets', size=20)
   plt.plot(btc_prices['Close'].loc[indices[train[LAGS:]]], label='Training set')
   plt.plot(btc_prices['Close'].loc[indices[test]], label='Test set', color='orange')
   plt.show()
"""
   #print(btc_prices['Close'].loc[indices[train[LAGS:]]])

   # Train
   logistic_model.fit(X_train, y_train)

   # Predict
   y_probs = logistic_model.predict_proba(X_test)[:, 1]
   y_pred = y_probs > 0.5

   # Calculate scores and errors
   average_accuracy += logistic_model.score(X_test, y_test)
   print("Score : " + str(logistic_model.score(X_test, y_test)))
   print(confusion_matrix(y_test, y_pred, labels=[True, False]))
   print(classification_report(y_test, y_pred, labels=[True, False], zero_division=0))

print(f"Average Accuracy :  {average_accuracy / N_SPLITS}")

