import pandas as pd
from preprocessing import btc_prices, indices, make_lags
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ARGS
LAGS = 1
N_SPLITS = 10

# -------------------------------------- STAGE 1 --------------------------------------

# Store target variable
y = btc_prices['Increased']

# Store features
X = pd.concat([make_lags(y, lags=LAGS, name='Increased')], axis=1)
# X = btc_prices[['Today Potential']]

# Normalization
min_max_scaler = MinMaxScaler()
x_scaled =  min_max_scaler.fit_transform(X.values)
X = pd.DataFrame(x_scaled).set_index(indices)

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
logistic_model = LogisticRegression(class_weight='balanced')

# DROPPING FIRST ROWS OUT
X.drop(index=indices[0:LAGS], axis=0, inplace=True)
y.drop(index=indices[0:LAGS], axis=0, inplace=True)

# -------------------------------------- STAGE 2 --------------------------------------
# Using time series cross validation to find the best model with which we later
# train our data

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

   # Train
   logistic_model.fit(X_train, y_train)

   # Predict
   y_pred = logistic_model.predict(X_test)

   """# Calculate scores and errors
   average_accuracy += logistic_model.score(X_test, y_test)
   print("Score : " + str(logistic_model.score(X_test, y_test)))
   print(confusion_matrix(y_test, y_pred, labels=[True, False]))
   print(classification_report(y_test, y_pred, labels=[True, False], zero_division=0))

print(f"Average Accuracy :  {average_accuracy / N_SPLITS}")
"""

# -------------------------------------- STAGE 3 --------------------------------------
# Using the optimal model we found in Cross-Validation, we train our model and calculate its accuracy

# Split data for training
X_train, X_test, y_train, y_test = \
   train_test_split(X, y, test_size=0.2, train_size=0.8, shuffle=False)

logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

# Calculate accuracy, Precision and Recall and showing the confusion matrix
accuracy = logistic_model.score(X_test, y_test)
print(f"Logistic Model Accuracy : {accuracy}")
print(f"\nConfusion Matrix : {confusion_matrix(y_test, y_pred, labels=[True, False])}")
print(f'\nClassification Report : {classification_report(y_test, y_pred, labels=[True, False], zero_division = 0)}')
