import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def extract_stats(df):
    """
    Statistics like min, max, mean, median, std and variance
    from a dataframe df and store them to an excel file
    :param df: Dataset to analyze in the form of Pandas Dataframe, dimensions (n x m)
    :return : Pandas Dataframe, dimensions (m x 6)
    """
    stats = df.describe()\
              .T\
              .drop(['count', '25%', '75%'], axis=1)\
              .rename(columns= {'50%' : 'median'})
    stats = stats[["min", "max", "mean", "median", "std"]]
    stats['Variance'] = stats['std'] ** 2
    stats = stats.rename(columns={"Variance" : "variance"})
    return stats

def corr_table(df):
    """
    Makes a correlation table of the features of the given dataset
    :param df: Dataset given, dimensions (n x m)
    :return: Pandas Dataframe, dimensions (m x m)
    """
    corr_table = df.corr(method = 'pearson')
    return corr_table


def make_lags(vector, lags, name='Close'):

    return pd.concat(
      {
         f'{name}_lag_{i}': vector.shift(i)
         for i in range(1, lags + 1)
      },
      axis=1)

def spot_outliers(vector):

    # IQR
    Q1 = np.nanpercentile(vector, 25,
                          interpolation='midpoint')

    Q3 = np.nanpercentile(vector, 75,
                          interpolation='midpoint')
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR
    lower = Q1 - 1.5 * IQR

    outlier_indices = btc_prices[
        (btc_prices['Close Difference'] < lower) | (btc_prices['Close Difference'] > upper)].index.tolist()

    return outlier_indices

#def compare_prices()

pd.set_option('display.max_columns', 20)

# Store Data
btc_prices = pd.read_csv("resources/BTC-USD.csv")
# btc_prices = yf.download('BTC-USD', start='2010-12-25', end='2017-09-11')

#Check for NaN values
nan_values_sum = btc_prices.isnull().sum() # There are no NaN values

# Set Date as an Index
btc_prices['Date'] = pd.to_datetime(btc_prices['Date'])
btc_prices.set_index(['Date'], inplace=True)

# Descriptive Analysis (to excel)
btc_stats = extract_stats(btc_prices)
btc_stats.to_excel("resources/btc_stats.xlsx", engine="openpyxl")

# Correlation Table
initial_btc_corr_table = corr_table(btc_prices)

# We notice that close and adjusted close prices have a Pearson correlation of 1, meaning that
# in the present dataset where Close is supposed to be our response variable we do not need adjusted close
btc_prices = btc_prices.drop(columns="Adj Close")

# Store indices
indices = btc_prices.index

# We run again our correlation table without using adjusted close and store it to an excel file
btc_corr_table = corr_table(btc_prices)
btc_corr_table.to_excel("resources/btc_corr_table.xlsx", engine="openpyxl")

# Plot histograms of features to check if they are normally distributed

"""btc_prices.hist(bins=30, figsize=(15, 10))
plt.savefig("resources/btc_features_distributions.png", dpi = 100)

# Plot daily close price
btc_prices.plot(y = 'Close', use_index = True)
plt.savefig("resources/btc_daily_close_price.png", dpi = 100)
"""

# We create two extra features in our dataset
# Close Lag shows the previous value of close
# Increased has True value if price has rised from yesterday (i.e. if Close Lag < Lag), else False
# Relative Close shows the relative difference between High and Close price from the previous day
btc_prices['Close Lag'] = btc_prices['Close'].shift(1)
btc_prices['Increased'] = np.where(btc_prices['Close'] > btc_prices['Close Lag'], True, False)
btc_prices['Relative Close'] = ((btc_prices['Close'] - btc_prices['Low']) / (btc_prices['High'] - btc_prices['Low'])).shift(1)
btc_prices['Today Potential'] = btc_prices['Open'] > btc_prices['Open'].shift(1)
btc_prices['Close Difference'] = btc_prices['Close'] - btc_prices['Close Lag']

# Add column that has True value if price has rised from yesterday, else False
# btc_prices['Increased'] = np.where(btc_prices['Close'] > )

# X = btc_prices[['Open', 'High', 'Low', 'Volume']]    # Features
# y = btc_prices['Close'] # Response Variable



"""
Date = pd.Series(btc_prices.index.values)
Adj_Close = btc_prices['Adj Close']

# Plot needs beautification
plt.plot(Date, Adj_Close)
plt.savefig('resources/daily_adj_close.png')
"""

