import yfinance as yf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
from arch import arch_model
from numpy.linalg import inv
from scipy import stats
from tqdm import tqdm

start = "2022-01-01"
end = "2024-01-01"

spy_data = yf.download('SPY', start=start, end=end, group_by='ticker', auto_adjust=True)
spy_close = spy_data['SPY']['Close'].dropna()
spy_returns = spy_close.pct_change().dropna()

tickers = ['APO', 'HESAY', 'PM', 'LYV', 'WM', 'RHI', 'BBY']
data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True)
adj_close = pd.DataFrame()
for ticker in tickers:
    try:
        adj_close[ticker] = data[ticker]['Close']
    except KeyError:
        print(f"Missing data for {ticker}, skipping...")
adj_close.dropna(inplace=True)
returns = adj_close.pct_change().dropna()


# --------

common_index = returns.index.intersection(spy_returns.index)
spy_returns = spy_returns.loc[common_index]
aligned_returns = returns.loc[common_index]

# Calculate rolling beta for each ticker (or static)
beta = pd.Series(index=aligned_returns.columns)

rolling_window = 60

# stores rolling beta
rolling_beta = pd.DataFrame(index=aligned_returns.index[rolling_window - 1:], columns=aligned_returns.columns)

for ticker in aligned_returns.columns:
    stock_returns = aligned_returns[ticker]

    for i in range(len(stock_returns) - rolling_window + 1):
        spy_window = spy_returns.iloc[i:i + rolling_window]
        stock_window = stock_returns.iloc[i:i + rolling_window]
        
        # linear reg
        X = spy_window.values.reshape(-1,1)
        y = stock_window.values
        model = LinearRegression().fit(X, y)
        
        # storing
        rolling_beta.iloc[i, rolling_beta.columns.get_loc(ticker)] = model.coef_[0]

rolling_beta.dropna(inplace=True)

print("\nRolling BETA:\n", rolling_beta)