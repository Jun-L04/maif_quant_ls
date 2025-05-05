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

tickers = ['APO', 'HESAY', 'PM', 'LYV', 'WM', 'RHI', 'BBY']
start = "2005-01-01"
end = "2025-05-01"

idio_vol = pd.DataFrame()
idio_dict = {}
for ticker in tqdm(tickers):
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval='1wk',
        auto_adjust=True,
        progress=False,
        threads=False
    )
    df = df.dropna()
    df = df[['Close']]            # keep just Close
    df.columns = ['Close']
    df.reset_index(inplace=True)  # Date becomes a column

    # Market benchmark, SPY
    spy = yf.download(
        tickers="SPY",
        start=start,
        end=end,
        interval='1wk',
        auto_adjust=True,
        progress=False,
        threads=False
    )
    spy = spy.dropna()
    spy = spy[['Close']]            # keep just Close
    spy.columns = ['Close']
    spy.reset_index(inplace=True)  # Date becomes a column

    # avoid rate limit
    time.sleep(3)

    # risk free return
    risk_free_return = 0.02 / 365

    # log returns and excess returns on market
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    df['ExcessReturn'] = spy['Close'] / spy['Close'].shift(1) - risk_free_return

    # rolling window size
    window = 30

    idiosyncratic_volatility = []

    # go through each window
    for i in range(len(df) - window + 1):
        # copy over the content for current window
        rolling_window = df.iloc[i:i + window].copy()

        # CAPM linear regression on the rolling window
        slope, intercept, r, p, std_err = stats.linregress(
            rolling_window['LogReturn'], rolling_window['ExcessReturn']
        )

        # get residuals for the rolling window
        rolling_window['Residual'] = rolling_window['ExcessReturn'] - (
            slope * rolling_window['LogReturn'] + intercept
        )

        # the standard deviation of the residuals
        # or volatility
        residual_std = rolling_window['Residual'].std()
        idiosyncratic_volatility.append(residual_std)

    # append back to the dataframe
    df['IdiosyncraticVolatility'] = [np.nan] * (window - 1) + idiosyncratic_volatility
    df = df.dropna()

    idio_vol[ticker] = df['IdiosyncraticVolatility'].reset_index(drop=True)

    garch_model = arch_model(idio_vol[ticker], vol='Garch', p=1, q=1)
    fitted_model = garch_model.fit(disp="off")
    forecast = fitted_model.forecast(horizon=1)
    next_week_volatility = forecast.variance.values[-1, 0] ** 0.5
    idio_dict[ticker] = next_week_volatility


#print('\n', idio_vol)
print(idio_dict)