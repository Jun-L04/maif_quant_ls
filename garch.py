import pandas as pd
from arch import arch_model
import numpy as np
import yfinance as yf
import pandas as pd
import time

#tickers = ['APO',  'RHI', 'BBY', 'PM', 'HIMS', 'WM', 'LYV', 'UONE']
tickers = ['APO', 'HESAY', 'RHI', 'BBY', 'PM', 'HIMS', 'WM', 'LYV', 'UONE']
results = {}
for ticker in tickers:
    df = yf.download(
        tickers=ticker,
        start='2005-01-01',
        end='2025-04-18',
        interval='1mo',
        auto_adjust=True,
        progress=False, 
        threads=False 
    )
    df = df.dropna()
    df = df[['Close']]            # keep just Close if you like
    df.columns = ['Close']
    df.reset_index(inplace=True)  # Date becomes a column

    #df.to_excel(f'{ticker}.xlsx', index=False)
    time.sleep(1)
    
    # get the log returns
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    # rescaling for garch
    df['LogReturn'] = df['LogReturn'] * 10
    
    # fitting GARCH(1,1) model
    model = arch_model(df['LogReturn'].dropna(), vol='Garch', p=1, q=1)
    fitted_model = model.fit(disp="off")
    
    # forecast one month into future
    forecast = fitted_model.forecast(horizon=1)
    next_month_volatility = forecast.variance.values[-1, 0] ** 0.5  # variance to std (volatility)
    next_month_volatility /= 10
    print(f"{ticker}: {next_month_volatility}")
    results[ticker] = next_month_volatility
    
print(results)

def predict_garch_volatility(company_name: str, file_path: str):
    # predicts next month's volatility based on data given the data and ticker
    
    # get df from excel 
    df = pd.read_excel(file_path)
    
    # get the log returns
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    # rescaling for garch
    df['LogReturn'] = df['LogReturn'] * 10
    
    # fitting GARCH(1,1) model
    model = arch_model(df['LogReturn'].dropna(), vol='Garch', p=1, q=1)
    fitted_model = model.fit(disp="off")
    
    # forecast one month into future
    forecast = fitted_model.forecast(horizon=1)
    next_month_volatility = forecast.variance.values[-1, 0] ** 0.5  # variance to std (volatility)
    next_month_volatility /= 10
    print(f"{company_name}: {next_month_volatility}")
    return next_month_volatility


# tickers = ['APO', 'HESAY', 'RHI', 'BBY', 'PM', 'HIMS', 'WM', 'LYV', 'UONE']
# results = {}
# for ticker in tickers:
#     path = ticker + '.xlsx'
#     results[ticker] = predict_garch_volatility(ticker, path)

# print(results)