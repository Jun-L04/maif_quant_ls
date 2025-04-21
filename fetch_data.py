# get the excel data needed to set up the ARCH model
import yfinance as yf
import pandas as pd
import time

#tickers = ['APO', 'HESAY', 'RHI', 'BBY', 'PM', 'HIMS', 'WM', 'LYV', 'UONE']
tickers = ['APO', 'RHI', 'BBY', 'PM', 'HIMS', 'WM', 'LYV', 'UONE']
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

    df.to_excel(f'{ticker}.xlsx', index=False)
    time.sleep(2)




