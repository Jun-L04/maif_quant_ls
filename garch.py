import pandas as pd
from arch import arch_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

tickers = ['APO', 'HESAY', 'PM', 'LYV', 'WM', 'RHI', 'BBY']
results = {}
for ticker in tickers:
    path = ticker + '.xlsx'
    results[ticker] = predict_garch_volatility(ticker, path)

#print(results)

plt.bar(results.keys(), results.values(), color="lightblue", edgecolor ='black', width=0.5)
plt.xlabel("Tickers")
plt.ylabel("Volatility")
plt.title("Forecasted Future Week Volatility")
plt.show()
#plt.savefig("forecast.png")



plt.figure(figsize=(10, 6))
for ticker in tickers:
    path = ticker + '.xlsx'
    df = pd.read_excel(path)
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    
    # Plot the KDE (Kernel Density Estimate) for the log returns
    sns.kdeplot(df['LogReturn'], label=ticker, linewidth=2, bw_adjust=200)

plt.title("Log Return Distribution Across Tickers")
plt.xlabel("Log Return")
plt.ylabel("Density")
plt.legend(title="Tickers")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
#plt.savefig("log_return_dist.jpg")