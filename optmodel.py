import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting
import matplotlib.pyplot as plt

portfolio = pd.read_csv('portfolio.csv')
print(portfolio.head())

tickers = portfolio['Symbol'].dropna().unique().tolist()
tickers.pop()
tickers.pop()
# print(tickers)

price_data = yf.download(tickers, start="2015-12-02", end="2024-12-01")['Adj Close']
# print(price_data.head())

returns = price_data.pct_change().dropna()
annual_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

print("Annual Returns:\n", annual_returns)
print("Covariance Matrix:\n", cov_matrix)

mu = expected_returns.capm_return(price_data)
S = risk_models.sample_cov(price_data)
ef = EfficientFrontier(mu, S)
ef.add_constraint(lambda w: w <= 0.15)

weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print("Optimized weights:\n", cleaned_weights)
ef.portfolio_performance(verbose=True)

current_weights = portfolio['MarketValue'] / portfolio['MarketValue'].sum()

# Plot current vs optimized weights and stuff
labels = portfolio['Symbol']
optimized_weights = [cleaned_weights.get(ticker, 0) for ticker in labels]

# graph stuff
x = range(len(labels))
plt.bar(x, current_weights, width=0.4, label="Current", align='center')
plt.bar(x, optimized_weights, width=0.4, label="Optimized", align='edge')
plt.xticks(x, labels, rotation=90)
plt.legend()
plt.title("Portfolio Allocation Comparison")
plt.show()