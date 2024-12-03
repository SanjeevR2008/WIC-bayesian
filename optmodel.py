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
bonds_sym = ['B-T-4.750-15022041', 'B-T-6.250-15052030']
bonds = portfolio[portfolio['Symbol'].isin(bonds_sym)]
bond_allocation_percentage = bonds['MarketValue'].sum() / portfolio['MarketValue'].sum() 


price_data = yf.download(tickers, start="2015-12-02", end="2024-12-01")['Adj Close']


returns = price_data.pct_change().dropna()
annual_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

print("Annual Returns:\n", annual_returns)
print("Covariance Matrix:\n", cov_matrix)

mu = expected_returns.mean_historical_return(price_data)
print(mu)
S = risk_models.sample_cov(price_data)
ef = EfficientFrontier(mu, S)
ef.add_constraint(lambda w: w <= 0.15)
positive_weight_stocks = ['KMI', 'MAA', 'NU']  # Stocks that must have positive weights

# Add constraints for positive weights
for ticker in positive_weight_stocks:
    ef.add_constraint(lambda w, ticker=ticker: w[tickers.index(ticker)] >= 0.01)
ef.add_constraint(lambda w: w[tickers.index('KMI')] == 0.02)
# ef.add_constraint(lambda w: w[tickers.index('B-T-4.750-15022041')] == 0.02)
# ef.add_constraint(lambda w: w[tickers.index('B-T-6.250-15052030')] == 0.02)
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
plt.ylim(0, 0.2)
plt.xticks(x, labels, rotation=90)
plt.legend()
plt.title("Portfolio Allocation Comparison")
plt.show()