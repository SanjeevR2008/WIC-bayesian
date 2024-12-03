import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.optimize import minimize

# Step 1: Load and preprocess data
# Simulate factor data (e.g., value, momentum, quality)
np.random.seed(42)
num_assets = 50
num_factors = 5
num_obs = 252

factor_exposures = np.random.randn(num_obs, num_factors)  # Simulated factors
returns = np.random.randn(num_obs, num_assets) * 0.01    # Simulated returns

# Step 2: Factor discovery using PCA
pca = PCA(n_components=3)
latent_factors = pca.fit_transform(factor_exposures)
print("Explained Variance Ratios:", pca.explained_variance_ratio_)

# Step 3: Factor modeling
# Train a Ridge regression to predict returns
ridge = Ridge(alpha=1.0)
ridge.fit(latent_factors, returns.mean(axis=1))
predicted_returns = ridge.predict(latent_factors)

# Step 4: Portfolio optimization
def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def optimize_portfolio(expected_returns, cov_matrix):
    num_assets = len(expected_returns)
    init_guess = np.ones(num_assets) / num_assets
    bounds = [(0.0, 1.0) for _ in range(num_assets)]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    def objective(weights):
        return -np.dot(weights, expected_returns) / portfolio_volatility(weights, cov_matrix)
    
    result = minimize(objective, init_guess, bounds=bounds, constraints=constraints)
    return result.x

cov_matrix = np.cov(returns.T)
optimal_weights = optimize_portfolio(predicted_returns, cov_matrix)

# Step 5: Backtest
portfolio_returns = np.dot(returns, optimal_weights)
sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std()
print("Sharpe Ratio:", sharpe_ratio)

