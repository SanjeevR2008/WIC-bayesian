import numpy as np
import pandas as pd
import pymc as pm
from scipy.optimize import minimize

# Step 1: Load and preprocess data
# Simulate some stock return data for 3 assets
np.random.seed(42)
num_assets = 3
num_obs = 100
returns = np.random.randn(num_obs, num_assets) * 0.02  # Simulated daily returns
print(returns)
# Step 2: Define Bayesian model
def bayesian_inference(returns):
    num_assets = returns.shape[1]
    with pm.Model() as model:
        # Priors for expected returns
        mu = pm.Normal("mu", mu=0, sigma=1, shape=num_assets)
        # Priors for covariance matrix using LKJ prior
        cov = pm.LKJCholeskyCov("cov", n=num_assets, eta=2, sd_dist=pm.HalfCauchy.dist(2))
        chol = pm.expand_packed_triangular(num_assets, cov)
        sigma = pm.Deterministic("sigma", chol.dot(chol.T))  # Covariance matrix
        
        # Likelihood for observed returns
        returns_obs = pm.MvNormal("returns_obs", mu=mu, chol=chol, observed=returns)
        
        # Sample from the posterior
        trace = pm.sample(1000, return_inferencedata=False, progressbar=True)
        
    return trace

# Run Bayesian inference
trace = bayesian_inference(returns)

# Extract posterior samples for expected returns and covariance matrix
posterior_mu = trace["mu"].mean(axis=0)
posterior_cov = np.mean(trace["sigma"], axis=0)

# Step 3: Portfolio optimization
def portfolio_sharpe_ratio(weights, mu, cov, risk_free_rate=0.0):
    """
    Objective function: Negative Sharpe ratio (we minimize this)
    """
    portfolio_return = np.dot(weights, mu)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio

def optimize_portfolio(mu, cov, risk_free_rate=0.0):
    num_assets = len(mu)
    init_guess = np.ones(num_assets) / num_assets  # Equal weight initialization
    bounds = [(0.0, 1.0) for _ in range(num_assets)]  # No short-selling
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  # Weights sum to 1
    
    result = minimize(portfolio_sharpe_ratio, init_guess, args=(mu, cov, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x  # Optimized weights

# Optimize portfolio weights
optimized_weights = optimize_portfolio(posterior_mu, posterior_cov)

# Step 4: Display results
print("Posterior Expected Returns:", posterior_mu)
print("Posterior Covariance Matrix:\n", posterior_cov)
print("Optimized Portfolio Weights:", optimized_weights)

# Step 5: Backtesting (Optional)
# Simulate portfolio returns using out-of-sample data
portfolio_returns = np.dot(returns, optimized_weights)
print("Simulated Portfolio Returns:", portfolio_returns.mean())
