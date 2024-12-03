import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scipy.optimize import minimize

# Step 1: Generate Simulated Factor Data
np.random.seed(42)
num_assets = 10      # Number of assets
num_factors = 5      # Number of factors
num_obs = 252        # Number of observations (e.g., trading days)

# Simulate factor exposures and asset returns
factor_data = np.random.randn(num_obs, num_factors)  # Simulated factor data
true_weights = np.random.uniform(0.1, 0.5, (num_factors, num_assets))  # True mapping
returns = factor_data @ true_weights + np.random.randn(num_obs, num_assets) * 0.01

# Step 2: Preprocessing
scaler = StandardScaler()
factor_data_scaled = scaler.fit_transform(factor_data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(factor_data_scaled, returns, test_size=0.2, random_state=42)

# Step 3: Define and Train the Neural Network
model = Sequential([
    Dense(64, activation='relu', input_dim=num_factors),
    Dense(32, activation='relu'),
    Dense(num_assets)  # Output layer for asset returns
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# Predict asset returns
predicted_returns_train = model.predict(X_train)
predicted_returns_test = model.predict(X_test)

# Step 4: Portfolio Optimization
def portfolio_volatility(weights, cov_matrix):
    """Calculate portfolio volatility."""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def optimize_portfolio(expected_returns, cov_matrix):
    """Optimize portfolio weights."""
    num_assets = len(expected_returns)
    init_guess = np.ones(num_assets) / num_assets  # Start with equal weights
    bounds = [(0.0, 1.0) for _ in range(num_assets)]  # No short-selling
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # Weights sum to 1

    def objective(weights):
        return -np.dot(weights, expected_returns) / portfolio_volatility(weights, cov_matrix)

    result = minimize(objective, init_guess, bounds=bounds, constraints=constraints)
    return result.x

# Calculate covariance matrix
cov_matrix = np.cov(returns.T)

# Use the test set predictions for portfolio optimization
expected_returns = predicted_returns_test.mean(axis=0)
optimal_weights = optimize_portfolio(expected_returns, cov_matrix)

# Step 5: Evaluate Portfolio Performance
portfolio_returns = np.dot(returns, optimal_weights)
sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std()

print("Optimal Weights:", optimal_weights)
print("Portfolio Sharpe Ratio:", sharpe_ratio)
