import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Define the number of data points
n_points = 100

# Generate random input data (X)
X = np.linspace(0, 10, n_points)

# Define a nonlinear, increasing function (e.g., quadratic)
true_val = 2 * np.sin(X) + 0.5 * X

# Add Gaussian noise
noise = np.random.normal(0, 1, n_points)
y = true_val + noise

# Combine the data into a DataFrame
data = pd.DataFrame({'X': X, 'y': y})

# Output the data to a CSV file
output_path = 'p5_training_data.csv'
data.to_csv(output_path, index=False)

# Display path to user
output_path
