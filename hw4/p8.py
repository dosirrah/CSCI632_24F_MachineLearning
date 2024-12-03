# This code was taken from 
#     olemiss/CSCI632_23F_MachineLearning/project1/p2
#
# I am reusing the p2d2 dataset from project 1 problem 2 of the 2023 fall
# Machine Learning class.

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import pandas as pd


def generate_p2d2(n_samples=1000, noise=0.3):
    """
    Generates the p2d2 dataset and rotates it by 90 degrees.

    Parameters:
    n_samples (int): The total number of samples to generate.
    noise (float): The standard deviation of the Gaussian noise added to the data.

    Returns:
    tuple: Tuple containing the feature matrix and labels vector.
    """
    # Generate data with some noise
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=0)

    # Apply a 90-degree counter-clockwise rotation matrix
    rotation_matrix = np.array([[0, -1], [1, 0]])
    X_rotated = np.dot(X, rotation_matrix)

    # To make the dataset less separable, we will add additional points with random coordinates
    random_points = np.random.uniform(low=-1.5, high=2.0, size=(n_samples // 5, 2))
    random_labels = np.random.randint(0, 2, size=n_samples // 5)

    # Rotate the random points as well to maintain consistency
    random_points_rotated = np.dot(random_points, rotation_matrix)

    # Append these random points to the dataset
    X_final = np.vstack((X_rotated, random_points_rotated))
    y_final = np.hstack((y, random_labels))

    return X_final, y_final


def plot_p2d2(X, y):
    """
    Plots the p2d2 dataset.

    Parameters:
    X (np.ndarray): The feature matrix.
    y (np.ndarray): The labels vector.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', alpha=0.7)
    plt.title('p2d2 Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.show()

def save_p2d2(X, y, filepath):
    """
    Saves the p2d2 dataset to a CSV file.

    Parameters:
    X (np.ndarray): The feature matrix.
    y (np.ndarray): The labels vector.
    filepath (str): The path and name of the file where the data will be saved.
    """
    # Combine the features and the labels into one DataFrame
    df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
    df['Label'] = y

    # Save the DataFrame to a CSV file
    df.to_csv(filepath, index=False)

# Set random seed for reproducibility
np.random.seed(0)

n_samples = 1000  # you can specify the number of samples
noise = 0.3  # you can adjust the level of noise
p2d2_features, p2d2_labels = generate_p2d2(n_samples, noise)

# Save the dataset to a CSV file locally
save_p2d2(p2d2_features, p2d2_labels, "p8_data.csv")  # you can specify a custom filepath if needed

# Visualize the dataset
plot_p2d2(p2d2_features, p2d2_labels)

