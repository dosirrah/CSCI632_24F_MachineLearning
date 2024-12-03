# This code was taken from 
#     olemiss/CSCI632_23F_MachineLearning/project1/p2

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs


def generate_p2d1(n_samples=1000):
    """
    Generates the p2d1 dataset.

    Parameters:
    n_samples (int): The total number of samples to generate for each class.

    Returns:
    tuple: Tuple containing the feature matrix and labels vector.
    """
    # Set random seed for reproducibility
    np.random.seed(0)

    # Generate first part of the dataset
    X_part1, y_part1 = make_blobs(n_samples=n_samples, centers=[(2, 2)], cluster_std=1.5, random_state=0)

    # Generate second part of the dataset with some noise for complexity
    X_part2 = np.random.uniform(low=-10, high=10, size=(n_samples, 2))
    noise = np.random.normal(scale=1.0, size=n_samples)
    y_part2 = np.array([1 if (x[1] - 0.2 * x[0] - 2 + n > 0) else 0 for x, n in zip(X_part2, noise)])

    # Combine the datasets
    X = np.vstack((X_part1, X_part2))
    y = np.hstack((y_part1, y_part2))

    return X, y


def plot_p2d1(X, y):
    """
    Plots the p2d1 dataset.

    Parameters:
    X (np.ndarray): The feature matrix.
    y (np.ndarray): The labels vector.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', alpha=0.7)
    plt.title('p2d1 Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.show()

def save_p2d1(X, y, filepath):
    """
    Saves the p2d1 dataset to a CSV file.

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


def main():
    # Usage
    n_samples = 1000  # you can specify the number of samples
    p2d1_features, p2d1_labels = generate_p2d1(n_samples)

    # Save the file locally (you can specify a custom filepath if needed)
    #save_p2d1(p2d1_features, p2d1_labels, "p2d1_data.csv")
    save_p2d1(p2d1_features, p2d1_labels, "p8_data.csv")

    plot_p2d1(p2d1_features, p2d1_labels)


if __name__ == "__main__":
    main()


