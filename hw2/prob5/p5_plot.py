import pandas as pd
import matplotlib.pyplot as plt

# To correct the filename, let's reload the original file path where data was stored.
data = pd.read_csv('p5_training_data.csv')

# Plot the data without revealing the underlying function
plt.figure(figsize=(8, 6))
plt.scatter(data['X'], data['y'], color='blue', label='Training Data', alpha=0.7)
plt.title('Plot of Training Data', fontsize=16)
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.grid(True)
plt.legend()
plt.savefig('p5_plot.png')  # Save as PNG
