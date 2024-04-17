import pandas as pd

import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv('./data-visualization/PAC2/EX1/data/cancer.csv')

# Extract the column you want to plot
column_to_plot = data['Radius (mean)']

# Create the histogram
plt.hist(column_to_plot, bins=10, edgecolor='black')

# Customize the plot
plt.title('Histogram of Column')
plt.xlabel('Values')
plt.ylabel('Frequency')

# Show the plot
plt.show()