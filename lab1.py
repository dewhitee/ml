

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the dataset into the numpy array
dataset = np.genfromtext('mydataset.csv', delimiter=',', skip_header=True)

# Get observations and variables count
observationsCount, variablesCount = dataset.shape
print(f'Count of observations: {observationsCount}, Count of variables {variablesCount}')

# Delete rows with missing data
#dataset = np.delete()

# Categorical variables to binary
for v in dataset:
    if type(v) is not int:
        exit()

# Print first 5
print(dataset[:5])

# Split into training and test datasets
splitter = round(observationsCount * 0.75)
print(splitter)

train_set = dataset[:splitter]
test_set = dataset[splitter:]
#y_train = train_set[:, -1]
#y_test = test_set[:, -1]
#x_train = train_set[:, :variablesCount - 1]
#x_test = test_set[:, :variablesCount - 1]

X_train, X_test, y_train, y_test = train_test_split(dataset, shuffle=True, test_size=0.25)

