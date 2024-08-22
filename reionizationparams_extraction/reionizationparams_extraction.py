#!python

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

import pickle

def load_dataset(filename):
	X = []
	y = []
	lines = 0
	with open(filename, 'rb') as f:  # open a text file
		while 1:
			try:
				e = pickle.load(f)
				y.append([e['zeta'], e['m_min']])
				X.append( e['ps']) 
				lines = lines + 1
			except EOFError:
				break
	print("--- read %d lines ---" % lines)
	return (np.array(X), np.array( y))

X, y = load_dataset("../21cm_simulation/output/ps-20240821225155")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(y_test)
print(X_test)

# Initialize the neural network model
model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Calculate R² scores
r2_scores = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(2)]

# Plot R² scores
plt.bar(['zeta', 'mmin'], r2_scores)
plt.ylim(0, 1)
plt.title('R² Scores for Predictions')
plt.ylabel('R² Score')
plt.show()


plt.scatter(y_pred[:, 0], y_test[:, 0])
plt.scatter(y_pred[:, 1], y_test[:, 1])
plt.show()

