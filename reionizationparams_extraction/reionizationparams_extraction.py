#!python

import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import Adagrad

from scikeras.wrappers import KerasRegressor

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

def create_model(optimizer='adam', hidden_layer_dim = 20, activation = "sigmoid"):
	# Create a neural network model
	model = Sequential()

	# First hidden layer 
	model.add(Dense(hidden_layer_dim, input_shape=(10,), activation=activation))

	# Second hidden layer 
	#model.add(Dense(hidden_layer_dim, activation='sigmoid'))

	# Output layer with 2 neurons (corresponding to Zeta and M_min respectively) 
	model.add(Dense(2, activation='linear'))

	# Compile the model  
	model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

	# Summary of the model
	model.summary()
	print("######## completed model setup #########")

	return model

def grid_search(X, y):
	model = KerasRegressor(model=create_model)
	param_grid = {
		"epochs": [80], "batch_size":[20], 
		"model__hidden_layer_dim": [20, 40, 180],
		"model__activation": ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
		"loss": ["mean_squared_error"],
		"optimizer": ['SGD', 'Adam'],
		"optimizer__learning_rate": [0.2],
	}
	grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10)
	grid_result = grid.fit(X, y)
	# summarize results
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))

def run(X, y):

	model = create_model()
	# Train the model
	model.fit(X_train, y_train, epochs=80, batch_size=20)

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


	plt.scatter(y_test[:, 0], y_pred[:, 0])
	plt.show()
	plt.scatter(y_test[:, 1], y_pred[:, 1])
	plt.show()




X, y = load_dataset("../21cm_simulation/output/ps-20240821225155")
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#run(X,y)
grid_search(X_train, y_train)

