#!python

import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

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

def create_model(optimizer='adam', hidden_layer_dim = 256, activation = "sigmoid", activation2 = "leaky_relu"):
	# Create a neural network model
	model = Sequential()

	dim = hidden_layer_dim
	# First hidden layer 
	model.add(Dense(dim, input_shape=(20,), activation=activation))

	dim = dim//2
	# Second hidden layer 
	model.add(Dense(dim, activation=activation2))

	dim = dim//2
	# third hidden layer 
	model.add(Dense(dim, activation=activation2))

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
		"epochs": [64, 132], "batch_size":[8, 16], 
		"model__hidden_layer_dim": [256, 512],
		"model__activation": ['sigmoid', 'tanh'],
		"model__activation2": ['leaky_relu', 'linear'],
		"loss": ["mean_squared_error"],
		"optimizer": ['Adam'],
		"optimizer__learning_rate": [0.5, 0.2, 1.0],
	}
	grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
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
	model.fit(X_train, y_train, epochs=128, batch_size=8)

	# Test the model
	y_pred = model.predict(X_test)

	# Calculate rmse scores
	rms_scores = np.sqrt([mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(2)])
	rms_scores_percent = rms_scores * 100 / np.mean(y_test, axis=0)
	print("RMS Error: " + str(rms_scores_percent))

	# Plot RMS scores
	plt.bar(['zeta', 'mmin'], rms_scores_percent)
	plt.ylim(0, 10)
	plt.title('% RMS Error for Predictions')
	plt.ylabel('% RMS Error (RMSE*100/mean)')
	for i, v in enumerate(rms_scores_percent):
		plt.text(i, v, "{:.2f}%".format(v), ha='center', va='bottom')
	plt.show()
	plt.show()


	plt.scatter(y_pred[:, 0], y_test[:, 0])
	plt.title('Predictions vs True Values for Zeta')
	plt.ylabel('Prediction')
	plt.xlabel('True Value')
	plt.show()
	plt.scatter(y_pred[:, 1], y_test[:, 1])
	plt.title('Predictions vs True Values for M_min')
	plt.ylabel('Prediction')
	plt.xlabel('True Value')
	plt.show()




#X, y = load_dataset("../21cm_simulation/output/ps-consolidated")
X, y = load_dataset("../21cm_simulation/output/ps-20240908194837.pkl")
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
run(X,y)
#grid_search(X_train, y_train)

