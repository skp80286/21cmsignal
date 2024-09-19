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
from keras.optimizers import Adadelta
from keras.optimizers import RMSprop
from keras.regularizers import l1
from scikeras.wrappers import KerasRegressor
from sklearn import preprocessing

import optuna

def load_dataset(filename):
	X = []
	y = []
	lines = 0
	with open(filename, 'rb') as f:  # open a text file
		while 1:
			try:
				e = pickle.load(f)
				params = [float(e['zeta']), float(e['m_min'])]
				y.append(params)
				ps = [float(x) for x in e['ps']]
				X.append(ps) 
				#print(f'params={params}, ps={ps}')
				lines = lines + 1
			except EOFError:
				break
	print("--- read %d lines ---" % lines)
	#X = preprocessing.normalize(X)
	return (np.array(X), np.array( y))

def create_model(optimizer='Adagrad', learning_rate = 0.0001, hidden_layer_dim = 2048, activation = "tanh", activation2 = "leaky_relu",
				 bias_regularization=1e-5):
	# Create a neural network model
	model = Sequential()

	dim = hidden_layer_dim
	# First hidden layer 
	model.add(Dense(dim, input_shape=(20,), activation=activation, bias_regularizer=l1(bias_regularization)))

	dim = dim//2
	# Second hidden layer 
	model.add(Dense(dim, activation=activation2))

	dim = dim//2
	# third hidden layer 
	model.add(Dense(dim, activation=activation2))

	dim = dim//2
	# fourth hidden layer 
	model.add(Dense(dim, activation=activation2))


	# Output layer with 2 neurons (corresponding to Zeta and M_min respectively) 
	model.add(Dense(2, activation='linear'))

	# setup optimizer
	opt = None
	if optimizer=="SGD":
		opt = SGD(learning_rate=learning_rate)
	if optimizer=="Adagrad":
		opt = Adagrad(learning_rate=learning_rate)
	if optimizer=="Adadelta":
		opt = Adadelta(learning_rate=learning_rate)
	if optimizer=="RMSprop":
		opt = RMSprop(learning_rate=learning_rate)
	else: #optimizer=="Adam":
		opt = Adam(learning_rate=learning_rate)

	# Compile the model  
	model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

	# Summary of the model
	model.summary()
	print("######## completed model setup #########")

	return model

def objective(trial):

	X, y = load_dataset("../21cm_simulation/output/ps-20240918122730.pkl")
	# Split the data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=49)

	model = create_model(
		optimizer=trial.suggest_categorical('optimizaer', ['Adam']),
		learning_rate=0.0002737675618455643, #trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
		hidden_layer_dim=1024, #suggest_int('hidden_layer_dim', 512, 1024, log=True),
		activation=trial.suggest_categorical('activation', ['tanh']),
		activation2=trial.suggest_categorical('activation2', ['leaky_relu']),
		bias_regularization=trial.suggest_float('bias_regularization', 1e-6, 1e-2, log=True)
		)
	# Train the model
	model.fit(X_train, y_train, epochs=128, batch_size=10)#trial.suggest_int('batch_size', 8, 32, log=True))

	# Test the model
	y_pred = model.predict(X_test)

	# Calculate R2 scores
	r2 = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(2)]
	print("R2 Score: " + str(r2))

	return r2[0]+r2[1]

if __name__ == "__main__":
	study = optuna.create_study(direction="maximize")
	study.optimize(objective, n_trials=100, timeout=1200)

	print("Number of finished trials: {}".format(len(study.trials)))

	for trial in study.trials:
		print("  Value: {}".format(trial.value))
		print("  Params: ")
		for key, value in trial.params.items():
			print("    {}: {}".format(key, value))

	print("Best trial:")
	trial = study.best_trial

	print("  Value: {}".format(trial.value))
	print("  Params: ")
	for key, value in trial.params.items():
		print("    {}: {}".format(key, value))