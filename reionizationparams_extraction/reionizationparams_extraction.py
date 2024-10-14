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
from keras.regularizers import l1_l2, l1
from keras.losses import huber
from keras.saving import load_model
from scikeras.wrappers import KerasRegressor
from sklearn import preprocessing

import tensorflow as tf

from datetime import datetime
import time
import argparse
import glob

parser = argparse.ArgumentParser(description='Build a neural network based ML model to predict reionization parameters from power spectrum of 21 cm brightness temperature.')
parser.add_argument('-i', '--inputfile', type=str, default='../21cm_simulation/saved_output/newsim/ps-*.pkl', help='path pattern for the powerspectrum files')
parser.add_argument('-x', '--excludeouter', action='store_true', help='exclude outer range of reionization parameters in the dataset to get more accurate results.')
parser.add_argument('-r', '--previewrows', type=int, default=1, help='Number of rows to print on screen for a preview of input data.')
parser.add_argument('-d', '--runmode', type=str, default='train_test', help='one of train_test, test_only, grid_search')
parser.add_argument('-m', '--modelfile', type=str, default='./output/reion-par-extr-model.keras', help='saved NN model to use for testing')
parser.add_argument('-s', '--numsamplebatches', type=int, default=1, help='Number of batches of sample data to use for plotting learning curve by sample size.')
parser.add_argument('-e', '--epochs', type=int, default=80, help='Number of epochs in training.')
parser.add_argument('-b', '--batchsize', type=int, default=6, help='Batch size in training.')

args = parser.parse_args()

def load_testdataset(filename):
    X = []
    y = []
    lines = 0
    with open(filename, 'rb') as f:  # open a text file
        while 1:
            try:
                e = pickle.load(f)
                print("Fields in e:", list(e.keys()))
                X.append(e['X'])
                y.append(e['y'])
                if lines < args.previewrows: print(f'X={e["X"]}, y={e["y"]}')
                lines = lines + 1
            except EOFError:
                break
    print("--- read %d lines ---" % lines)
    X, y = (np.array(X), np.array(y))
    # Data validation and cleaning
    valid_indices = np.all(~np.isnan(X) & ~np.isinf(X), axis=1) & np.all(~np.isnan(y) & ~np.isinf(y), axis=1)
    X = X[valid_indices]
    y = y[valid_indices]
    return (X, y)

def load_dataset(file_pattern):
    X = []
    y = []
    lines = 0
    
    for filename in glob.glob(file_pattern):
        print(f"Reading file: {filename}")
        with open(filename, 'rb') as f:
            while True:
                try:
                    e = pickle.load(f)
                    #print("Fields in e:", list(e.keys()))
                    # We scale the M_min to bring it to similar level
                    # as Zeta. This is to avoid the skewing of model to optimizing 
                    # one of the outputs at the expense of the other
                    #params = [float(e['zeta']), float(e['m_min'])*90-320]
                    if ('X' in e):
                        X.append(e['X'])
                        y.append(e['y'])
                    else:
                        params = [float(e['zeta']), float(e['m_min'])]
                        #if True:
                        if((not args.excludeouter) or
                            (params[1] > 4.25 and params[0] > 25 and params[0] < 150)):
                            y.append(params)
                            ps = [float(x) for x in e['ps']]
                            X.append(ps) 
                    if lines < args.previewrows: print(f'params={y[-1]}, ps={X[-1]}')
                    lines = lines + 1
                except EOFError:
                    break
    
    print(f"--- read {lines} lines from {len(glob.glob(file_pattern))} files ---")
    
    X, y = (np.array(X), np.array(y))
    # Data validation and cleaning
    valid_indices = np.all(~np.isnan(X) & ~np.isinf(X), axis=1) & np.all(~np.isnan(y) & ~np.isinf(y), axis=1)
    X = X[valid_indices]
    y = y[valid_indices]

    # Split the dataset and normalize
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    return (X_train, X_test, y_train, y_test)

## Not used
def my_loss_fn(y_true, y_pred):
    SS_res = tf.square(y_true - y_pred)
    SS_tot = tf.square(y_true - tf.reduce_mean(y_true))
    r2_inv = tf.reduce_sum(SS_res / (SS_tot + tf.constant(1e-7, dtype=tf.float32)))
    return r2_inv

def create_model(optimizer='Adgrad', learning_rate = 0.0001, hidden_layer_dim = 16, 
                 activation = "tanh", activation2 = "leaky_relu"):
    """"
    Create a neural network model with specified paramters.

    This function creates a Sequential model with multiple hidden layers and an output layer.
    The number of neurons in each hidden layer is halved progressively.

    Parameters:
    optimizer (str): The optimizer to use for training. Default is 'Adgrad'.
    learning_rate (float): The learning rate for the optimizer. Default is 0.0001.
    hidden_layer_dim (int): The number of neurons in the first hidden layer. Default is 16. Subsequent 
                            layers have a dimension that is half of the previous layer.
    activation (str): The activation function for the first hidden layer. Default is "tanh".
    activation2 (str): The activation function for subsequent hidden layers. Default is "leaky_relu".

    Returns:
    keras.models.Sequential: A compiled Keras Sequential model ready for training.

    The model architecture:
    - Input layer: 80 neurons (matching the input shape)
    - 5 hidden layers with progressively halved dimensions
    - Output layer: 2 neurons (for Zeta and M_min predictions)

    The model uses the Huber loss function and is compiled with the specified optimizer.
    """
    # Create a neural network model
    model = Sequential()

    dim = hidden_layer_dim
    # First hidden layer 
    model.add(Dense(dim, input_shape=(80,), activation=activation))

    #dim = dim//2
    # Second hidden layer 
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
    model.compile(loss=huber, optimizer=opt, metrics=['accuracy'])

    # Summary of the model
    model.summary()
    print("######## completed model setup #########")

    return model

def grid_search(X, y):
    model = KerasRegressor(model=create_model)
    param_grid = {
        "epochs": [64], "batch_size":[8], 
        "model__hidden_layer_dim": [1024, 2048],
        "model__activation": ['tanh','sigmoid'],
        "model__activation2": ['leaky_relu', 'linear'],
        "loss": ['mean_squared_error'],
        "optimizer": ['Adam'],
        "model__learning_rate": [0.0001],
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

def save_model(model):
    # Save the model architecture and weights
    keras_filename = datetime.now().strftime("output/reion-par-extr-model-%Y%m%d%H%M%S.keras")
    print(f'Saving model to: {keras_filename}')
    model_json = model.save(keras_filename)

def summarize_test(y_pred, y_test):
    errors = (y_pred - y_test)**2/y_test**2

    # Calculate R2 scores
    r2 = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(2)]
    print("R2 Score: " + str(r2))
    # Calculate rmse scores
    rms_scores = [mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(2)]
    rms_scores_percent = np.sqrt(rms_scores) * 100 / np.mean(y_test, axis=0)
    print("RMS Error: " + str(rms_scores_percent))
    plt.scatter(y_test[:, 0], y_test[:, 1], c=errors[:,0])
    plt.xlabel('Zeta')
    plt.ylabel('M_min')
    plt.title('Zeta Error')
    plt.colorbar()
    plt.show()
    plt.scatter(y_test[:, 0], y_test[:, 1], c=errors[:,1])
    plt.xlabel('Zeta')
    plt.ylabel('M_min')
    plt.title('M_min Error')
    plt.colorbar()
    plt.show()
    plt.scatter(y_test[:, 0], y_pred[:, 0])
    plt.title('Predictions vs True Values for Zeta')
    plt.ylabel('Prediction')
    plt.xlabel('True Value')
    plt.show()
    plt.scatter(y_test[:, 1], y_pred[:, 1])
    plt.title('Predictions vs True Values for M_min')
    plt.ylabel('Prediction')
    plt.xlabel('True Value')
    plt.show()
    ## Train the model
    #history = model.fit(X_train, y_train, epochs=512, batch_size=11)
    ## Plot the training and validation los

def run(X_train, X_test, y_train, y_test):

    # Split the data into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Define a list to store training loss and validation loss
    training_loss = []
    validation_loss = []
    test_loss = []
    r2_scores = []
    
    """
    We can specify sample sizes in ascending order here if we want to see 
    the training and test loss curve trend as sample size increases. By default, 
    we will train for only the full length of the available samples.
    """
    num_samples = len(X_train)
    min_sample_size = num_samples//args.numsamplebatches
    sample_sizes = []
    for i in range(args.numsamplebatches - 1):
        sample_sizes.append((i+1)*min_sample_size)
    sample_sizes.append(num_samples)  
    
    y_pred = None
    history = None
    rms_scores = None
    y_train_subset = None
    errors = None

    optimizer='Adagrad'
    learning_rate = 0.0001
    hidden_layer_dim = 2048
    activation = "tanh"
    activation2 = "linear"
    
    # Train model with different sample sizes
    for size in sample_sizes:
        print (f'## Sample size: {size}')
        X_train_subset = X_train[:size]
        y_train_subset = y_train[:size]

        model = create_model(
            optimizer=optimizer, learning_rate = learning_rate, 
            hidden_layer_dim = hidden_layer_dim, 
            activation = activation, activation2 = activation2
            )
        history = model.fit(X_train_subset, y_train_subset, epochs=args.epochs, batch_size=args.batchsize, shuffle=True)
            
        training_loss.append(history.history['loss'][-1])  # Store last training loss for each iteration
        #validation_loss.append(history.history['val_loss'][-1])  
        # Test the model
        y_pred = model.predict(X_test)

        errors = (y_pred - y_test)**2/y_test**2

        # Calculate R2 scores
        r2 = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(2)]
        print("R2 Score: " + str(r2))
        r2_scores.append((r2[0]+r2[1]))
        # Calculate rmse scores
        rms_scores = [mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(2)]
        rms_scores_percent = np.sqrt(rms_scores) * 100 / np.mean(y_test, axis=0)
        print("RMS Error: " + str(rms_scores_percent))
        test_loss.append(rms_scores[0]+rms_scores[1])

    # Plot the results
    #print(errors[:,0].shape)
    #print(errors[:,1].shape)
    #print(y_test[:,0].shape)
    #print(y_test[:,1].shape)
    #print(errors)
    summarize_test(y_pred, y_test)
    plt.scatter(y_test[:, 0], y_test[:, 1], c=np.mean(X_test, axis=1))
    plt.xlabel('Zeta')
    plt.ylabel('M_min')
    plt.title('Mean power')
    plt.colorbar()
    plt.show()
    plt.scatter(y_test[:, 0], y_test[:, 1], c=np.var(X_test, axis=1))
    plt.xlabel('Zeta')
    plt.ylabel('M_min')
    plt.title('Variance in power')
    plt.colorbar()
    plt.show()

    plt.plot(sample_sizes, training_loss, label='Training Loss')
    #plt.plot(sample_sizes, validation_loss, label='Validation Loss')
    plt.plot(sample_sizes, test_loss, label='Test Loss')
    plt.plot(sample_sizes, r2_scores, label='R2 Score')
    plt.xlabel('Number of Samples')
    plt.ylabel('Loss')
    plt.title(f'opt={optimizer},lr={learning_rate},hl_dim={hidden_layer_dim},activ={activation},activ2={activation2}')
    plt.legend()
    plt.show()

    ## Train the model
    #history = model.fit(X_train, y_train, epochs=512, batch_size=11)
    ## Plot the training and validation loss
    plt.plot(history.history['loss'])
    ##plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    #plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

    save_model(model)

# main code start here
tf.config.list_physical_devices('GPU')
print("### GPU Enabled!!!")
if not args.runmode == "test_only":
    #X, y = load_dataset("../21cm_simulation/output/ps-consolidated")
    X_train, X_test, y_train, y_test = load_dataset(args.inputfile)
    #X_train, X_test, y_train, y_test = load_dataset("../21cm_simulation/output/ps-noise-fg-80-7000.pkl")
    #X_train, X_test, y_train, y_test = load_dataset("../21cm_simulation/output/ps-noise-20240929160608.pkl")
    #X_train, X_test, y_train, y_test = load_dataset("../21cm_simulation/output/ps-noise-20240925215505.pkl")
    #X_train, X_test, y_train, y_test = load_dataset("../21cm_simulation/output/ps-80-7000.pkl")
    if args.runmode == "train_test":
        run(X_train, X_test, y_train, y_test)
    elif args.runmode == "grid_search":
        grid_search(X_train, y_train)
else: # testonly
    X_test, y_test = load_testdataset(args.inputfile)
    model = load_model(args.modelfile)
    y_pred = model.predict(X_test)
    summarize_test(y_pred, y_test)