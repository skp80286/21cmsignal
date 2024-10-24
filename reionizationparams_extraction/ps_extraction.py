#!python

import xgboost as xgb
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
from scikeras.wrappers import KerasRegressor
from sklearn import preprocessing

import tensorflow as tf

from datetime import datetime
import time
import argparse

parser = argparse.ArgumentParser(description='Build a neural network based ML model to predict 21cm powerspectrum from the total power spectrum containing instrument noise and foreground.')
parser.add_argument('-t', '--totalpsfile', type=str, default='../21cm_simulation/output/ps-noise-fg-80-7000.pkl', help='path to the total powerspectrum file')
parser.add_argument('-c', '--cosmopsfile', type=str, default='../21cm_simulation/output/ps-80-7000.pkl', help='path to the cosmological (21cm) powerspectrum file')
parser.add_argument('-r', '--previewrows', type=int, default=1, help='Number of rows to print on screen for a preview of input data.')
parser.add_argument('-s', '--samplesizes', type=int, default=1, help='Number of sample sizes to train on, to see the trend of learning and training loss.')
parser.add_argument('-m', '--savemodel', action='store_true', help='whether to save the trained model')
parser.add_argument('-e', '--trainingepochs', type=int, default=80, help='Number of epochs for training.')
parser.add_argument('-b', '--trainingbatchsize', type=int, default=6, help='batch size for training.')
parser.add_argument('-d', '--hiddenlayerdim', type=int, default=2048, help='Size of the first hidden layer in NN.')
parser.add_argument('-n', '--noninteractive', action='store_true', help='noninteractive mode. do not show plots.')
parser.add_argument('-p', '--saveprediction', action='store_true', help='whether to save the predicted powerspectrum along with corresponding reionization parameters.')
parser.add_argument('-f', '--predfilename', type=str, default=datetime.now().strftime("output/ps_extraction-pred-%Y%m%d%H%M%S.pkl"), help='file to save the predicted powerspectrum along with corresponding reionization parameters')
parser.add_argument('-a', '--predictall', action='store_true', help='whether to predict and save all samples for 21cm powerspectrum (default is to save only the test portion of samples).')
parser.add_argument('--modeltype', type=str, default='ANN', help='ML model to use: xgb or ANN')

args = parser.parse_args()
ps_size = -1

def load_dataset(totalps_filename, cosmops_filename):
    X = []
    y = []
    p = [] # reionization params
    kset = [] # array of 'k's
    lines = 0
    print(f'Reading files totalps={totalps_filename}, 21cmps={cosmops_filename}')
    with open(totalps_filename, 'rb') as total_f, open(cosmops_filename, 'rb') as cosmo_f:
        while True:
            try:
                total_e = pickle.load(total_f)
                cosmo_e = pickle.load(cosmo_f)
                
                params = [float(total_e['zeta']), float(total_e['m_min'])]
                
                total_ps = [float(x) for x in total_e['ps']]
                cosmo_ps = [float(x) for x in cosmo_e['ps']]
                ks = [float(x) for x in cosmo_e['k']]
                if (len(total_ps) != len(cosmo_ps)): 
                    raise ValueError(f"Length mismatch: total_ps ({len(total_ps)}) != cosmo_ps ({len(cosmo_ps)})")
                global ps_size
                if ps_size < 0: 
                    ps_size = len(total_ps)
                    print(f'Powerspectrum of size={ps_size}')
                if (len(total_ps) != ps_size):
                    raise ValueError(f"Length mismatch: total_ps ({len(total_ps)}) != {ps_size})")

                y.append(cosmo_ps)
                X.append(total_ps)
                p.append(params)
                kset.append(ks)
                    
                if lines < args.previewrows:
                    print(f'params={params}, total_ps={total_ps[:5]}..., cosmo_ps={cosmo_ps[:5]}...')
                lines = lines + 1
            except EOFError:
                break
    
    print(f"--- read {lines} lines ---")
    X, y, p, kset = (np.array(X), np.array(y), np.array(p), np.array(kset))
    
    valid_indices = np.all(~np.isnan(X) & ~np.isinf(X), axis=1) & np.all(~np.isnan(y) & ~np.isinf(y), axis=1)
    X = X[valid_indices]
    y = y[valid_indices]
    p = p[valid_indices]
    kset = kset[valid_indices]

    # Split the dataset and normalize
    split_index = int(len(X) * 0.8)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    # Deterministic split because we need the params for the test set
    X_train, X_test, y_train, y_test, p_train, p_test, k_test = X[:split_index], X[split_index:], y[:split_index], y[split_index:], p[:split_index], p[split_index:], kset[split_index:]
    return (X_train, X_test, y_train, y_test, p_train, p_test, k_test, X, y, p, kset)

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
    - Output layer: 80 neurons (for 21cm powerspectrum predictions)

    The model uses the Huber loss function and is compiled with the specified optimizer.
    """
    # Create a neural network model
    model = Sequential()

    global ps_size

    dim = hidden_layer_dim
    # First hidden layer 
    model.add(Dense(dim, input_shape=(ps_size,), activation=activation))

    dim = dim//2
    # Second hidden layer 
    model.add(Dense(dim, activation=activation2))

    dim = dim//2
    # third hidden layer 
    model.add(Dense(dim, activation=activation2))

    dim = dim//2
    # fourth hidden layer 
    model.add(Dense(dim, activation=activation2))

    dim = dim//2
    # fifth hidden layer 
    model.add(Dense(dim, activation=activation2))

    # Output layer with 80 neurons (corresponding to Zeta and M_min respectively) 
    model.add(Dense(ps_size, activation='linear'))

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
    keras_filename = datetime.now().strftime("output/ps-extr-model-%Y%m%d%H%M%S.keras")
    print(f'Saving model to: {keras_filename}')
    model.save(keras_filename)

def save_prediction(y_pred, p_test, kset):
    with open(args.predfilename, 'a+b') as f:  # open a text file
        for y, p, k in zip(y_pred, p_test, kset):
            pickle.dump({"X": y, "y": p, "k": k}, f)


def run(X_train, X_test, y_train, y_test, p_train, p_test, k_test, X_all, y_all, p_all, k_all):

    # Split the data into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Define a list to store training loss and validation loss
    training_loss = []
    test_loss = []
    r2_scores = []
    
    """
    We can specify sample sizes in ascending order here if we want to see 
    the training and test loss curve trend as sample size increases. By default, 
    we will train for only the full length of the available samples.
    """
    l = len(X_train)//args.samplesizes
    sample_sizes = []
    for i in range(args.samplesizes - 1):
        sample_sizes.append((i+1) * l)
    sample_sizes.append(len(X_train))  
    print(f'l={l}, len(Xtrain)={len(X_train)}, args.samplesizes={args.samplesizes}')
    print(f'sample_sizes={sample_sizes}')
    
    y_pred = None
    history = None
    rms_scores = None
    y_train_subset = None
    errors = None

    optimizer='Adagrad'
    learning_rate = 0.0001
    hidden_layer_dim = args.hiddenlayerdim
    activation = "tanh"
    activation2 = "linear"
    
    # Train model with different sample sizes
    for size in sample_sizes:
        print (f'## Sample size: {size}')
        X_train_subset = X_train[:size]
        y_train_subset = y_train[:size]

        model = None
        if (args.modeltype == 'xgb'):
            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ) 
            model.fit(X_train_subset, y_train_subset)
        else: # ANN
            model = create_model(
                optimizer=optimizer, learning_rate = learning_rate, 
                hidden_layer_dim = hidden_layer_dim, 
                activation = activation, activation2 = activation2
                )
            history = model.fit(X_train_subset, y_train_subset, epochs=args.trainingepochs, batch_size=args.trainingbatchsize, shuffle=True)
            training_loss.append(history.history['loss'][-1])  # Store last training loss for each iteration
        #validation_loss.append(history.history['val_loss'][-1])  
        # Test the model
        if args.predictall:
            y_pred = model.predict(X_all)
            y_test = y_all # y_test is used for score calculations
            save_prediction(y_pred, p_all, k_all)
        else:
            y_pred = model.predict(X_test)
            save_prediction(y_pred, p_test, k_test)

        # Calculate R2 scores
        r2 = r2_score(y_test, y_pred)
        print(f"R2 Score: {r2}")
        r2_scores.append(r2)
        # Calculate rmse scores
        rms_score = mean_squared_error(y_test, y_pred) 
        #rms_scores_percent = np.sqrt(rms_scores) * 100 / np.mean(y_test, axis=0)
        print(f"RMS Error: {rms_score}")
        test_loss.append(rms_score)

    # Plot the results
    #print(errors[:,0].shape)
    #print(errors[:,1].shape)
    #print(y_test[:,0].shape)
    #print(y_test[:,1].shape)
    #print(errors)
    """
    plt.scatter(y_test.flatten(), y_test[:, 1], c=errors[:,0])
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
    """

    if len(sample_sizes) > 1:
        plt.plot(sample_sizes, training_loss, label='Training Loss')
        #plt.plot(sample_sizes, validation_loss, label='Validation Loss')
        plt.plot(sample_sizes, test_loss, label='Test Loss')
        plt.plot(sample_sizes, r2_scores, label='R2 Score')
        plt.xlabel('Number of Samples')
        plt.ylabel('Loss')
        plt.title(f'opt={optimizer},lr={learning_rate},hl_dim={hidden_layer_dim},activ={activation},activ2={activation2}')
        plt.legend()
        plt.show()

    ## Plot the training and validation loss
    if not args.noninteractive and history != None:
        plt.plot(history.history['loss'])
        ##plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        #plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()

    ## Test the model
    #y_pred = model.predict(X_test)

    ## Calculate R2 scores
    #r2 = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(2)]
    #print("R2 Score: " + str(r2))

    ## Calculate rmse scores
    #rms_scores = np.sqrt([mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(2)])
    #rms_scores_percent = rms_scores * 100 / np.mean(y_test, axis=0)
    #print("RMS Error: " + str(rms_scores_percent))

    # Plot RMS scores
    #plt.bar(['zeta', 'mmin'], rms_scores_percent)
    #plt.ylim(0, 10)
    #plt.title('% RMS Error for Predictions')
    #plt.ylabel('% RMS Error (RMSE*100/mean)')
    #for i, v in enumerate(rms_scores_percent):
    #    plt.text(i, v, "{:.2f}%".format(v), ha='center', va='bottom')
    #plt.show()

    if not args.noninteractive:
        plt.scatter(y_test.flatten(), y_pred.flatten())
        plt.title('Predictions vs True Values for Power')
        plt.ylabel('Prediction')
        plt.xlabel('True Value')
        plt.show()

    if (args.savemodel): save_model(model)

tf.config.list_physical_devices('GPU')
print("### GPU Enabled!!!")
#X, y = load_dataset("../21cm_simulation/output/ps-consolidated")
X_train, X_test, y_train, y_test, p_train, p_test, k_test, X_all, y_all, p_all, k_all = load_dataset(args.totalpsfile, args.cosmopsfile)
#X_train, X_test, y_train, y_test = load_dataset("../21cm_simulation/output/ps-noise-fg-80-7000.pkl")
#X_train, X_test, y_train, y_test = load_dataset("../21cm_simulation/output/ps-noise-20240929160608.pkl")
#X_train, X_test, y_train, y_test = load_dataset("../21cm_simulation/output/ps-noise-20240925215505.pkl")
#X_train, X_test, y_train, y_test = load_dataset("../21cm_simulation/output/ps-80-7000.pkl")
start_time = time.time()
run(X_train, X_test, y_train, y_test, p_train, p_test, k_test, X_all, y_all, p_all, k_all)
print(f'args={args}')
print(f'Finished run in {time.time()-start_time} seconds.')
#grid_search(X, y)
