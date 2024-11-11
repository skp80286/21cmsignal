#!python

import xgboost as xgb
from xgboost import plot_tree

import numpy as np
import pickle

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import argparse
import glob
import math
from datetime import datetime

parser = argparse.ArgumentParser(description='Build a neural network based ML model to predict reionization parameters from power spectrum of 21 cm brightness temperature.')
parser.add_argument('-i', '--inputfile', type=str, default='../21cm_simulation/saved_output/newsim/ps-*.pkl', help='path pattern for the powerspectrum files')
parser.add_argument('-x', '--excludeouter', action='store_true', help='exclude outer range of reionization parameters in the dataset to get more accurate results.')
parser.add_argument('-r', '--previewrows', type=int, default=1, help='Number of rows to print on screen for a preview of input data.')
parser.add_argument('-d', '--runmode', type=str, default='train_test', help='one of train_test, test_only, grid_search, plot_only')
parser.add_argument('-m', '--modelfile', type=str, default='./output/reion-par-extr-model.keras', help='saved NN model to use for testing')
parser.add_argument('-s', '--numsamplebatches', type=int, default=1, help='Number of batches of sample data to use for plotting learning curve by sample size.')
parser.add_argument('-e', '--epochs', type=int, default=80, help='Number of epochs in training.')
parser.add_argument('-b', '--batchsize', type=int, default=6, help='Batch size in training.')
parser.add_argument('-l', '--logpowerspectrum', action='store_true', help='use the logarithm of the powerspectrum')

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
    kset = []
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
                        kset.append(e['k'])
                        if lines < args.previewrows: print(f'params={y[-1]}, k={kset[-1]}, ps={X[-1]}')
                    else:
                        params = [float(e['zeta']), float(e['m_min'])]
                        #if True:
                        if((not args.excludeouter) or
                            (params[1] > 4.25 and params[0] > 25 and params[0] < 150)):
                            y.append(params)

                            ps = e['ps']
                            ks = e['k']
                            for i, (p, k) in enumerate(zip(ps, ks)):
                                if (p != 0 and k !=0):
                                    ps[i] = p/k**3
                            
                            if args.logpowerspectrum:
                                ps = [math.log10(x+1) for x in ps]
                            X.append(ps) 
                            kset.append(ks)
                            if lines < args.previewrows: print(f'params={y[-1]}, k={kset[-1]}, ps={X[-1]}')
                    lines = lines + 1
                except EOFError:
                    break
    
    print(f"--- read {lines} lines from {len(glob.glob(file_pattern))} files ---")
    
    X, y, kset = (np.array(X), np.array(y), np.array(kset))
    # Data validation and cleaning
    valid_indices = np.all(~np.isnan(X) & ~np.isinf(X), axis=1) & np.all(~np.isnan(y) & ~np.isinf(y), axis=1)
    X = X[valid_indices]
    y = y[valid_indices]
    kset = kset[valid_indices]

    return (X, y, kset)

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
def plot_power_spectra(X, y, kset):
    print(f'shapes X:{X.shape} y:{y.shape} kset:{kset.shape}')
    plt.rcParams['figure.figsize'] = [15, 6]
    plt.title('Spherically averaged power spectra.')
    for i, (row_x, row_y, row_k) in enumerate(zip(X, y, kset)):
        #label = f'Zeta:{row_y[0]:.2f}-M_min:{row_y[1]:.2f}'
        plt.loglog(row_k[1:40], row_x[1:40]) #, label=label)
        #plt.annotate(text=label, xy=(row_x[2*i+1], row_k[2*i+1]))
    plt.xlabel('k (Mpc$^{-1}$)')
    plt.ylabel('P(k) k$^3$/(2*pi^2)')
    #plt.legend(loc='lower right')
    plt.show()

def save_model(model):
    # Save the model architecture and weights
    model_filename = datetime.now().strftime("output/xgboost-extr-model-%Y%m%d%H%M%S.json")
    print(f'Saving model to: {model_filename}')
    model_json = model.save_model(model_filename)

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

        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )

        history = model.fit(X_train_subset, y_train_subset)

        #print(f"History: {history}")
            
        #training_loss.append(history.best_score)  # Store last training loss for each iteration
        #validation_loss.append(history.history['val_loss'][-1])  
        # Test the model
        y_pred = model.predict(X_test)

        errors = (y_pred - y_test)**2/y_test**2

        # Calculate R2 scores
        r2 = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(2)]
        print("R2 Score: " + str(r2))
        r2_scores.append(50*(r2[0]+r2[1]))
        # Calculate rmse scores
        rms_scores = [mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(2)]
        rms_scores_percent = np.sqrt(rms_scores) * 100 / np.mean(y_test, axis=0)
        print("RMS Error: " + str(rms_scores_percent))
        test_loss.append(0.5*(rms_scores_percent[0]+rms_scores_percent[1]))

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

    #plt.plot(sample_sizes, training_loss, label='Training Loss')
    #plt.plot(sample_sizes, validation_loss, label='Validation Loss')
    plt.plot(sample_sizes, test_loss, label='Test Loss')
    plt.plot(sample_sizes, r2_scores, label='R2 Score')
    plt.xlabel('Number of Samples')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title(f'XGB Learning Trend')
    plt.legend()
    plt.show()

    ## Train the model
    #history = model.fit(X_train, y_train, epochs=512, batch_size=11)
    ## Plot the training and validation loss
    """
    #plt.plot(history.history['loss'])
    ##plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    #plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
    """
    print('Plotting Decision Tree')
    plot_tree(model)
    plt.savefig("output/xgboost_tree.png", dpi=600) 
    save_model(model)

# main code start here
#tf.config.list_physical_devices('GPU')
print("### GPU Enabled!!!")
if not args.runmode == "test_only":
    #X, y = load_dataset("../21cm_simulation/output/ps-consolidated")
    X, y, kset = load_dataset(args.inputfile)
    # Split the dataset and normalize
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    #X_train, X_test, y_train, y_test = load_dataset("../21cm_simulation/output/ps-noise-fg-80-7000.pkl")
    #X_train, X_test, y_train, y_test = load_dataset("../21cm_simulation/output/ps-noise-20240929160608.pkl")
    #X_train, X_test, y_train, y_test = load_dataset("../21cm_simulation/output/ps-noise-20240925215505.pkl")
    #X_train, X_test, y_train, y_test = load_dataset("../21cm_simulation/output/ps-80-7000.pkl")
    if args.runmode == "train_test":
        run(X_train, X_test, y_train, y_test)
    elif args.runmode == "plot_only":
        plot_power_spectra(X, y, kset)
else: # testonly
    X_test, y_test = load_testdataset(args.inputfile)
    model = xgb.XGBRegressor()
    model.load_model(args.modelfile)
    y_pred = model.predict(X_test)
    summarize_test(y_pred, y_test)