import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
sys.path.append('ics-anomaly-detection-main/detector')  # Add folder to system path

from detector import rf, svr, xgb, ada, cnn, lstm, ae
import backtime
import data_loader, main_eval, utils
import time
from sklearn.metrics import f1_score
import pickle

#progressing bar
from itertools import product
from tqdm import tqdm

import numpy as np

def detection_hyperparameter_tuning(detector, X_test, X_val, Y_test):
    #Tuning detection hyperparameters
    detection_param = {
        'quantile' : [0.9, 0.905, 0.95, 0.99, 0.995, 0.9995],
        'window' : [1, 10, 15, 20, 50, 100, 200]
    }    
    param_combinations = list(product(detection_param['quantile'], detection_param['window']))
    best_f1= 0
    best_quantile= 0
    best_window= 0
    
    print("(+) Tuning detection hyperparameters...")
    for quantile, window in tqdm(param_combinations, desc="Tuning params", total=len(param_combinations)):
        Y_pred= detector.detect(X_test, X_val, quantile, window)
        f1 = f1_score(Y_test, Y_pred, average= 'macro')
        if f1 > best_f1:
            best_f1= f1   
            best_quantile= quantile
            best_window= window
    print(f"Best detection hyperparameters set to quantile={best_quantile}, window={best_window}")
    return best_quantile, best_window
            
def backtime_attack(model_name, dataset_name):
    #Load original dataset
    X_train, _ = data_loader.load_train_data(dataset_name)
    x_test, y_test, _ = data_loader.load_test_data(dataset_name)
    
    #split test data into validation data and test data
    X_val, X_test, _, Y_test = utils.custom_train_test_split(dataset_name, x_test, y_test)
    
    # Create poisoned dataset
    atk_path = f"X_ATK_{dataset_name}.pkl"
    backdoor_features = [0, 2, 5]
    if not os.path.exists(atk_path):
        # Define feature indices to backdoor
        backtime.attack(dataset_name, backdoor_features)
    
    #Load poisoned dataset
    with open(atk_path, 'rb') as f:
        X_ATK = pickle.load(f)
    print(f"[BackTime] Loaded existing poisoned dataset from {atk_path}.")
    print("[BackTime] Loaded X_ATK with shape:", X_ATK.shape) 
    
    backtime.plot_full_timeseries_clean(X_train, X_ATK, backdoor_features)

    #Set detector and hyperparameters    
    detector = ae.AE(nI=X_ATK.shape[1])
    
    #Train model
    option= input("Enable tuning (Y/n): ").upper()
    if option== 'Y':
        detector.hyperparameter_tuning(X_ATK, X_val)
    else:
        detector.train(X_ATK)
    
    #Tuning detection hyperparameters
    best_quantile, best_window= detection_hyperparameter_tuning(detector, X_test, X_val, Y_test)
    
    #Detection
    Y_pred= detector.detect(X_test, X_val, best_quantile, best_window)
    return Y_test, Y_pred

def reconstructed_detector(model_name, dataset_name):
    #load train, test data
    X_train, _ = data_loader.load_train_data(dataset_name)
    x_test, y_test, _ = data_loader.load_test_data(dataset_name)
    
    #split test data into validation data and test data
    X_val, X_test, _, Y_test = utils.custom_train_test_split(dataset_name, x_test, y_test)
    
    #Set detector and hyperparameters    
    if model_name == 'SVR':   
        detector = svr.SVR()
    elif model_name == 'RF':
        detector = rf.RF()
    elif model_name == 'ADA':
        detector = ada.AdaBoost()
    elif model_name == 'XGB':
        detector = xgb.XGBoost()
    elif model_name == 'CNN':
        detector = cnn.CNN(input_length=X_train.shape[1])
    elif model_name == 'LSTM':
        detector = lstm.LSTM(nI=X_train.shape[1])
    elif model_name == 'AE':
        detector = ae.AE(nI=X_train.shape[1])
    else:
        raise ValueError('Unsupported model')
    
    #Train model
    option= input("Enable tuning (Y/n): ").upper()
    if option== 'Y':
        detector.hyperparameter_tuning(X_train, X_val)
    else:
        detector.train(X_train)
    
    #Tuning detection hyperparameters
    best_quantile, best_window= detection_hyperparameter_tuning(detector, X_test, X_val, Y_test)
    
    #Detection
    print(f"Best detection hyperparameters set to quantile={best_quantile}, window={best_window}")
    Y_pred= detector.detect(X_test, X_val, best_quantile, best_window)
    return Y_test, Y_pred

if __name__ == '__main__':
    model_name = input("Enter model name (SVR/RF/ADA/XGB/CNN/LSTM/AE/BACKTIME): ").upper()
    dataset_name = input("Enter dataset name (BATADAL/SWAT/WADI): ").upper()
    start_time = time.time()
    if model_name == 'BACKTIME':
        y_test, y_pred = backtime_attack(model_name, dataset_name)
    else:
        y_test, y_pred = reconstructed_detector(model_name, dataset_name)
    end_time= time.time()
    # Evaluate
    #main_eval.plot_evaluation(model_name, y_test, y_pred)
    main_eval.show_classification_report(y_test, y_pred)
    print(f"Execution time: {end_time-start_time:.2f}s")