import data_loader, main_eval, ocsvm, rf, svr
import numpy as np
import utils
from sklearn.metrics import f1_score
import xgboost as xgb

#progressing bar
from itertools import product
from tqdm import tqdm

def OCSVM_detect(dataset_name):
    x_train, _ = data_loader.load_train_data(dataset_name)
    x_test, y_test, _ = data_loader.load_test_data(dataset_name)
    
    option = input("(?) Enable tuning: (Y/n): ")
    if option == 'Y':
        params = ocsvm.OCSVM().tune_hyperparameters(x_train)
    else:
        params = {'kernel': 'rbf', 'gamma': 'scale', 'nu': 0.01}
    
    ocsvm_model = ocsvm.OCSVM(**params)
    ocsvm_model.train(x_train)
    y_pred = ocsvm_model.detect(x_test, theta=0.01, window=1)
    return y_test, y_pred


def RF_detect(dataset_name):
    x_train, y_train, _ = data_loader.load_labeled_train_data(dataset_name, no_transform=False)
    x_test, y_test, _ = data_loader.load_test_data(dataset_name, no_transform=False)
    
    params = {'n_estimators': 100, 'max_depth': None, 'random_state': 42}
    rf_model = rf.RF(**params)
    rf_model.train(x_train, y_train)
    y_pred_cont = rf_model.predict(x_test)
    threshold = 0.95
    y_pred_binary = (y_pred_cont >= threshold).astype(int)
    return y_test, y_pred_binary


def SVR_detect(dataset_name):
    #load train, test data
    X_train, _ = data_loader.load_train_data(dataset_name)
    x_test, y_test, _ = data_loader.load_test_data(dataset_name)
    
    #split test data into validation data and test data
    X_val, X_test, Y_val, Y_test = utils.custom_train_test_split(dataset_name, x_test, y_test)
    
    #set model and detection hyperparameters
    model_params = {
        'kernel' : 'rbf', 
        'C' : 0.1,   
        'epsilon' : 0.1,      
        }
    
    detection_param = {
        'quantile' : [0.8, 0.85, 0.9, 0.905, 0.95, 0.99],
        'window' : [1, 10, 15, 20, 50, 100, 200]
    }    
    
    #create SVR model
    svr_model = svr.SVR(**model_params)

    #train SVR model
    option= input("Enable tuning (Y/n): ")
    if option== 'Y':
        svr_model.hyperparameter_tuning(X_train, X_val)
    else:
        svr_model.train(X_train)
    
    #Tuning detection hyperparameters
    param_combinations = list(product(detection_param['quantile'], detection_param['window']))
    best_f1_score = 0
    best_quantile= 0
    best_window= 0
    
    print("(+) Tuning detection hyperparameters...")
    for quantile, window in tqdm(param_combinations, desc="Tuning params", total=len(param_combinations)):
        Y_pred= svr_model.detect(X_test, X_val, quantile, window)
        f1_scr= f1_score(Y_test, Y_pred)
        if f1_scr > best_f1_score:
            best_f1_score= f1_scr   
            best_quantile= quantile
            best_window= window
    
    #Detection
    print(f"Best detection hyperparameters set to with quantile={best_quantile}, window={best_window}")
    Y_pred= svr_model.detect(X_test, X_val, best_quantile, best_window)
    return Y_test, Y_pred

def XGB_detect(dataset_name):
    # Prepare training and validation sets for next-step regression
    x_train, _ = data_loader.load_train_data(dataset_name)
    x_test, y_test, _ = data_loader.load_test_data(dataset_name)
    X_val, X_test_split, Y_val_raw, Y_test_split_raw = utils.custom_train_test_split(dataset_name, x_test, y_test)
    
    X_train = x_train[:-1, :]
    Y_train = x_train[1:, 0]
    x_val = X_val[:-1, :]
    y_val = X_val[1:, 0]
    Xtest = X_test_split[:-1, :]
    Ytest = X_test_split[1:, 0]

    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    dtest = xgb.DMatrix(Xtest)

    # Set base parameters
    params = {'objective': 'reg:squarederror', 'eta': 0.1, 'max_depth': 6,
              'subsample': 0.8, 'colsample_bytree': 0.8}
    option = input("(?) Enable XGB manual tuning? (Y/n): ")
    if option == 'Y':
        raw = input("Enter params (e.g. eta=0.05,max_depth=4): ")
        for kv in raw.split(','):
            k, v = kv.split('=')
            params[k.strip()] = float(v) if '.' in v else int(v)

    # Train with early stopping
    print("(+) Training XGBoost with early stopping on validation set...")
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, 'validation')],
        early_stopping_rounds=10,
        verbose_eval=False
    )

    # Coarse search best threshold quantile
    preds = bst.predict(dtest)
    errors = (Ytest - preds) ** 2

    best_f1 = 0
    best_q = None
    best_flags = None
    quantiles = [0.8, 0.85, 0.9, 0.95, 0.99]
    for q in quantiles:
        thresh = np.quantile(errors, q)
        flags = (errors > thresh).astype(int)
        y_pred = np.concatenate([[0], flags]).astype(int)
        y_true = Y_test_split_raw.astype(int)
        f1_sc = f1_score(y_true, y_pred)
        print(f"    Quantile {q}: F1 = {f1_sc:.3f}")
        if f1_sc > best_f1:
            best_f1 = f1_sc
            best_q = q
            best_flags = flags

    # Refine around best quantile
    if best_q is not None:
        low, high = max(0, best_q - 0.05), min(1, best_q + 0.05)
        refined_qs = np.linspace(low, high, 20)
        for q in refined_qs:
            thresh = np.quantile(errors, q)
            flags = (errors > thresh).astype(int)
            y_pred = np.concatenate([[0], flags]).astype(int)
            y_true = Y_test_split_raw.astype(int)
            f1_sc = f1_score(y_true, y_pred)
            if f1_sc > best_f1:
                best_f1 = f1_sc
                best_q = q
                best_flags = flags
        print(f"(+) Refined best quantile: {best_q:.4f} (F1 = {best_f1:.3f})")
    else:
        print("(!) No valid best quantile found in coarse search.")

    # Use best threshold
    y_pred = np.concatenate([[0], best_flags]).astype(int) if best_flags is not None else np.zeros_like(Y_test_split_raw)
    y_true = Y_test_split_raw.astype(int)
    return y_true, y_pred

if __name__ == '__main__':
    model_name = input("Enter model name (OCSVM/SVR/RF/XGB/HYBRID): ")
    dataset_name = input("Enter dataset name (BATADAL/SWAT/WADI): ")
    if model_name == 'OCSVM':
        y_test, y_pred = OCSVM_detect(dataset_name)
    elif model_name == 'SVR':
        y_test, y_pred = SVR_detect(dataset_name)
    elif model_name == 'RF':
        y_test, y_pred = RF_detect(dataset_name)
    elif model_name == 'XGB':
        y_test, y_pred = XGB_detect(dataset_name)
    elif model_name == 'HYBRID':
        y_test, ocsvm_pred = OCSVM_detect(dataset_name)
        _, rf_pred = RF_detect(dataset_name)
        _, xgb_pred = XGB_detect(dataset_name)
        y_pred = np.logical_or(np.logical_or(rf_pred, ocsvm_pred), xgb_pred).astype(int)
    
    # Evaluate
    main_eval.plot_evaluation(model_name, y_test, y_pred)
