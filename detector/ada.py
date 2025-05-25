import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from itertools import product
from tqdm import tqdm

class AdaBoost(object):
    """
    AdaBoost-based Detector for ICS anomaly detection using one-step-ahead regression.
    """
    def __init__(self, **kwargs):
        print("(+) Initializing AdaBoost Regressor model...")
        
        # Default parameter values.
        params = {
        'n_estimators' : 100, 
        'learning_rate' : 0.1,   
        'random_state' : 42,      
        }
        
        #Adjust parameters
        for key,item in kwargs.items():
            params[key] = item
        self.params = params
        
    def create_model(self):
        """Instantiate the ADB model with stored parameters."""
        print("(+) Creating ADB model...")
        self.ada = AdaBoostRegressor(
            n_estimators = self.params['n_estimators'],
            learning_rate = self.params['learning_rate'],
            random_state = self.params['random_state']
        )

        return self.ada
            
    def train(self, X_train):
        """
        Train the AdaBoost model to predict the next timestep of the first sensor feature.
        """
        print("(+) Training AdaBoost Regressor...")
        X = X_train[:-1, :]
        y = X_train[1:, 0]
        self.create_model()
        self.ada.fit(X, y)
        
        return self.ada

    def detect(self, X_test, X_val=None, quantile=0.95, window=1):
        """
        Detect anomalies in the test set using prediction error thresholding and sliding window.
        """
        if X_val is not None:
            Xv = X_val[:-1, :]
            yv = X_val[1:, 0]
            preds = self.ada.predict(Xv)
            errors = (preds - yv) ** 2
            self.threshold = np.quantile(errors, quantile)

        # Test prediction
        X = X_test[:-1, :]
        y_true = X_test[1:, 0]
        preds = self.ada.predict(X)
        errors = (preds - y_true) ** 2

        raw_flags = (errors > self.threshold).astype(int)
        raw_flags = np.concatenate([[0], raw_flags])  # prepend 0 for alignment

        # Apply sliding window
        if window > 1:
            flags = np.zeros_like(raw_flags)
            for i in range(len(raw_flags)):
                start = max(0, i - window + 1)
                flags[i] = raw_flags[start:i+1].max()
        else:
            flags = raw_flags

        return flags

    def hyperparameter_tuning(self, X_train, X_val, patience=3):
        """
        Grid search for best (n_estimators, learning_rate) using validation MSE.
        """
        print("(+) Tuning AdaBoost hyperparameters...")
        X = X_train[:-1, :]
        y = X_train[1:, 0]
        Xv = X_val[:-1, :]
        yv = X_val[1:, 0]

        best_model = None
        best_mse = float('inf')
        best_params = {}
        no_improve_count = 0

        n_estimators_list = [50, 100, 150, 200]
        learning_rates = [0.01, 0.05, 0.1, 0.2]

        for n, lr in tqdm(product(n_estimators_list, learning_rates), total=len(n_estimators_list) * len(learning_rates), desc="Hyperparameter tuning"):
            model = AdaBoostRegressor(n_estimators=n, learning_rate=lr, random_state=42)
            model.fit(X, y)
            preds = model.predict(Xv)
            mse = mean_squared_error(yv, preds)
            if mse < best_mse:
                best_mse = mse
                best_model = model
                best_params = {'n_estimators': n, 'learning_rate': lr}
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                break

        print(f"Best AdaBoost model at n_estimators={best_params['n_estimators']}, learning_rate={best_params['learning_rate']}")
        self.ada = best_model

    def get_model(self):
        return self.ada