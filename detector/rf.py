import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from itertools import product
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class RF(object):
    """
    Random Forest Regressor for ICS anomaly detection.
    Predicts the next value of the first sensor channel.
    """

    def __init__(self, **kwargs):
        print("(+) Initializing RF Regressor model...")
        
        # Default parameter values.
        params = {
        'n_estimators' : 100,    
        'random_state' : 42,      
        }
        
        #Adjust parameters
        for key,item in kwargs.items():
            params[key] = item
        self.params = params
        
    def create_model(self):
        """Instantiate the RF model with stored parameters."""
        print("(+) Creating RF model...")
        self.rf = RandomForestRegressor(
            n_estimators = self.params['n_estimators'],
            random_state = self.params['random_state']
        )

        return self.rf
       
    def train(self, X_train):
        """
        Train the model to predict the next timestep value of the first sensor.
        """
        print("(+) Training Random Forest Regressor...")
        X = X_train[:-1, :]
        y = X_train[1:, 0]  # predict next step of sensor 0
        self.create_model()
        self.rf.fit(X, y)

        return self.rf

    def detect(self, X_test, X_val=None, quantile=0.95, window=1):
        """
        Predict anomalies by comparing squared error to dynamic threshold.
        """
        if X_val is not None:
            Xv = X_val[:-1, :]
            yv = X_val[1:, 0]
            preds_val = self.rf.predict(Xv)
            val_errors = (preds_val - yv) ** 2
            self.threshold = np.quantile(val_errors, quantile)

        # Predict test data
        X = X_test[:-1, :]
        y_true = X_test[1:, 0]
        preds = self.rf.predict(X)
        errors = (preds - y_true) ** 2

        raw_flags = (errors > self.threshold).astype(int)
        raw_flags = np.concatenate([[0], raw_flags])  # Align with input length

        # Apply sliding window smoothing
        if window > 1:
            flags = np.zeros_like(raw_flags)
            for i in range(len(raw_flags)):
                start = max(0, i - window + 1)
                flags[i] = raw_flags[start:i + 1].max()
        else:
            flags = raw_flags

        return flags
    def hyperparameter_tuning(self, X_train, X_val, patience=3):
        """
        Tune n_estimators and max_depth with early stopping based on validation MSE.
        """
        print("(+) Tuning Random Forest hyperparameters...")
        X = X_train[:-1, :]
        y = X_train[1:, 0]
        Xv = X_val[:-1, :]
        yv = X_val[1:, 0]

        best_mse = float('inf')
        best_model = None
        best_params = {}
        no_improve_count = 0

        n_estimators_list = [50, 100, 150, 200]
        max_depth_list = [None, 5, 10, 20]

        for n, depth in tqdm(product(n_estimators_list, max_depth_list),
                             total=len(n_estimators_list) * len(max_depth_list),
                             desc="Hyperparameter tuning"):

            model = RandomForestRegressor(
                n_estimators=n,
                max_depth=depth,
                random_state= 42
            )
            model.fit(X, y)
            preds = model.predict(Xv)
            mse = mean_squared_error(yv, preds)

            if mse < best_mse:
                best_mse = mse
                best_model = model
                best_params = {'n_estimators': n, 'max_depth': depth}
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                print("(-) Early stopping: no improvement.")
                break

        print(f"(✓) Best RF model: {best_params}, Val MSE={best_mse:.5f}")
        self.rf = best_model
        
    def get_model(self):
        return self.rf
