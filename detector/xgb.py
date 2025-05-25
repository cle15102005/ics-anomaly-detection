import numpy as np
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# XGB-based anomaly detector for ICS
import xgboost
from sklearn.metrics import mean_squared_error

class XGBoost(object):
    """
    XGBoost-based Detector for ICS anomaly detection.
    Attributes:
    params
    """
    def __init__(self, **kwargs):
        print("(+) Initializing XGB model...")
        
        # Default parameter values.
        params = {
            'objective' : 'reg:squarederror', 
            'n_estimators' : 100,
            'learning_rate' : 0.1,
            'max_depth' : 4,
            'subsample' : 0.8,
            'colsample_bytree' : 0.8,
            'random_state' : 42   
        }
        
        #Adjust parameters
        for key,item in kwargs.items():
            params[key] = item
        self.params = params

    def create_model(self):
        """Instantiate the XGB model with stored parameters."""
        print("(+) Creating XGB model...")
        self.xgb = xgboost.XGBRegressor(
            objective = self.params['objective'],
            n_estimators = self.params['n_estimators'],
            learning_rate = self.params['learning_rate'],
            max_depth = self.params['max_depth'],
            subsample = self.params['subsample'],
            colsample_bytree = self.params['colsample_bytree'],
            random_state = self.params['random_state']
        )

        return self.xgb

    def train(self, X_train):
        """
        Train XGB on next-step regression of the first sensor/channel.
        Expects X_train shape (n_samples, n_features) containing only normal samples.
        """
        print("(+) Training XGB model...")
        # Prepare features and targets for one-step ahead prediction
        X = X_train[:-1, :]
        y = X_train[1:, 0]
        
        self.create_model()
        self.xgb.fit(X, y)
        
        return self.xgb
        
    def detect(self, X_test, X_val, quantile= 0.95, window= 1):
        """
        Detect anomalies on test set using sliding window. Returns binary flags aligned with original samples.

        X_test: array shape (n_samples, n_features)
        window: optional int, sliding window length (overrides self.window)
        returns: flags array length n_samples (0 normal, 1 anomaly)
        """
        #Set threshold
        X = X_val[:-1, :]
        y = X_val[1:, 0]
        preds = self.xgb.predict(X)
        mse = (preds - y)**2
        # threshold at specified quantile of error distribution
        self.threshold = np.quantile(mse, quantile)
        
        # Predict on all but last to compute next-step squared error
        X = X_test[:-1, :]
        y_true = X_test[1:, 0]
        preds = self.xgb.predict(X)
        mse = (preds - y_true) ** 2
        # Raw flags
        raw_flags = (mse > self.threshold).astype(int)
        # Prepend a 0 for the first sample
        raw_flags = np.concatenate([[0], raw_flags])

        # Apply sliding window: for each idx, flag if any in last 'window' raw flags
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
        Hyperparameter tuning using early stopping on validation set.
        """
        X = X_train[:-1, :]
        y = X_train[1:, 0]
        Xv = X_val[:-1, :]
        yv = X_val[1:, 0]

        # Grid of hyperparameters to test
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'n_estimators': [50, 100, 200],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
        
        param_combinations = list(ParameterGrid(param_grid))

        best_mse = float('inf')
        best_model = None
        best_params = None

        print("(+) Tuning model hyperparameters with early stopping...")
        for params in tqdm(param_combinations, desc="Hyperparameter tuning"):
            model = xgboost.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                early_stopping_rounds=patience,
                **params
            )
            model.fit(
                X, y,
                eval_set=[(Xv, yv)],
                verbose=False
            )
            
            preds = model.predict(Xv)
            mse = mean_squared_error(yv, preds)
            
            if mse < best_mse:
                best_mse = mse
                best_model = model
                best_params = params

        print(f"(+) Best model: {best_params}, MSE: {best_mse:.4f}")
        self.xgb = best_model
    
    def get_model(self):
        return self.xgb