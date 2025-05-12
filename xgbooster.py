import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

class XGBoost(object):
    def __init__(self, **kwargs):
        """
        XGBoost-based anomaly detector for ICS data.

        Defaults:
         - n_estimators=100, learning_rate=0.1, max_depth=6,
         - subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror'
        Override any via kwargs.
        """
        print("(+) Initializing XGBRegressor anomaly detector...")
        self.params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'verbosity': 0
        }
        self.params.update(kwargs)
        self.model = None
        self.threshold = None

    def create_model(self):
        """Instantiate the XGBRegressor with current parameters."""
        print("(+) Creating XGBRegressor with params:", self.params)
        self.model = XGBRegressor(**self.params)
        return self.model

    def train(self, X_train, y_train, **kwargs):
        """
        Train the model on full training set (early stopping not supported by this wrapper).
        """
        if self.model is None:
            self.create_model()
        print("(+) Training on full training set (no early stopping).")
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X):
        """Return predicted continuous values."""
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        return self.model.predict(X)

    def fit_threshold(self, X_val, y_val, quantile=0.95):
        """
        Compute squared errors on validation set and set threshold at given quantile.
        """
        print(f"(+) Computing threshold at {quantile:.2f} quantile of squared errors...")
        y_pred = self.predict(X_val)
        errors = (y_val - y_pred) ** 2
        self.threshold = np.quantile(errors, quantile)
        print(f"(+) Threshold set to: {self.threshold:.4f}")
        return self.threshold

    def detect(self, X, y_true=None, plot=False):
        """
        Flag anomalies where squared error > threshold.
        Returns binary flags and errors.
        """
        if self.threshold is None:
            raise RuntimeError("Threshold not set. Call fit_threshold() first.")

        y_pred = self.predict(X)
        errors = (y_true - y_pred) ** 2 if y_true is not None else None

        if plot and errors is not None:
            plt.figure(figsize=(15, 3))
            plt.plot(errors, label='Squared Error')
            plt.axhline(self.threshold, linestyle='--', label='Threshold')
            plt.legend()
            plt.title('XGB Anomaly Detection Errors')
            plt.show()

        flags = (errors > self.threshold).astype(int) if errors is not None else None
        return flags, errors

    def tune_hyperparameters(self, X_train, y_train, param_dist=None, cv=3, n_iter=10, n_jobs=-1, random_state=42):
        """
        Tune hyperparameters via RandomizedSearchCV.
        Returns best_params.
        """
        print("(+) Tuning hyperparameters with RandomizedSearchCV...")
        if param_dist is None:
            param_dist = {
                'n_estimators': [50, 100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }

        base = XGBRegressor(objective='reg:squarederror', verbosity=0)
        rand_search = RandomizedSearchCV(
            estimator=base,
            param_distributions=param_dist,
            scoring='neg_mean_squared_error',
            n_iter=n_iter,
            cv=cv,
            verbose=1,
            n_jobs=n_jobs,
            random_state=random_state
        )
        rand_search.fit(X_train, y_train)
        best = rand_search.best_params_
        print("(+) Best hyperparameters:", best)
        self.params.update(best)
        self.create_model()
        return best

# Note: For true early stopping, switch to xgboost.train with DMatrix and early_stopping_rounds.
