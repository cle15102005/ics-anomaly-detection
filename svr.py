import numpy as np
from tqdm import tqdm
from itertools import product
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# SVR-based anomaly detector for ICS (BATADAL dataset)
from sklearn import svm
from sklearn.metrics import mean_squared_error

class SVR(object):
    """
    SVR-based Detector for ICS anomaly detection.
    Attributes:
      model: trained sklearn SVR model
      threshold: float, decision threshold on absolute error
    """
    def __init__(self, **kwargs):
        print("(+) Initializing SVR model...")
        
        # Default parameter values.
        params = {
        'kernel' : 'rbf', 
        'C' : 0.1,   
        'epsilon' : 0.1,      
        }
        
        #Adjust parameters
        for key,item in kwargs.items():
            params[key] = item
        self.params = params

    def create_model(self):
        """Instantiate the SVR model with stored parameters."""
        print("(+) Creating SVR model...")
        self.svr = svm.SVR(
            kernel=self.params['kernel'],
            C=self.params['C'],
            epsilon=self.params['epsilon']
        )
        
        return self.svr

    def train(self, X_train):
        """
        Train SVR on next-step regression of the first sensor/channel.
        Expects X_train shape (n_samples, n_features) containing only normal samples.
        """
        print("(+) Training SVR model...")
        # Prepare features and targets for one-step ahead prediction
        X = X_train[:-1, :]
        y = X_train[1:, 0]
        
        self.create_model()
        self.svr.fit(X, y)
        
        return self.svr
        
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
        preds = self.svr.predict(X)
        mse = (preds - y)**2
        # threshold at specified quantile of error distribution
        self.threshold = np.quantile(mse, quantile)
        
        # Predict on all but last to compute next-step squared error
        X = X_test[:-1, :]
        y_true = X_test[1:, 0]
        preds = self.svr.predict(X)
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
    
    def hyperparameter_tuning(self, X_train, X_val, patience = 3):
        X = X_train[:-1, :]
        y = X_train[1:, 0]
        Xv = X_val[:-1, :]
        yv = X_val[1:, 0]
        best_mse= 10
        best_svr= None
        no_improve_count= 0
        bestC= 0
        Cs= [0.1, 0.5, 1, 5, 10]
        epsilons= [0.001, 0.01, 0.1]
        
        print("(+) Tuning model hyperparameters...")
        param_combinations = list(product(Cs, epsilons))

        for C, epsilon in tqdm(param_combinations, desc="Hyperparameter tuning", total=len(param_combinations)):    
            model = svm.SVR(kernel='rbf', C=C, epsilon=epsilon)
            model.fit(X, y)
            preds = model.predict(Xv)
            mse = mean_squared_error(yv, preds)
            print(f"-> C={C}, epsilon={epsilon}, Val MSE={mse:.5f}")
            if mse < best_mse:
                best_mse = mse
                best_svr = model
                no_improve_count = 0
                bestC= C
                bestepsilon= epsilon
            else:
                no_improve_count += 1
            if no_improve_count >= patience:
                break
            
        print(f"Best model at C= {bestC}, epsilon= {bestepsilon}")
        self.svr = best_svr


