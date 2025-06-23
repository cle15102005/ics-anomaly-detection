import numpy as np
from keras import models, layers, optimizers, callbacks
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

class CNN(object):
    def __init__(self, **kwargs):
        print("(+) Initializing CNN Predictor...")

        params = {
            'nI': None,           # Number of input features
            'n_layers': 2,        # Number of Conv1D layers
            'units': 32,          # Number of filters per Conv1D layer
            'kernel_size': 3,     # Size of the kernel
            'epochs': 100,
            'batch_size': 512,
            'history_length': 50,
        }

        for key, val in kwargs.items():
            params[key] = val
        self.params = params
        self.cnn = None
        self.threshold = None

    def create_model(self):
        print("(+) Creating CNN Predictor model...")
        self.cnn = models.Sequential()
        input_shape = (self.params['history_length'], self.params['nI'])

        # First layer with input shape
        self.cnn.add(layers.Conv1D(filters=self.params['units'], kernel_size=self.params['kernel_size'],
                                   activation='relu', input_shape=input_shape))

        # Additional hidden layers
        for _ in range(self.params['n_layers'] - 1):
            self.cnn.add(layers.Conv1D(filters=self.params['units'], kernel_size=self.params['kernel_size'],
                                       activation='relu'))

        self.cnn.add(layers.Flatten())
        self.cnn.add(layers.Dense(1))  # Regression output
        self.cnn.compile(optimizer=optimizers.Adam(), loss='mse')

    def prepare_data(self, X):
        print(f"(+) Preparing data with history_length={self.params['history_length']}...")
        seq_X, seq_y = [], []
        for i in range(len(X) - self.params['history_length']):
            seq_X.append(X[i:i + self.params['history_length']])
            seq_y.append(X[i + self.params['history_length'], 0])
        return np.array(seq_X), np.array(seq_y)

    def train(self, X_train, X_val=None, patience=3):
        X_seq, y_seq = self.prepare_data(X_train)
        self.create_model()

        _callbacks = []
        if X_val is not None:
            Xv_seq, yv_seq = self.prepare_data(X_val)
            early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            _callbacks.append(early_stop)
            validation_data = (Xv_seq, yv_seq)
        else:
            validation_data = None

        self.cnn.fit(
            X_seq[..., np.newaxis], y_seq,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            validation_data=(Xv_seq[..., np.newaxis], yv_seq) if X_val is not None else None,
            callbacks=_callbacks,
            verbose=1
        )

    def detect(self, X_test, X_val, quantile=0.95, window=1):
        print("(+) Detecting anomalies with CNN Predictor...")
        Xv_seq, yv_seq = self.prepare_data(X_val)
        preds_val = self.cnn.predict(Xv_seq[..., np.newaxis])
        val_errors = (preds_val.flatten() - yv_seq) ** 2
        self.threshold = np.quantile(val_errors, quantile)

        Xt_seq, yt_seq = self.prepare_data(X_test)
        preds_test = self.cnn.predict(Xt_seq[..., np.newaxis])
        test_errors = (preds_test.flatten() - yt_seq) ** 2
        raw_flags = (test_errors > self.threshold).astype(int)

        aligned_flags = np.concatenate([np.zeros(self.params['history_length'], dtype=int), raw_flags])

        if window > 1:
            flags = np.zeros_like(aligned_flags)
            for i in range(len(aligned_flags)):
                start = max(0, i - window + 1)
                flags[i] = aligned_flags[start:i+1].max()
        else:
            flags = aligned_flags

        return flags

    def hyperparameter_tuning(self, X_train, X_val, patience=3):
        print("(+) Starting CNN Predictor hyperparameter tuning...")
        if self.params['nI'] is None:
            self.params['nI'] = X_train.shape[1]

        param_grid = {
            'n_layers': [1, 2, 3, 4],
            'units': [4, 8, 16, 32, 64, 128],
            'history_length': [50, 100]
        }

        best_mse = float('inf')
        best_config = {}

        for combo in tqdm(list(ParameterGrid(param_grid)), desc="CNN Tuning"):
            self.params['n_layers'] = combo['n_layers']
            self.params['units'] = combo['units']
            self.params['history_length'] = combo['history_length']

            X_seq, y_seq = self.prepare_data(X_train)
            Xv_seq, yv_seq = self.prepare_data(X_val)

            self.create_model()
            early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

            self.cnn.fit(
                X_seq[..., np.newaxis], y_seq,
                epochs=self.params['epochs'],
                batch_size=self.params['batch_size'],
                validation_data=(Xv_seq[..., np.newaxis], yv_seq),
                callbacks=[early_stop],
                verbose=0
            )

            preds = self.cnn.predict(Xv_seq[..., np.newaxis])
            mse = mean_squared_error(yv_seq, preds)

            if mse < best_mse:
                best_mse = mse
                best_config = combo

        print(f"Best params: {best_config}, Validation MSE: {best_mse:.6f}")
        self.params.update(best_config)
        self.train(X_train, X_val)

    def get_model(self):
        return self.cnn
