import numpy as np
from keras import models, layers, optimizers, callbacks
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

class AE(object):
    """
    LSTM-based Autoencoder Detector for ICS anomaly detection.
    Attributes:
    params
    """
    def __init__(self, **kwargs):
        print("(+) Initializing LSTM-AE model...")

        # Default parameters
        params = {
            'nI': None,         # Number of input features (required)
            'nH': 2,            # Number of LSTM layers (not units)
            'cf': 3.5,          # Compression factor
            'epochs': 100,
            'batch_size': 512,
            'history_length' : 50
        }

        for key, val in kwargs.items():
            params[key] = val
        self.params = params


    def create_model(self):
        """
        Build LSTM Autoencoder with stacked LSTM layers.
        """
        print(f"(+) Creating LSTM Autoencoder with {self.params['nH']} hidden layers...")

        input_dim = self.params['nI']
        latent_dim = max(1, int(input_dim / self.params['cf']))
        units_per_layer = latent_dim * 2

        inputs = layers.Input(shape=(self.params['history_length'], input_dim))

        # Encoder: stacked LSTMs
        x = inputs
        for i in range(self.params['nH'] - 1):
            x = layers.LSTM(units_per_layer, return_sequences=True)(x)
        x = layers.LSTM(units_per_layer, return_sequences=False)(x)

        bottleneck = layers.Dense(latent_dim, activation='relu')(x)

        # Decoder: stacked LSTMs
        x = layers.RepeatVector(self.params['history_length'])(bottleneck)
        for i in range(self.params['nH'] - 1):
            x = layers.LSTM(units_per_layer, return_sequences=True)(x)
        x = layers.LSTM(units_per_layer, return_sequences=True)(x)

        outputs = layers.TimeDistributed(layers.Dense(input_dim))(x)

        self.autoencoder = models.Model(inputs, outputs)
        self.autoencoder.compile(optimizer=optimizers.Adam(), loss='mse')

        return self.autoencoder

    def prepare_data(self, X):
        print(f"(+) Preparing data with history_length={self.params['history_length']}...")
        sequences = []
        for i in range(len(X) - self.params['history_length'] + 1):
            window = X[i:i + self.params['history_length']]
            sequences.append(window)
        return np.array(sequences)

    def train(self, X_train, X_val=None, patience=3):
        print("(+) Training LSTM Autoencoder...")
        X_seq = self.prepare_data(X_train)
        self.create_model()

        _callbacks = []
        if X_val is not None:
            X_val_seq = self.prepare_data(X_val)
            early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            _callbacks.append(early_stop)
            validation_data = (X_val_seq, X_val_seq)
        else:
            validation_data = None

        self.autoencoder.fit(
            X_seq, X_seq,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            validation_data=validation_data,
            callbacks=_callbacks,
            verbose=1
        )

    def detect(self, X_test, X_val, quantile=0.95, window=1):
        print("(+) Detecting anomalies using LSTM Autoencoder...")
        Xv_seq = self.prepare_data(X_val)
        Xt_seq = self.prepare_data(X_test)

        val_preds = self.autoencoder.predict(Xv_seq)
        val_errors = np.mean((Xv_seq - val_preds) ** 2, axis=(1, 2))
        self.threshold = np.quantile(val_errors, quantile)

        test_preds = self.autoencoder.predict(Xt_seq)
        test_errors = np.mean((Xt_seq - test_preds) ** 2, axis=(1, 2))
        raw_flags = (test_errors > self.threshold).astype(int)

        aligned_flags = np.concatenate([np.zeros(self.params['history_length'] - 1, dtype=int), raw_flags])

        if window > 1:
            flags = np.zeros_like(aligned_flags)
            for i in range(len(aligned_flags)):
                start = max(0, i - window + 1)
                flags[i] = aligned_flags[start:i+1].max()
        else:
            flags = aligned_flags

        return flags

    def hyperparameter_tuning(self, X_train, X_val, patience=3):
        print("(+) Starting hyperparameter tuning for nH and cf...")

        if self.params['nI'] is None:
            self.params['nI'] = X_train.shape[1]

        X_train_seq = self.prepare_data(X_train)
        X_val_seq = self.prepare_data(X_val)

        grid = {
            'nH': [1, 2, 3, 4],
            'cf': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        }

        best_mse = float('inf')
        best_params = {}

        for combo in tqdm(list(ParameterGrid(grid)), desc="Tuning"):
            self.params['nH'] = combo['nH']
            self.params['cf'] = combo['cf']
            self.create_model()

            early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

            self.autoencoder.fit(
                X_train_seq, X_train_seq,
                epochs=self.params['epochs'],
                batch_size=self.params['batch_size'],
                validation_data=(X_val_seq, X_val_seq),
                callbacks=[early_stop],
                verbose=0
            )

            preds = self.autoencoder.predict(X_val_seq)
            mse = np.mean((X_val_seq - preds) ** 2)

            if mse < best_mse:
                best_mse = mse
                best_params = combo

        print(f"(+) Best parameters: {best_params}, Validation MSE: {best_mse:.6f}")
        self.params['nH'] = best_params['nH']
        self.params['cf'] = best_params['cf']
        self.train(X_train, X_val)

    def get_model(self):
        return self.autoencoder
