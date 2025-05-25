import numpy as np
from keras import models, layers, optimizers, callbacks
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from itertools import product

class CNN(object):
    def __init__(self, input_length= None, n_layers=1, n_units=32, kernel_size=3):
        self.input_length = input_length
        self.n_layers = n_layers
        self.n_units = n_units
        self.kernel_size = kernel_size
        self.model = self.create_model()
        self.threshold = None

    def create_model(self):
        model = models.Sequential()
        model.add(layers.Conv1D(filters=self.n_units, kernel_size=self.kernel_size, activation='relu',
                         input_shape=(self.input_length, 1)))
        for _ in range(self.n_layers - 1):
            model.add(layers.Conv1D(filters=self.n_units, kernel_size=self.kernel_size, activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(1))  # Regression output
        model.compile(optimizer=optimizers.Adam(), loss='mse')
        return model

    def create_sequences(self, X, length):
        X_seq, y_seq = [], []
        for i in range(len(X) - length):
            X_seq.append(X[i:i + length, 0])  # Use only 1D input for Conv1D
            y_seq.append(X[i + length, 0])    # Predict sensor 0
        X_seq = np.array(X_seq)
        return X_seq[..., np.newaxis], np.array(y_seq)  # Add channel dimension
    
    def train(self, X_train, X_val=None, epochs=100, batch_size=32, patience=5):
        X_seq, y_seq = self.create_sequences(X_train, self.input_length)
        if X_val is not None:
            Xv_seq, yv_seq = self.create_sequences(X_val, self.input_length)
        else:
            Xv_seq, yv_seq = None, None

        _callbacks = []
        if Xv_seq is not None:
            _callbacks.append(callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True))

        self.model.fit(
            X_seq, y_seq,
            validation_data=(Xv_seq, yv_seq) if Xv_seq is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=_callbacks
        )

    def detect(self, X_test, X_val=None, quantile=0.95, window=1):
        X_seq, y_seq = self.create_sequences(X_test, self.input_length)
        preds = self.model.predict(X_seq).flatten()
        errors = (preds - y_seq) ** 2

        # Compute threshold from validation set
        if X_val is not None:
            Xv_seq, yv_seq = self.create_sequences(X_val, self.input_length)
            val_preds = self.model.predict(Xv_seq).flatten()
            val_errors = (val_preds - yv_seq) ** 2
            self.threshold = np.quantile(val_errors, quantile)

        flags = (errors > self.threshold).astype(int)
        flags = np.concatenate([np.zeros(self.input_length), flags])  # Padding for alignment

        if window > 1:
            smoothed = np.zeros_like(flags)
            for i in range(len(flags)):
                smoothed[i] = np.max(flags[max(0, i - window + 1):i + 1])
            flags = smoothed

        return flags

    def hyperparameter_tuning(self, X_train, X_val, patience=3):
        print("(+) Tuning CNN hyperparameters...")
        best_model = None
        best_mse = float('inf')
        best_params = {}

        layer_range = [1, 2, 3, 4, 5]
        unit_range = [4, 8, 16, 32, 64, 128, 256]
        history_lengths = [50, 100, 200]

        for n_layers, n_units, hist_len in tqdm(product(layer_range, unit_range, history_lengths),
                                                total=len(layer_range)*len(unit_range)*len(history_lengths),
                                                desc="CNN tuning"):

            model = CNN(input_length=hist_len,
                                         n_layers=n_layers,
                                         n_units=n_units)

            model.train(X_train, X_val, patience=patience)
            Xv_seq, yv_seq = model.create_sequences(X_val, hist_len)
            preds = model.model.predict(Xv_seq).flatten()
            mse = mean_squared_error(yv_seq, preds)

            print(f"-> layers={n_layers}, units={n_units}, hist={hist_len}, Val MSE={mse:.5f}")

            if mse < best_mse:
                best_mse = mse
                best_model = model
                best_params = {'n_layers': n_layers, 'n_units': n_units, 'input_length': hist_len}

        print(f"(✓) Best CNN config: {best_params}, Val MSE={best_mse:.5f}")
        self.__dict__.update(best_model.__dict__)