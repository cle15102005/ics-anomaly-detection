import numpy as np
from keras import models, layers, optimizers, callbacks

class BackTimeAE:
    def __init__(self, **kwargs):
        """
        AE class tailored for BACKTIME.
        nI: number of input features
        history_length: length of time window
        nH: number of LSTM layers
        cf: compression factor (controls bottleneck size)
        """
        params = {
            'nI': None,
            'nH': 2,
            'cf': 3.5,
            'epochs': 10,
            'batch_size': 512,
            'history_length': 10
        }
        
        for key, val in kwargs.items():
            params[key] = val
        self.params = params

    def create_model(self):
        input_dim = self.params['nI']
        latent_dim = max(1, int(input_dim / self.params['cf']))
        units = latent_dim * 2
        hlen = self.params['history_length']

        inputs = layers.Input(shape=(hlen, input_dim))
        
        # Encoder
        x = inputs
        for _ in range(self.params['nH'] - 1):
            x = layers.LSTM(units, return_sequences=True)(x)
        x = layers.LSTM(units, return_sequences=False)(x)
        bottleneck = layers.Dense(latent_dim, activation='relu')(x)

        # Decoder
        x = layers.RepeatVector(hlen)(bottleneck)
        for _ in range(self.params['nH'] - 1):
            x = layers.LSTM(units, return_sequences=True)(x)
        x = layers.LSTM(units, return_sequences=True)(x)

        outputs = layers.TimeDistributed(layers.Dense(input_dim))(x)
        self.model = models.Model(inputs, outputs)
        self.model.compile(optimizer=optimizers.Adam(), loss='mse')

    def prepare_data(self, X):
        hlen = self.params['history_length']
        return np.array([X[i:i + hlen] for i in range(len(X) - hlen + 1)])

    def train(self, X_train, X_val=None, patience=3):
        print("[BackTimeAE] Training model...")
        X_seq = self.prepare_data(X_train)
        self.create_model()

        val_seq = self.prepare_data(X_val) if X_val is not None else None
        callbacks_list = []
        if val_seq is not None:
            es = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            callbacks_list.append(es)

        self.model.fit(
            X_seq, X_seq,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            validation_data=(val_seq, val_seq) if val_seq is not None else None,
            callbacks=callbacks_list,
            verbose=1
        )

    def get_reconstruction_errors(self, X):
        """
        Returns MAE per time window.
        """
        X_seq = self.prepare_data(X)
        preds = self.model.predict(X_seq)
        errors = np.mean(np.abs(X_seq - preds), axis=(1, 2))
        return errors

    def select_top_alpha_timestamps(self, X, alpha_T):
        """
        Selects top alpha_T timestamps with highest MAE.
        Returns indices aligned to original time (with offset).
        """
        errors = self.get_reconstruction_errors(X)
        top_indices = np.argsort(errors)[-alpha_T:][::-1]
        aligned_indices = top_indices + self.params['history_length'] - 1
        return aligned_indices, errors[top_indices]

    def get_model(self):
        return self.model
