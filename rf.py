# numpy stack
import numpy as np

# Ignore ugly futurewarnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# sklearn for ML models
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel(object):
    """ sklearn-based Random Forest model for ICS event classification.
    
        Attributes:
        params: dictionary with model hyperparameters.
    """
    def __init__(self, **kwargs):
        """ Class constructor: sets default parameters and allows overrides. """
        print("(+) Initializing Random Forest model...")

        # Default hyperparameters
        params = {
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42,
            'verbose': 0
        }

        # Override with provided kwargs
        for key, item in kwargs.items():
            params[key] = item
        self.params = params

    def create_model(self):
        """ Creates a sklearn RandomForestClassifier model. """
        print("(+) Creating Random Forest model...")

        rf = RandomForestClassifier(
            n_estimators=self.params['n_estimators'],
            max_depth=self.params['max_depth'],
            random_state=self.params['random_state']
        )

        if self.params['verbose'] > 0:
            print("Random Forest model created with parameters:", self.params)

        self.rf = rf
        return rf

    def train(self, x_train, y_train):
        """ Trains the Random Forest model using x and y. """
        self.create_model()
        print("(+) Training Random Forest model...")
        self.rf.fit(x_train, y_train)

    def predict(self, x_test):
        """ Predict labels for test data. """
        return self.rf.predict(x_test)

    def get_model(self):
        """ Returns the trained Random Forest model object. """
        return self.rf

if __name__ == "__main__":
    print("Not a main file.")