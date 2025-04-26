# numpy stack
import numpy as np

# Ignore ugly futurewarnings from np vs tf.
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

# sklearn for ML models
from sklearn import svm

# classes
class OCSVM(object):
    """ sklearn-based Detector for ICS event detection.

        Attributes:
        params: dictionary with parameters defining the model structure,
    """
    def __init__(self, **kwargs):
        """ Class constructor, stores parameters and initialize OCSVM sklearn model. """
        print("(+) Initializing OCSVM model...")
        
        # Default parameter values.
        params = {
        'kernel' : 'rbf',   #determines the decision boundary shape.
        'gamma' : 'auto',   #determines the influence of a single training point reaches.
        'nu' : 0.001,       #determines the sensitivity to anomalies
        }
        
        #Adjust parameters
        for key,item in kwargs.items():
            params[key] = item
        self.params = params

    def create_model(self):
        """ Creates OCSVM sklearn model.
        """
        print("(+) Creating OCSVM model...")
        
        # create model
        ocsvm = svm.OneClassSVM(kernel= self.params['kernel'], gamma= self.params['gamma'], nu= self.params['nu'])

        self.ocsvm = ocsvm
        return ocsvm

    def train(self, x_train):
        """Train OCSVM model
        """
        self.create_model()
        
        print("(+) Training OCSVM model...")    
        self.ocsvm.fit(x_train)
        return NotImplementedError

    def predict(self, x_test):
        """ Predict the prediction y_pred
        """
        y_pred = self.ocsvm.predict(x_test)
        return np.where(y_pred == -1, 1, 0)
  
    def get_ocsvm(self):
        """ Return the ocsvm model
        """
        return self.ocsvm

if __name__ == "__main__":
    print("Not a main file.")
