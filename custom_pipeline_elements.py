#function for the different scalers- chanel and sample
import numpy as np
import pdb

class SampleScaler:
    """
    class for scaleing individual sample means and deviations
    """
    def __init__(self):
        self.u= None
        self.s= None

    def fit(self,x):
        self.u = np.expand_dims(np.mean(x,axis = 1), axis = -1)
        self.s = np.expand_dims(np.std(x,axis = 1), axis = -1)

    def transform(self,x):
       z = (x - self.u) / self.s
       return z
