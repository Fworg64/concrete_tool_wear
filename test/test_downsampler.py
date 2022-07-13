import unittest
import numpy as np

from custom_pipeline_elements import SignalDecimate


class SignalDecimateTest(unittest.TestCase):

    def test_samples(self):
        num_samples = 4
        num_dims = 200
        x = np.random.rand(num_samples,num_dims) # 6 samples of 2 dimensions
        #print("x",x)
        signal = SignalDecimate(2)
        signal.fit(x)
        z = signal.transform(x)
        #print("z",z)
        self.assertEqual(x.shape[0],z.shape[0])
        self.assertEqual(x.shape[1],z.shape[1]*2,"x equal to z")

        



if __name__ == '__main__':
    unittest.main()