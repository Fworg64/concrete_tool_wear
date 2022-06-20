
import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler

from custom_pipeline_elements import ChannelScaler, SampleScaler, FFTMag

class ScalerTest(unittest.TestCase):

    def test_standard(self):
        x= [[2.2],[4.4],[0.5],[-10.0],[3.0],[8.0],[7.0],[9.0],[1.0],[23.0],[56.0],[0.0],[48.0]]
        scaler = StandardScaler()
        scaler.fit(x)
        z = scaler.transform(x)
        self.assertAlmostEqual(np.mean(z), 0.0, 4, "transformed mean not zero for standard")
        self.assertAlmostEqual(np.std(z), 1.0, 4, "transformed std deveation for standard")

    def test_sample(self):
        num_samples = 50
        num_dims = 100
        x = np.random.rand(num_samples,num_dims) # 50 samples of 99 dimensions
        # print(x)
        scaler = SampleScaler() 
        scaler.fit(x)
        z = scaler.transform(x)
        #print (z)
        for y in z:
            self.assertAlmostEqual(np.mean(y), 0.0, 4, "transformed mean not zero for sample")
            self.assertAlmostEqual(np.std(y), 1.0, 4, "transformed std deveation for sample")

    def test_channel(self): #original number of channels of four
        num_samples = 50
        num_dims = 100
        x = np.random.rand(num_samples,num_dims) # 50 samples of 99 dimensions
        #print("oringinall data X", x)
        scaler = ChannelScaler(4) 
        scaler.fit(x)
        z = scaler.transform(x)
        #print(z)
        for y in z:
            self.assertAlmostEqual(np.mean(y), 0.0, 4, "transformed mean not zero for channels")
            self.assertAlmostEqual(np.std(y), 1.0, 4, "transformed std deveation for channels")
        # reshape z by channel then checking means and std deviations 

    def test_1and3channels(self):
        # testing 1, 2, and 3 channels
        i = 1
        channel = 1
        while i<=3:
            #print (i)
            num_samples = 50
            num_dims = 96
            x = np.random.rand(num_samples,num_dims) # 50 samples of 99 dimensions        
            #print("oringinall data X", x)
            scaler = ChannelScaler(channel) 
            #print("c",channel)
            scaler.fit(x)
            z = scaler.transform(x)
            #print(z)
            for y in z:
                self.assertAlmostEqual(np.mean(y), 0.0, 4, "transformed mean not zero for one channel")
                self.assertAlmostEqual(np.std(y), 1.0, 4, "transformed std deveation isn't 1 for one channel")
            # reshape z by channel then checking means and std deviations 
            channel = channel + 1
            i = i + 1

    def test_featureCheckChannel(self): #original number of channels of four
        
        # goal is to see if the index error occurs if the features aren't 
        # divisible by number of channels
        # 23 isn't divisible by 4
        # index error was raised
        num_samples = 50
        num_dims = 99
        x = np.random.rand(num_samples,num_dims) # 50 samples of 99 dimensions
        #print("oringinall data X", x)
        scaler = ChannelScaler(4) 
        self.assertRaises(IndexError, scaler.fit, x)
        #scaler.fit(x)
        

class FFTMagTest(unittest.TestCase):

    def test_FFT(self):
        num_samples = 50
        num_dims = 100
        x = np.random.rand(num_samples,num_dims)
        tf = FFTMag(4)
        tf.fit(x)
        z = tf.transform(x)
        self.assertEqual(z.shape[0], num_samples)
        self.assertGreaterEqual(z.shape[1], num_dims/2.)
        self.assertLessEqual(z.shape[1],num_dims)

    def test_featureCheckFFTmag(self):
        # goal is to see if the index error occurs if the features aren't 
        # divisible by number of FFTmags
        # current number of FFTmags is 4
        # so num_dims cannot be divisible by 4 in order to raise error
        # num_dims = 99
        # index error was raised 
        num_samples = 50
        num_dims = 99
        x = np.random.rand(num_samples,num_dims)
        tf = FFTMag(4)
        self.assertRaises(IndexError, tf.fit, x)
        #tf.fit(x)

if __name__ == '__main__':
    unittest.main()
