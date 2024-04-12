#function for the different scalers- chanel and sample
import numpy as np
import pywt
#from stingray import lightcurve
#from stingray.bispectrum import Bispectrum

from scipy import signal

import pdb

class SampleScaler:
    """
    class for scaling individual sample means and deviations
    """
    def __init__(self):
       pass

    def fit(self, x, y=None, **fit_params):
       return self

    def transform(self, x):
       u = np.expand_dims(np.mean(x,axis = 1), axis = -1)
       s = np.expand_dims(np.std(x,axis = 1), axis = -1)
       z = (x - u) / s
       return z

class ChannelScaler:
    """
    Class for scaling individual channels within samples to zero mean and unit std. dev
    Reduces to sample scaler for num_channels=1
    """
    
    def __init__(self, num_channels=1):
      self.num_channels = num_channels

    def fit(self, x, y=None, **fit_params):
      dims = np.shape(x)
      if dims[1] % self.num_channels != 0:
        raise IndexError("Number of features must be divisable by number of channels!")
      return self
      
    def transform(self, x):
      z = np.array(x)
      z = z.reshape((z.shape[0],self.num_channels,-1), order='F')
      u = np.expand_dims(np.mean(z,axis = -1), axis = -1)
      s = np.expand_dims(np.std(z,axis = -1), axis = -1)
      z = (z - u) / s
      z = z.reshape((z.shape[0], -1), order='F')
      return z

class FFTMag:
    """
    Class for computing magnitude of right side of FFT of input signal,
    optional num_channels parameter causes computation to be broken down along chunks of signal
    optional power parameter transforms data with SQRT or SQUARE, SQUARE is approx PSD
    Input must be 2D np.array with second dimension multiple of number of channels
    First dim is samples, second dim is features of samples
    Transform returns the right side of the FFT magnitude, reducing features roughly by factor of 2
    """

    def __init__(self, num_channels=1, power=None, after=None):
      self.num_channels = num_channels
      self.power = power
      self.after = after
      self.recognized_powers = {  "SQRT": lambda x : np.sqrt(x), 
                                "SQUARE": lambda x : np.multiply(x,x),
                                 "OUTER": lambda x : np.multiply(np.expand_dims(x, axis=1), np.expand_dims(x,axis=2)).reshape(x.shape[0],-1),
                                  "DIFF": lambda x : np.diff(x, n=1),
                                 "DIFF2": lambda x : np.diff(x, n=2),
                                 "BOOST": lambda x : np.subtract(x, (1.0) * np.diff(x, n=2, prepend=0, append=0)),
                                 "BOOST2": lambda x : np.subtract(x, (2.0) * np.diff(x, n=2, prepend=0, append=0)),
                                 "BOOST10": lambda x : np.subtract(x, (10.0) * np.diff(x, n=2, prepend=0, append=0)),
                                 "BOOSTm10": lambda x : np.add(x, (10.0) * np.diff(x, n=2, prepend=0, append=0)),
                                 "BOOST20": lambda x : np.subtract(x, (20.0) * np.diff(x, n=2, prepend=0, append=0)),
                                 "BOOST50": lambda x : np.subtract(x, (50.0) * np.diff(x, n=2, prepend=0, append=0)),
                                   "SUM": lambda x : np.cumsum(x, axis=-1),
                                  "FILT75": lambda x : signal.sosfiltfilt(signal.butter(8, 0.75, "lp", output='sos'), x),
                                  "FILT50": lambda x : signal.sosfiltfilt(signal.butter(8, 0.50, "lp", output='sos'), x),
                                  "FILT25": lambda x : signal.sosfiltfilt(signal.butter(8, 0.25, "lp", output='sos'), x),
                                  "FILT15": lambda x : signal.sosfiltfilt(signal.butter(8, 0.15, "lp", output='sos'), x),
                                    None: lambda x : x}
      if power not in self.recognized_powers:
        raise ValueError("power param must be in %s" % (str(self.recognized_powers)))
      if after not in self.recognized_powers:
        raise ValueError("after param must be in %s" % (str(self.recognized_powers)))

    def fit(self, x, y=None, **fit_params):
      dims = np.shape(x)
      if dims[1] % self.num_channels != 0:
        raise IndexError("Number of features must be divisable by number of channels!")
      return self


    def transform(self, x):
      z = np.array(x)
      if self.num_channels != 1:
        z = z.reshape((z.shape[0],self.num_channels,-1), order='F')
        z = np.abs(np.fft.rfft(z, axis=1)).reshape((z.shape[0], -1), order='F')
      else:
        z = np.abs(np.fft.rfft(z))
      z = self.recognized_powers[self.power](z)
      z = self.recognized_powers[self.after](z)
      return z

class FFTFull:
  """
  Class for computing full FFT for real valued signals, returns either magnitude and phase
  or real and imaginary coefficients depending on if MagPhase=True
  """
  def __init__(self, MagPhase=True):
    self.MagPhase = MagPhase

  def fit(self, x, y=None, **fit_params):
    return self

  def transform(self, x):
    z = np.array(x)
    z = np.fft.rfft(z)
    if self.MagPhase:
      z1 = np.abs(z)
      z2 = np.angle(z)
    else:
      z1 = np.real(z)
      z2 = np.imag(z)
    z = np.hstack((z1,z2))
    return z

class WaveletDecomposition:
  """
  Class for computing wavelet coefficients using the specified wavelet basis
  """
  def __init__(self, num_channels=1, basis='db1', num_levels=1, 
                     decomp_ratio=None, sample_size=None):
    self.num_channels = num_channels
    self.basis = basis
    if decomp_ratio:
      max_levels = pywt.dwt_max_level(sample_size, self.basis)
      self.num_levels = int(decomp_ratio*max_levels)
    else:
      self.num_levels = num_levels

  def fit(self, x, y=None, **fit_params):
    return self
 
  def transform(self, x):
    z = np.array(x)
    coeff = pywt.wavedec(z, self.basis, level=self.num_levels)
    out = np.column_stack(coeff)
    return out  


class BispectrumEstimation:
  """
  Class for computing bispectrum magnitude transformation
  """
  def __init__(self, sample_rate=44100):
    self.sample_rate = sample_rate
    pass

  def fit(self, x, y=None, **fit_params):
    return self

  def transform(self, x):
    # outer product of FFT of each row
    
    # offset, rolled up FFT of each row

    # return product
    pass

class SignalDecimate:
  """
  Class for downsampling factors higher than 13
  """
  def __init__(self,downsample_factor):
     self.downsample_factor = downsample_factor

  def fit(self, x):
     return self

  def transform(self,x):
     z = signal.decimate(x,self.downsample_factor)
     return z
