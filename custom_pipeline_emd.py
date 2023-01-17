import numpy as np
import emd

class EMDIMFHHT:
  """
  Class for computing HHT decomposition of IMFs using iterated sifting of EMD.
  """
  def __init__(self, max_imfs=5, sampling_rate_hz=11025, max_freq=10000, num_bins=100):
    self.max_imfs = max_imfs
    self.sampling_rate_hz = sampling_rate_hz
    self.max_freq = max_freq
    self.num_bins = num_bins

  def fit(self, x, y=None, **fit_params):
    return self

  def transform(self, x):
    imf = emd.sift.iterated_mask_sift(
           x, sample_rate=self.sampling_rate_hz, max_imfs=self.max_imfs)
    IP, IF, IA = emd.spectra.frequency_transform(imf, self.sampling_rate_hz, 'nht')
    f, hht = emd.spectra.hilberthuang(IF, IA, sum_imfs=False, sample_rate=self.sampling_rate_hz)

    return hht
    