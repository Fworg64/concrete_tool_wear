import unittest

from custom_pipeline_emd import EMDIMFHHT

class EMDTest(unittest.TestCase):

  def test_emd(self):
    num_samples = 50
    num_dims = 100
    x = np.random.rand(num_samples,num_dims) # 50 samples of 100 dimensions

    my_hht = EMDIMFHHT()
    hht = my_hht.transform(x)
    self.assertEqual(hht.shape[0], x.shape[0])

