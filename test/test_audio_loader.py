import unittest
import numpy as np

from windowizer import Windowizer, window_maker

import pdb

from loader import load_audio_files

class AudioLoaderTest(unittest.TestCase):

  def test_load_dims(self):
    # Load and Downsample
    audio_fs = 44100
    downsample_factor = 1
    raw_audio_data, metadata = load_audio_files(
      "./raw_audio/classifications.txt", integer_downsample=downsample_factor)
    audio_fs = int(audio_fs/downsample_factor)

    # Apply windowing
    window_duration = 0.2 # seconds
    window_overlap  = 0.5 # ratio of overlap [0,1)
    window_shape    = "hamming" #"boxcar" # from scipy.signal.windows
    audio_window = Windowizer(window_maker(window_shape, int(window_duration*audio_fs)), window_overlap)
    windowed_audio_data, windowed_audio_labels = audio_window.windowize(raw_audio_data, metadata)
    # need to confirm that audio is loaded properly

if __name__ == '__main__':
    unittest.main()