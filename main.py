import pdb
import time
import os

from sklearn import svm
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.metrics import confusion_matrix, f1_score

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import numpy as np

#from dataloading.loaders import load_strain_gauge_limestone, load_cap_limestone
from loader import load_audio_files
from windowizer import Windowizer, window_maker
from custom_pipeline_elements import SampleScaler, ChannelScaler, FFTMag, WaveletDecomposition

number_parallel_jobs = 5#40

window_duration = 0.1 # seconds
window_overlap  = 0.9 # ratio of overlap [0,1)
window_shape    = "boxcar" #"boxcar" # from scipy.signal.windows

number_cross_validations = 10
my_test_size = 0.5

# Load data
#cap_fs = 400 # Samples per second for each channel
#lcm_fs = 537.6 # Samples per second for each channel
audio_fs = 44100 # Samples per second for each channel

print("Loading data...")
this_time = time.time()

#raw_cap_data, cap_metadata = load_cap_limestone()
#raw_lcm_data, lcm_metadata = load_strain_gauge_limestone()
raw_audio_data, metadata = load_audio_files("./raw_audio/classifications.txt")

# Apply windowing
#cap_window = Windowizer(window_maker(window_shape, int(window_duration*cap_fs)), window_overlap)
#lcm_window = Windowizer(window_maker(window_shape, int(window_duration*lcm_fs)), window_overlap)
#windowed_cap_data, windowed_cap_labels = cap_window.windowize(raw_cap_data, cap_metadata)
#windowed_lcm_data, windowed_lcm_labels = lcm_window.windowize(raw_lcm_data, lcm_metadata)
audio_window = Windowizer(window_maker(window_shape, int(window_duration*audio_fs)), window_overlap)
windowed_audio_data, windowed_audio_labels = audio_window.windowize(raw_audio_data, metadata)
      
wear_classes2ints = {"New":0, "Moderate":1, "Worn":2}
wear_ints2classes = {v: k for k,v in wear_classes2ints.items()}

# Build preprocessing lists for pipeline
# scale1: [std, samp, chan, none]
# freq_transform: [abs(rfft()), abs(rfft()).^2, sqrt(abs(rfft())), none]
# scale2: [std, samp, chan, none]

that_time = time.time()
print("Data loaded in {0} sec; performing experiments".format(that_time - this_time),
      end='', flush=True)
this_time = time.time()
# Build pipeline
scalings1 = [("ScaleControl1", None)] # ("FeatureScaler1", StandardScaler())
scalings2 = [("FeatureScaler2", StandardScaler())] #, ("ScaleControl2", None)]
#freq_transforms = [('FFT_Mag', FFTMag(4)), ('FFT_MagSq', FFTMag(4,"SQUARE")),
#                   ('FFT_MagRt', FFTMag(4,"SQRT")), ("FreqControl", None)]
freq_transforms = [('db1, 5 level', WaveletDecomposition(num_levels=5)), 
                   ('db2, 5 level', WaveletDecomposition(basis='db2', num_levels=5)), 
                   ('db3, 5 level', WaveletDecomposition(basis='db3', num_levels=5)),
                   ('db4, 5 level', WaveletDecomposition(basis='db4', num_levels=5)),
                   ('biorg 1.3 5 level', WaveletDecomposition(basis='bior1.3', num_levels=5)),
                   ('biorg 1.5 5 level', WaveletDecomposition(basis='bior1.5', num_levels=5)),
                   ('biorg 2.2 5 level', WaveletDecomposition(basis='bior2.2', num_levels=5)),
]
classifiers = [('rbf_svm', svm.SVC(class_weight='balanced'))] #, ('linear_svm', svm.LinearSVC(class_weight='balanced', max_iter=10000))]

#pdb.set_trace()
# Do experiment, record data to list
results = [["application", "num_splits", "num_samples", "test_ratio",
            "window_duration", "window_overlap", "window_shape",
            "stand1", "fft", "stand2", "classifier", "mean_score", "std_dev"]]
#for name, (data_X, data_Y) in app_data_sets.items():
name = "Concrete Tool Wear"
data_X = windowed_audio_data
data_Y = [wear_classes2ints[label] for label in windowed_audio_labels] 

for ft in freq_transforms:
 for sc1 in scalings1:
  for sc2 in scalings2:
   for cls in classifiers:
      cross_val = ShuffleSplit(n_splits=number_cross_validations, test_size=my_test_size, 
                               random_state = 711711)
      my_pipeline = Pipeline([sc1, ft, sc2, cls])
      scores = cross_val_score(my_pipeline, data_X, data_Y, cv=cross_val,
                                scoring='f1_macro', n_jobs=number_parallel_jobs)
      results.append([name, str(number_cross_validations), str(len(data_X)), str(my_test_size),
                      str(window_duration), str(window_overlap), window_shape,
                      my_pipeline.steps[0][0], my_pipeline.steps[1][0], my_pipeline.steps[2][0],
                      my_pipeline.steps[3][0], str(scores.mean()), str(scores.std())])
      print(".", end='', flush=True)

that_time = time.time()
print(" Done! Took {0} sec; Saving data...".format(that_time - this_time))
# print list and save to file
for result in results:
  print(result)

#pdb.set_trace()
os.makedirs('./out', exist_ok=True)
timestr = time.strftime("%Y%m%d_%H%M%Sresults.csv")
with open('./out/' + timestr, 'w') as f:
  for line in results:
    f.write(','.join(line) + '\n')

print("Have a nice day!")

# Score pipelines using default SVM with linear kernal
# Iterate through material and wear for both sensors
# and all hyperparams using the chosen window settings
# also test different test train splits
