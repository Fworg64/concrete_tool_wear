import pdb
import time
import os

from sklearn import svm
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate
from sklearn.metrics import confusion_matrix, f1_score, make_scorer

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import pywt
import numpy as np

#from dataloading.loaders import load_strain_gauge_limestone, load_cap_limestone
from loader import load_audio_files
from windowizer import Windowizer, window_maker
from custom_pipeline_elements import SampleScaler, ChannelScaler, FFTMag, WaveletDecomposition

number_parallel_jobs = 8#40

window_duration = 0.2 # seconds
window_overlap  = 0.95 # ratio of overlap [0,1)
window_shape    = "boxcar" #"boxcar" # from scipy.signal.windows

number_cross_validations = 8
my_test_size = 0.75

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
freq_transforms = [('db13, 1 level', WaveletDecomposition(basis='db13', decomp_ratio=0.5, sample_size=window_duration*audio_fs)),
                   ('db14, 2 level', WaveletDecomposition(basis='db14', decomp_ratio=0.5, sample_size=window_duration*audio_fs)),
                   ('db15, 3 level', WaveletDecomposition(basis='db15', decomp_ratio=0.5, sample_size=window_duration*audio_fs)),
                   ('db16, 4 level', WaveletDecomposition(basis='db16', decomp_ratio=0.5, sample_size=window_duration*audio_fs)),
                   ('db17, 5 level', WaveletDecomposition(basis='db17', decomp_ratio=0.5, sample_size=window_duration*audio_fs)),
                   ('db18, 6 level', WaveletDecomposition(basis='db18', decomp_ratio=0.5, sample_size=window_duration*audio_fs)),
                   ('db19, 7 level', WaveletDecomposition(basis='db19', decomp_ratio=0.5, sample_size=window_duration*audio_fs))
 ]
classifiers = [('rbf_svm', svm.SVC(class_weight='balanced'))] #, ('linear_svm', svm.LinearSVC(class_weight='balanced', max_iter=10000))]

print("Wavelet max decomp for db1: {0}".format(pywt.dwt_max_level(window_duration*audio_fs, 'db1')))

#pdb.set_trace()
# Do experiment, record data to list
results = [["application", "num_splits", "num_samples", "test_ratio",
            "window_duration", "window_overlap", "window_shape",
            "stand1", "fft", "stand2", "classifier", "mean_score", "std_dev", "acc", "acc_dev"]]
#for name, (data_X, data_Y) in app_data_sets.items():
name = "Concrete Tool Wear"
data_X = windowed_audio_data
data_Y = [wear_classes2ints[label] for label in windowed_audio_labels] 

scorings = ['f1_macro','accuracy']

for ft in freq_transforms:
 for sc1 in scalings1:
  for sc2 in scalings2:
   for cls in classifiers:
      cross_val = ShuffleSplit(n_splits=number_cross_validations, test_size=my_test_size, 
                               random_state = 711711)
      my_pipeline = Pipeline([sc1, ft, sc2, cls])
      scores = cross_validate(my_pipeline, data_X, data_Y, cv=cross_val,
                                scoring=scorings, n_jobs=number_parallel_jobs)
      results.append([name, str(number_cross_validations), str(len(data_X)), str(my_test_size),
                      str(window_duration), str(window_overlap), window_shape,
                      my_pipeline.steps[0][0], my_pipeline.steps[1][0], my_pipeline.steps[2][0],
                      my_pipeline.steps[3][0], 
                      str(scores["test_f1_macro"].mean()), str(scores["test_f1_macro"].std()),
                      str(scores["test_accuracy"].mean()), str(scores["test_accuracy"].std())])
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
