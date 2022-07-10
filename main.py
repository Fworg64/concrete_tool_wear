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

number_parallel_jobs = 40

window_duration = 0.2 # seconds
window_overlap  = 0.5 # ratio of overlap [0,1)
window_shape    = "hamming" #"boxcar" # from scipy.signal.windows

number_cross_validations = 100
my_test_size = 0.5

# Load data
#cap_fs = 400 # Samples per second for each channel
#lcm_fs = 537.6 # Samples per second for each channel
audio_fs = 44100 # Samples per second for each channel

print("Loading data...")
this_time = time.time()

# Load and Downsample
downsample_factor = 5
raw_audio_data, metadata = load_audio_files("./raw_audio/classifications.txt", integer_downsample=downsample_factor)
audio_fs = int(audio_fs/downsample_factor)

window_len = int(window_duration*audio_fs)

# Apply windowing
audio_window = Windowizer(window_maker(window_shape, window_len), window_overlap)
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
#scalings1 = [("ScaleControl1", None)] # ("FeatureScaler1", StandardScaler())
scalings2 = [("FeatureScaler2", StandardScaler())] #, ("ScaleControl2", None)]
freq_transforms1 = [('FFT_outer', FFTMag(1, "OUTER")), 
                    ("FreqControl1", None)]
freq_transforms2 = [
                    ("FreqControl2", None)
                    ]
classifiers = [('rbf_svm', svm.SVC(class_weight='balanced'))] #, ('linear_svm', svm.LinearSVC(class_weight='balanced', max_iter=10000))]


#pdb.set_trace()
# Do experiment, record data to list
results = [["application", "num_splits", "num_samples", "test_ratio",
            "window_duration", "window_overlap", "window_shape",
            "freq1", "freq2", "stand2", "classifier", "mean_score", "std_dev", "acc", "acc_dev"]]
#for name, (data_X, data_Y) in app_data_sets.items():
name = "Concrete Tool Wear"
data_X = windowed_audio_data
data_Y = [wear_classes2ints[label] for label in windowed_audio_labels] 

scorings = ['f1_macro','accuracy']

for ft1 in freq_transforms1:
 for ft2 in freq_transforms2:
  for sc2 in scalings2:
   for cls in classifiers:
      cross_val = ShuffleSplit(n_splits=number_cross_validations, test_size=my_test_size, 
                               random_state = 711711)
      my_pipeline = Pipeline([ft1, ft2, sc2, cls])
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
