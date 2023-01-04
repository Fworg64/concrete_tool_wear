import pdb
import time
import os
import argparse
import decimal

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate
from sklearn.metrics import confusion_matrix, f1_score, make_scorer

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import pywt
import numpy as np
import pandas as pd

#from dataloading.loaders import load_strain_gauge_limestone, load_cap_limestone
from loader import load_audio_files
from windowizer import Windowizer, window_maker
from custom_pipeline_elements import SampleScaler, ChannelScaler, FFTMag, WaveletDecomposition

number_parallel_jobs = 8

## Start Simulation Parameters ##

#default values
window_shape    = "hamming" #"boxcar" # from scipy.signal.windows
window_duration = 0.2 # seconds
window_overlap  = 0.5 # ratio of overlap [0,1)
#help words
shape_options = "hamming,boxcar"
duration_options = "0 - 10 second duration"
overlap_options = "overlap ratio 0-1"
#required inputs
allowed_overlap = [x/100 for x in range(0, 101, 5)]


# Machine learning sampling hyperparameters #
number_cross_validations = 8
my_test_size = 0.5

# Load data
audio_fs = 44100 # Samples per second for each channel

print("Loading data...")
this_time = time.time()

# Load and Downsample
downsample_factor = 16
raw_audio_data, metadata = load_audio_files("./raw_audio/classifications.txt", integer_downsample=downsample_factor)
audio_fs = int(audio_fs/downsample_factor)

window_len = int(window_duration*audio_fs)

## End default parameters and loading ##
## Allow command line overrides of parameters ##

# Making command line argument for window shape
parser = argparse.ArgumentParser()
parser.add_argument("--window_shape", default=window_shape, type=str,
  help=shape_options)
# Making command line argument for window duration
parser.add_argument("--window_duration", default=window_duration, type=float,
  help=duration_options)
# Making command line argument for window overlap
parser.add_argument("--window_overlap", type=float, default=window_overlap,
  help=overlap_options)

args = parser.parse_args()
# Making the overlap between 0-1
if args.window_overlap > 1:
  raise Exception("Sorry, no numbers above 1")
else:
  pass
if args.window_overlap < 0:
  raise Exception("Sorry, no numbers below zero") 
else:
  pass
# Printing what the values are
if args.window_shape:
    print("window shape is",args.window_shape)
if args.window_duration:
    print("window duration is",args.window_duration)
if args.window_overlap:
    print("window overlap is",args.window_overlap)
else: 
      print("windows don't overlap")

## End parameters ##

# Apply windowing

audio_window = Windowizer(window_maker(args.window_shape, int(args.window_duration*audio_fs)), args.window_overlap)
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
freq_transforms1 = [('FFT_Mag', FFTMag(1))] #,("FreqControl1", None)]
freq_transforms2 = [
                    ("FreqControl2", None)
                    ]
classifiers = [('rbf_svm', svm.SVC(class_weight='balanced')),
               #('MLPClass1', MLPClassifier(solver='lbfgs', activation='relu', 
               # alpha=1e-10, tol=1e-8,
               # hidden_layer_sizes=(windowed_audio_data[0].shape[0], 
               #                     windowed_audio_data[0].shape[0]), 
               # max_iter=300, verbose=False)),
               #('MLPClass2', MLPClassifier(solver='lbfgs', activation='relu', 
               # alpha=1e-10, tol=1e-8,
               # hidden_layer_sizes=(2*windowed_audio_data[0].shape[0], 
               #                     2*windowed_audio_data[0].shape[0]),
               # max_iter=300, verbose=False)),
               #('MLPClass3', MLPClassifier(solver='lbfgs', activation='relu', 
               # alpha=1e-10, tol=1e-8,
               # hidden_layer_sizes=(2*windowed_audio_data[0].shape[0], 
               #                     2*windowed_audio_data[0].shape[0], 
               #                     windowed_audio_data[0].shape[0]), 
               # max_iter=300, verbose=False))
               ('K5N', KNeighborsClassifier(n_neighbors=5)),
               #('K15N', KNeighborsClassifier(n_neighbors=15)),
               #('K25N', KNeighborsClassifier(n_neighbors=25))
] 


#pdb.set_trace()
# Do experiment, record data to list
results = [["application", "num_splits", "num_samples", "test_ratio",
            "window_duration", "window_overlap", "window_shape",
            "freq1", "freq2", "stand2", "classifier", "mean_score", "std_dev", "acc", "acc_dev"]]
# Save results from experiments to dataframe

results_frame = pd.DataFrame(columns=results[0][0:1], dtype=str) 



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

      # Concat to data frame
      experiment_data_dict = {results[0][0] : [name],
                              results[0][1] : [str(number_cross_validations)]}
      experiment_data_frame = pd.DataFrame(data=experiment_data_dict, dtype=str)
      results_frame = pd.concat([results_frame, experiment_data_frame], ignore_index=True)

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

print("results frame")
print(results_frame)

pdb.set_trace()

## Write file
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
