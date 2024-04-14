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

from sklearn.model_selection import GridSearchCV

import pywt
import numpy as np
import pandas as pd

#from dataloading.loaders import load_strain_gauge_limestone, load_cap_limestone
from loader import load_audio_files
from windowizer import Windowizer, window_maker
from custom_pipeline_elements import SampleScaler, ChannelScaler, FFTMag, WaveletDecomposition

#help words
shape_options = "hamming,boxcar"
duration_options = "0 - 10 second duration"
overlap_options = "overlap ratio 0-1"
#required inputs
allowed_overlap = [x/100 for x in range(0, 101, 5)]

## Start Simulation Parameters ##

name = "Concrete Tool Wear"

# Computation parameter
number_parallel_jobs = 40

#default values
window_shape    = "hamming" #"boxcar" # from scipy.signal.windows
window_duration = 0.2 # seconds
window_overlap  = 0.5 # ratio of overlap [0,1)

# Machine learning sampling hyperparameters #
number_cross_validations = 40
my_test_size = 0.7

# Load data
audio_fs = 44100 # Samples per second for each channel
downsample_factor = 12

print("Loading data...")
this_time = time.time()

# Load and Downsample, adjust audio_fs
audio_fs = int(audio_fs/downsample_factor)
raw_audio_data, metadata = load_audio_files(
    "./raw_audio/classifications.txt", 
    integer_downsample=downsample_factor, lpf=True)


## Allow command line overrides
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

window_len = int(args.window_duration*audio_fs)
## End command line parsing

static_params_pairs = [ ("name", name),
                        ("window_shape", args.window_shape),
                        ("window_duration", args.window_duration),
                        ("window_overlap", args.window_overlap),
                        ("window_len", window_len),
                        ("number_parallel_jobs", number_parallel_jobs),
                        ("number_cross_validations", number_cross_validations),
                        ("my_test_size", my_test_size),
                        ("audio_fs", audio_fs),
                        ("downsample_factor", downsample_factor),
                        ("load_date_time", this_time) ] # All parameters 

## End default parameters and loading ##
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

freq_transforms1 = [('FFT_Mag', FFTMag(1, power=None)),
#                    ('FFT_MagLPF75', FFTMag(1, power="FILT75")),
#                    ('FFT_MagLPF50', FFTMag(1, power="FILT50")),
#                    ('FFT_MagLPF25', FFTMag(1, power="FILT25")),
#                    ('FFT_MagLPF15B10', FFTMag(1, power="FILT15", after="BOOST10")),
#                    ('FFT_MagLPF15B20', FFTMag(1, power="FILT15", after="BOOST20")),
#                    ('FFT_MagLPF15B50', FFTMag(1, power="FILT15", after="BOOST50")),
#                    ("FreqControl1", None),
]

freq_transforms2 = [
                    ("FreqControl2", None)
                    ]

scalings2 = [("FeatureScaler2", StandardScaler())] #, ("ScaleControl2", None)]

classifiers = [
               ('rbf_svm', svm.SVC(class_weight='balanced')),
               ('MLPClass1', MLPClassifier(solver='lbfgs', activation='relu', 
                alpha=1e-10, tol=1e-8,
                hidden_layer_sizes=(windowed_audio_data[0].shape[0], 
                                    windowed_audio_data[0].shape[0]), 
                max_iter=900, verbose=False)),
#               ('MLPClass2', MLPClassifier(solver='lbfgs', activation='relu', 
#                alpha=1e-10, tol=1e-8,
#                hidden_layer_sizes=(windowed_audio_data[0].shape[0], 
#                                    windowed_audio_data[0].shape[0], 
#                                    windowed_audio_data[0].shape[0]),
#                max_iter=900, verbose=False)),
#               ('MLPClass3', MLPClassifier(solver='lbfgs', activation='relu', 
#                alpha=1e-10, tol=1e-8,
#                hidden_layer_sizes=(windowed_audio_data[0].shape[0], 
#                                    windowed_audio_data[0].shape[0], 
#                                    windowed_audio_data[0].shape[0], 
#                                    windowed_audio_data[0].shape[0]), 
#                max_iter=900, verbose=False)),
               ('K5N', KNeighborsClassifier(n_neighbors=5)),
#               ('K10N', KNeighborsClassifier(n_neighbors=10)),
#               ('K15N', KNeighborsClassifier(n_neighbors=15))
] 



# Do experiment, record data to list
# Save results from experiments to list of list of pairs
results_list = []
data_X = np.array(windowed_audio_data)
data_Y = np.array([wear_classes2ints[label] for label in windowed_audio_labels] )

scorings = ['f1_macro','accuracy']
confusion_mats = []

for ft1 in freq_transforms1:
 for ft2 in freq_transforms2:
  for sc2 in scalings2:
   for cls in classifiers:
      cross_val = ShuffleSplit(n_splits=number_cross_validations, test_size=my_test_size, 
                               random_state = 711711)
      my_pipeline = Pipeline([ft1, ft2, sc2, cls])

      params = "None"
      if (cls[0] == 'rbf_svm'):
        print("Fitting svm hyper param")
        C_range = np.logspace(0, 3, 7)
        gamma_range = np.logspace(-4, -1, 7)
        param_grid = {"rbf_svm__gamma" : gamma_range, 
                          "rbf_svm__C" : C_range,
                      "rbf_svm__class_weight" : ["balanced"]}
        grid = GridSearchCV(my_pipeline, param_grid=param_grid, cv=cross_val, verbose=1, n_jobs=number_parallel_jobs)
        grid.fit(data_X, data_Y)

        print(
             "The best parameters are %s with a score of %0.2f"
             % (grid.best_params_, grid.best_score_)
        )
        params = str(grid.best_params_)        

        my_pipeline.set_params(**grid.best_params_)

      scores = cross_validate(my_pipeline, data_X, data_Y, cv=cross_val,
                                scoring=scorings, n_jobs=number_parallel_jobs)

      # Get confusion matricies for cross validation too
      for traindex, testdex in cross_val.split(data_X):
        x_train, x_test = data_X[traindex], data_X[testdex]
        y_train, y_test = data_Y[traindex], data_Y[testdex]
        my_pipeline.fit(x_train, y_train)
        conf_mat = confusion_matrix(y_test, my_pipeline.predict(x_test), normalize="true")
        confusion_mats.append(conf_mat)
        print('|', end='', flush=True)

      # Concat to data frame
      dynamic_params_pairs = [("num_samples", str(len(data_X))),
                              ("sample_lens", data_X[0].shape[0]),
                              ("freq1", my_pipeline.steps[0][0]), 
                              ("freq2", my_pipeline.steps[1][0]), 
                              ("stand2", my_pipeline.steps[2][0]),
                              ("classifier", my_pipeline.steps[3][0]),
                              ("params", params),
                              ("mean_score", str(scores["test_f1_macro"].mean())),
                              ("std_dev", str(scores["test_f1_macro"].std())),
                              ("acc", str(scores["test_accuracy"].mean())), 
                              ("acc_dev", str(scores["test_accuracy"].std()))]

      # Append score values to get accurate distribution
      f1_vals_pairs = [(f"f1 {idx}", score) for idx, score in enumerate(scores["test_f1_macro"])]
      acc_vals_pairs = [(f"acc {idx}", score) for idx, score in enumerate(scores["test_accuracy"])]

      # Create data frame from static and dynamic data, append to results dataframe
      experiment_data_pairs = static_params_pairs + dynamic_params_pairs + f1_vals_pairs + acc_vals_pairs
      results_list.append(experiment_data_pairs)

      # Progress UI
      print(f". T+: {time.time() - this_time} seconds, {cls[0]} completed.", flush=True)
      print(f"F1 scores: {scores['test_f1_macro'].mean()} +/- {scores['test_f1_macro'].std()}")
      print(f"Accuracy scores: {scores['test_accuracy'].mean()} +/- {scores['test_accuracy'].std()}")

 print(f". T+: {time.time() - this_time} seconds, {ft1[0]} completed.", flush=True)

# Get timestamp before big conf mat print
that_time = time.time()
print("conf_mats:")
print(confusion_mats)
# Print timestamp after conf mat print for viewing
print(" Done! Took {0} sec; Saving data...".format(that_time - this_time))

# DONT print list
##for result in results:
##  print(result)

## Save file
#
# Get list of column names from first entry in results list
result_columns = [item[0] for item in results_list[0]] 

# Make list of rows as tuples
result_rows = []
for res in results_list:
  names  = [item[0] for item in res] # Unused
  values = [item[1] for item in res] 
  result_rows.append(values)

results_frame = pd.DataFrame(data=result_rows, columns=result_columns)

print("results frame")
print(results_frame)

## Write file with timestamp, make dir if not exist
os.makedirs('./out', exist_ok=True)
timestr = time.strftime("%Y%m%d_%H%M%Sresults.csv")
outfilename = './out/' + "CONCRETE_" + timestr
results_frame.to_csv(outfilename, index_label=False, index=False) 

print(f"File saved to {outfilename}")

print("Have a nice day!")

