# This file plots the data used in main.py after it has been normallized
# and for each pre-processing technique and the time domain control.
# 12 Mar 2024

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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#help words
shape_options = "hamming,boxcar"
duration_options = "0 - 10 second duration"
overlap_options = "overlap ratio 0-1"
#required inputs
allowed_overlap = [x/100 for x in range(0, 101, 5)]

## Start Simulation Parameters ##

name = "Concrete Tool Wear"

# Computation parameter
number_parallel_jobs = 3

#default values
window_shape    = "hamming" #"boxcar" # from scipy.signal.windows
window_duration = 0.10 # seconds
window_overlap  = 0.5 # ratio of overlap [0,1)

# Machine learning sampling hyperparameters #
number_cross_validations = 8
my_test_size = 0.5

# Load data
audio_fs = 44100 # Samples per second for each channel
downsample_factor = 2

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

static_params_pairs = [ ("name", [name]),
                        ("window_shape", [args.window_shape]),
                        ("window_duration", [args.window_duration]),
                        ("window_overlap", [args.window_overlap]),
                        ("window_len", [window_len]),
                        ("number_parallel_jobs", [number_parallel_jobs]),
                        ("number_cross_validations", [number_cross_validations]),
                        ("my_test_size", [my_test_size]),
                        ("audio_fs", [audio_fs]),
                        ("downsample_factor", [downsample_factor]),
                        ("load_date_time", [this_time]) ] # All parameters 

## End default parameters and loading ##
## End parameters ##

# Apply windowing

audio_window = Windowizer(window_maker(args.window_shape, int(args.window_duration*audio_fs)), args.window_overlap)
windowed_audio_data, windowed_audio_labels = audio_window.windowize(raw_audio_data, metadata)
      
wear_list = ["New", "Moderate", "Worn"]
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
freq_transforms1 = [
                    ("FFT Mag.", FFTMag(1, power=None)),
                    ('FFT_MagBoost10', FFTMag(1, power="BOOSTm10")),
                    ('FFT_MagFilt15', FFTMag(1, power="FILT15", after=None)),
                    ('FFT_MagFilt15B20', FFTMag(1, power="FILT15", after="BOOST20")),
]

# Do experiment, record data to list
# Save results from experiments to list of list of pairs
results_list = []
data_X = windowed_audio_data
data_Y = [wear_classes2ints[label] for label in windowed_audio_labels] 

# Get Worn data, Mod data, New data
labelled_data = {label:[] for label in windowed_audio_labels}
for x,y in zip (data_X, data_Y):
  labelled_data[wear_ints2classes[y]].append(x)

transformed_data = {label:{} for label in windowed_audio_labels}

for ft1 in freq_transforms1:
  for sc2 in scalings2:
    # Fit scaler for population data
    my_pipeline = Pipeline([ft1, sc2])
    my_pipeline.fit(data_X)
    for wear in wear_list:
      # Transform each subgroup using population fit
      transformed_data[wear][ft1[0]] = my_pipeline.transform(labelled_data[wear])

      # Progress UI
      print(f"{wear} {ft1[0]} is {transformed_data[wear][ft1[0]].shape}.", flush=True)

that_time = time.time()
print(" Done! Took {0} sec; Saving data...".format(that_time - this_time))
#print(transformed_data)


# Compute distributions
# Plot for each wear type and technique
# Sort data by wear level
# Find average and variance of each wear level and processing technique
names = []
values_list = []
num_samples_list = []
dim_samples_list = []
for ft1 in freq_transforms1:
  for wear in wear_list:
    names.append(f"{wear} with {ft1[0]} [{downsample_factor}x downsample]")
    values_list.append(np.array(transformed_data[wear][ft1[0]]))
    num_samples_list.append(values_list[-1].shape[0])
    dim_samples_list.append(values_list[-1].shape[1])

avgs = {}
devs = {}
nums = {}
dims = {}
for val, name, num, dim in zip(values_list, names, num_samples_list, dim_samples_list):
  avgs[name] = np.mean(val, axis=0)
  devs[name] = np.std(val, axis=0)
  nums[name] = num
  dims[name] = dim

print("names:", names)
print("nums:", nums)
print("dims:", dims)
#print(avgs)

def calculate_cont_t_stat(means1, means2, dev1, dev2, num_samples1, num_samples2):
  t_stat_list = []
  for m1, m2, d1, d2 in zip(means1, means2, dev1, dev2):
    t_stat_list.append(
        (m1 - m2) / np.sqrt(d1 * d1 / num_samples1 + d2 * d2 / num_samples2))
  #two_tails_value = 2.581 # Approximate 99% chance distributions are different for about 940 degrees of freedom 
  two_tails_value = 3.291 # Approximate 99.9% chance distributions are different for about 940 degrees of freedom 
  t_reject_null = [float(np.abs(t_stat) > two_tails_value) for t_stat in t_stat_list]
  return t_stat_list, t_reject_null

# Plot distribution vectors for each wear level for frequency
# Set figure sizes
fontsize = 22
legendfontsize = 14 
plt.rc('font', size=fontsize, family='sans')
plt.rc('axes', titlesize=fontsize)
plt.rc('axes', labelsize=fontsize)
plt.rc('legend', fontsize=legendfontsize)
plot_width = 0.8 # pt

fig1, axs1 = plt.subplots(3,2, sharex="col", sharey=True) # time domain and fftmag
fig2, axs2 = plt.subplots(3,2, sharex="col", sharey=True) # freq domain sqrt and sq
fig3, axs3 = plt.subplots(2,1) # t test for distributions, time domain and fft mag

mosaic_names_list = ["A", "B", "C", "D"]
fig, axes = plt.subplot_mosaic([["B", "A"], ["C", "A"], ["D", "A"]])
mosaic_leg_loc = ["upper right", "lower right", "lower right"]

for idx,ft1 in enumerate(freq_transforms1):
  for index in range(len(wear_list)):

    namedex = idx * len(wear_list) + index

    # Plot domain
    print(f"Plotting {names[namedex]}")
    domain_vals = np.linspace(0, audio_fs/2, len(avgs[names[namedex]]))
    xlabel = "Freq (Hz)"
    if "Control" in names[namedex]:
      domain_vals = np.linspace(0, window_duration, len(avgs[names[namedex]]))
      xlabel = "Time (s)"
      print("DOMAIN CONTROL!")

    # Fig 4
    if idx == 0:
      # FFTs, first freq1 value
      std_err = devs[names[namedex]] / np.sqrt(nums[names[namedex]])
      axes[mosaic_names_list[1 + index]].fill_between(domain_vals,
        avgs[names[namedex]] + std_err,
        avgs[names[namedex]] - std_err,
        alpha=0.5, color='tab:purple')
      axes[mosaic_names_list[1 + index]].plot(domain_vals, avgs[names[namedex]],
        color='tab:green', linewidth=plot_width)
      axes[mosaic_names_list[1 + index]].set_title(f"{names[namedex]}, N={nums[names[namedex]]}")
      axes[mosaic_names_list[1 + index]].legend(["Mean", r"$\pm$ 1 Std. Err."], 
          loc=mosaic_leg_loc[index], ncol=1, prop = {"size":18})
      axes[mosaic_names_list[1 + index]].set_ylabel("Normalized Mag.")
      axes[mosaic_names_list[1 + index]].set_ylim(-0.4, 0.4)
      axes[mosaic_names_list[1 + index]].grid(True, which="minor")
      axes[mosaic_names_list[1 + index]].minorticks_on()
      axes[mosaic_names_list[1 + index]].tick_params(which="minor", bottom=False, left=False)
      axes[mosaic_names_list[1 + index]].grid(True, which="major", linewidth=2, color='k')
      axes["D"].set_xlabel("Frequency (Hz)")
      plt.suptitle("Frequency Spectra Comparison after Normalization")
      # Comparison graphs
      # Calculate tstat for distributions
      if wear_list[0] in names[namedex]: # new, compare with others
          # New vs Mod.
          t_stat, t_reject_null = calculate_cont_t_stat(
            avgs[names[namedex]], avgs[names[namedex + 1]],
            devs[names[namedex]], devs[names[namedex + 1]], 
            nums[names[namedex]], nums[names[namedex + 1]])
          axes["A"].fill_between(domain_vals, t_reject_null, 
              step="mid", label=f"New vs {names[namedex + 1]}", color="green",
              lw=0.00) 
          print(f"fig 4 Plotted {names[namedex]} vs {names[namedex + 1]}")
          print(f"fig 4 Percent of regions with significant diffs: {sum(t_reject_null)/len(t_reject_null)}")
          # New vs Worn
          t_stat, t_reject_null = calculate_cont_t_stat(
            avgs[names[namedex]], avgs[names[namedex + 2]],
            devs[names[namedex]], devs[names[namedex + 2]], 
            nums[names[namedex]], nums[names[namedex + 2]])
          offset = [t - 2.0 for t in t_reject_null]
          axes["A"].fill_between(domain_vals, offset, -2.0, 
              step="mid", label=f"New vs {names[namedex + 2]}", color="blue",
              lw=0.00) 
          print(f"fig 4 Plotted {names[namedex]} vs {names[namedex + 2]}")
          print(f"fig 4 Percent of regions with significant diffs: {sum(t_reject_null)/len(t_reject_null)}")
      if wear_list[1] in names[namedex]: # mod, compare with others
          # Mod vs Worn
          t_stat, t_reject_null = calculate_cont_t_stat(
            avgs[names[namedex]], avgs[names[namedex + 1]],
            devs[names[namedex]], devs[names[namedex + 1]], 
            nums[names[namedex]], nums[names[namedex + 1]])
          offset = [t - 4.0 for t in t_reject_null]
          axes["A"].fill_between(domain_vals, offset, -4.0, 
              step="mid", label=f"Mod vs {names[namedex + 1]}", color="brown",
              lw=0.00) 
          ticks = [0.5, -1.5, -3.5]
          labels = ["New Vs Mod.", "New Vs Worn", "Mod. Vs Worn"]
          axes["A"].set_yticks(ticks)
          axes["A"].grid(True, which="minor", axis='x')
          axes["A"].minorticks_on()
          axes["A"].tick_params(which="minor", bottom=False, left=False)
          axes["A"].grid(True, which="major", axis='x', linewidth=2, color='k')
          axes["A"].set_yticklabels(labels, rotation=90, va="center") 
          axes["A"].set_title(f"Bins with different means, p<.001, {dims[names[namedex]]} bins") 
          axes["A"].set_xlabel("Frequency (Hz)")
          print(f"fig 4 Plotted {names[namedex]} vs {names[namedex + 1]}")
          print(f"fig 4 Percent of regions with significant diffs: {sum(t_reject_null)/len(t_reject_null)}")

    if idx in [0,1]:
      axs1[index][idx].fill_between(domain_vals,
        avgs[names[namedex]] + devs[names[namedex]],
        avgs[names[namedex]] - devs[names[namedex]],
        alpha=0.5, color='tab:purple')
      axs1[index][idx].plot(domain_vals, avgs[names[namedex]],
      color='tab:green', linewidth=plot_width)
      axs1[index][idx].set_title(names[namedex])
      axs1[index][idx].legend(["Mean", r"$\pm$ 1 Std. Dev."], loc="upper right", ncol=2, prop = {"size":18})
      axs1[index][idx].set_ylabel("Normalized Mag.")
      axs1[index][idx].set_ylim(-2.8, 2.8)
      axs1[index][idx].grid(True, which="minor")
      axs1[index][idx].minorticks_on()
      axs1[index][idx].tick_params(which="minor", bottom=False, left=False)
      axs1[index][idx].grid(True, which="major", linewidth=2, color='k')
      if index == 2:
        axs1[index][idx].set_xlabel(xlabel)

    else:
      namedex = idx * len(wear_list) + index
      axs2[index][idx-2].fill_between(domain_vals,
        avgs[names[namedex]] + devs[names[namedex]],
        avgs[names[namedex]] - devs[names[namedex]],
        alpha=0.5, color='tab:purple')
      axs2[index][idx-2].plot(domain_vals, avgs[names[namedex]],
      color='tab:green', linewidth=plot_width)
      axs2[index][idx-2].set_title(names[namedex])
      axs2[index][idx-2].legend(["Mean", r"$\pm$ 1 Std. Dev."], loc="upper right", ncol=2, prop = {"size":18})
      axs2[index][idx-2].set_ylabel("Normalized Mag.")
      axs2[index][idx-2].set_ylim(-2.8, 2.8)
      axs2[index][idx-2].grid(True, which="minor")
      axs2[index][idx-2].minorticks_on()
      axs2[index][idx-2].tick_params(which="minor", bottom=False, left=False)
      axs2[index][idx-2].grid(True, which="major", linewidth=2, color='k')
      if index == 2:
        axs2[index][idx-2].set_xlabel(xlabel)

    # fig 3
    if idx in [0,1]:
        # Calculate tstat for distributions
        if wear_list[0] in names[namedex]: # new, compare with others
          # New vs Mod.
          t_stat, t_reject_null = calculate_cont_t_stat(
            avgs[names[namedex]], avgs[names[namedex + 1]],
            devs[names[namedex]], devs[names[namedex + 1]], 
            nums[names[namedex]], nums[names[namedex + 1]])
          axs3[idx].fill_between(domain_vals, t_reject_null, 
              step="mid", label=f"New vs {names[namedex + 1]}", color="green",
              lw=0.00) 
          print(f"Plotted {names[namedex]} vs {names[namedex + 1]}")
          print(f"Percent of regions with significant diffs: {sum(t_reject_null)/len(t_reject_null)}")
          # New vs Worn
          t_stat, t_reject_null = calculate_cont_t_stat(
            avgs[names[namedex]], avgs[names[namedex + 2]],
            devs[names[namedex]], devs[names[namedex + 2]], 
            nums[names[namedex]], nums[names[namedex + 2]])
          offset = [t - 2.0 for t in t_reject_null]
          axs3[idx].fill_between(domain_vals, offset, -2.0, 
              step="mid", label=f"New vs {names[namedex + 2]}", color="blue",
              lw=0.00) 
          print(f"Plotted {names[namedex]} vs {names[namedex + 2]}")
          print(f"Percent of regions with significant diffs: {sum(t_reject_null)/len(t_reject_null)}")
        if wear_list[1] in names[namedex]: # mod, compare with others
          # Mod vs Worn
          t_stat, t_reject_null = calculate_cont_t_stat(
            avgs[names[namedex]], avgs[names[namedex + 1]],
            devs[names[namedex]], devs[names[namedex + 1]], 
            nums[names[namedex]], nums[names[namedex + 1]])
          offset = [t - 4.0 for t in t_reject_null]
          axs3[idx].fill_between(domain_vals, offset, -4.0, 
              step="mid", label=f"Mod vs {names[namedex + 1]}", color="brown",
              lw=0.00) 
          axs3[idx].set_title(f"Bins with different means, p<.001, using {ft1[0]} and {dims[names[namedex]]} bins") 
          ticks = [0.5, -1.5, -3.5]
          labels = ["New Vs Mod.", "New Vs Worn", "Mod. Vs Worn"]
          axs3[idx].set_yticks(ticks)
          axs3[idx].set_yticklabels(labels)
          axs3[idx].grid(True, which="minor", axis='x')
          axs3[idx].minorticks_on()
          axs3[idx].tick_params(which="minor", bottom=False, left=False)
          axs3[idx].grid(True, which="major", axis='x', linewidth=2, color='k')
          if idx == 0:
            axs3[idx].set_xlabel("Time (s)")
          else:
            axs3[idx].set_xlabel("Freq. (Hz)")
          print(f"Plotted {names[namedex]} vs {names[namedex + 1]}")
          print(f"Percent of regions with significant diffs: {sum(t_reject_null)/len(t_reject_null)}")

fig1.show()
fig2.show()
fig3.show()
fig.show()
  
## T test stat for difference in distributions



plt.show(block=False)
input("Press Enter to close...")



