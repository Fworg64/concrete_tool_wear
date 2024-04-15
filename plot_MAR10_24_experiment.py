# This script plots the results from the experiment concluded
# on March 10, 2024

import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pdb

# Set figures
fontsize = 30
legendsize = 30
plt.rc('font', size=fontsize, family='serif')
plt.rc('axes', titlesize=fontsize)
plt.rc('axes', labelsize=fontsize)
plt.rc('legend', fontsize=legendsize)
plot_width = 1.8 # pt

# load all data in folder

results_path = './results_2024_MAR10/'

file_names = glob.glob(results_path + "*.csv")

frames = []

for name in file_names:
  frames.append(pd.read_csv(name))

mega_frame = pd.concat(frames)

print(mega_frame)

# for each downsampling level
# plot classification method, freqeuncy method series against window len
downsample_factors = mega_frame.downsample_factor.unique()
classification_methods = mega_frame.classifier.unique()
frequency_methods = mega_frame.freq1.unique()
win_lens = mega_frame.window_duration.unique()
audio_freqs = mega_frame.audio_fs.unique()

dflist = downsample_factors.tolist()
dflist.sort()

cmlist = classification_methods.tolist()
cmlist.sort(reverse=True)

fmlist = frequency_methods.tolist()
fmlist.sort()

wllist = win_lens.tolist()
wllist.sort()

aflist = audio_freqs.tolist()
aflist.sort()

print("downsamplings: ")
print(dflist)
print("classification: ")
print(cmlist)
print("freq trans : ")
print(fmlist)
print("window len: ")
print(wllist)
print("audio freqs: ")
print(aflist)

downsample_color_dict = {
    2: "#FF0000", 3: "#FFA500", 4: "#FFFF00",
    8: "#008000", 12: "#0000FF", 16: "#4B0082"
}

freq_colors_dict = {"FFT_Mag":"red", "FFT_Rt":"blue", "FFT_Sq": "green", "FreqControl1": "orange"}
downsample_shapes_dict = {
    2: "*", 3: "X", 4: "D",
    8: "s", 12: "^", 16: "o",
}
downsample_sizes_dict = {
    2: 15, 3: 13, 4: 11,
    8: 15, 12: 13, 16: 11,
}

audio_fs_from_dsf_dict = {df:af for df, af in zip(dflist, aflist)}
family_method_cols_dict = {"svm": 3, "knn": 3, "ffnn": 3}
method_display_names_dict = {"rbf_svm": "SVM RBF", 
                             "K5N": "KNN: 5 Neighbors",
#                             "K10N": "KNN(10)",
#                             "K15N": "KNN(15)",
                             "MLPClass1": "MLP: 2-Layer",
#                             "MLPClass2": "MLP B",
#                             "MLPClass3": "MLP C"
}
family_display_names_dict = {"svm": "SVM", "knn": "KNN", "ffnn": "MLP"}

# Just FFT Mag
freq = fmlist[0]

classification_group_members_dict = {
    "svm":{"names":["rbf_svm"], "data":[]}, 
    "knn":{"names":['K5N'], "data":[]},
   "ffnn":{"names":['MLPClass1'], "data":[]}
  }

# Make figure for each classifier
data_dict = {}
for classification in ["rbf_svm", "K5N", "MLPClass1"]:
    # Sort classification groups

    classifier_data = mega_frame.loc[mega_frame["classifier"] == classification]

    print(classifier_data.columns)
    plot_frame = classifier_data[["mean_score", "std_dev", "acc", "acc_dev", 
                             "window_duration", "classifier", "freq1", "downsample_factor"]]

    # Make figure
    fig, axe = plt.subplots()

    metrics = ["mean_score", "std_dev", "acc", "acc_dev"]
    data_dict[classification] = {}
    legend_artists = []
    legend_labels = []
    for downsample in dflist:
        data_dict[classification][downsample] = {met: [] for met in metrics}
        # Select value for chosen classifier method and frequency method and window len
        for wlen in wllist:
          for met in metrics:
            data_dict[classification][downsample][met].append(
              plot_frame[(plot_frame["downsample_factor"] == downsample) 
              & (plot_frame["window_duration"] == wlen)][met].values[0]
            )
        print(f"Downsampling: {downsample}> For {classification}, with {freq}")
        print("Mean F1  : " + str(data_dict[classification][downsample]["mean_score"]))
        print("Std. dev : " + str(data_dict[classification][downsample]["std_dev"]))
        # Plot trace
        
        # Plot classification with different shape for different hyperparameters
        # Use diferent color for different downsampling 
        std_dev = np.array(data_dict[classification][downsample]["std_dev"]) 
        axe.fill_between(wllist,
          np.array(data_dict[classification][downsample]["mean_score"]) + std_dev,
          np.array(data_dict[classification][downsample]["mean_score"]) - std_dev,
          alpha=0.8, 
          color=downsample_color_dict[downsample],
          zorder=0,
        )

        dashes, = axe.plot(wllist, data_dict[classification][downsample]["mean_score"], 
                 color='k', #downsample_color_dict[downsample],
                 linestyle='--', linewidth=plot_width,
                 zorder=10,
        )
        outlines, = axe.plot(wllist, data_dict[classification][downsample]["mean_score"], 
                 color='k',
                 marker=downsample_shapes_dict[downsample],
                 markersize=downsample_sizes_dict[downsample] + 4,
                 linewidth=0,
                 zorder=20,
                 )
        dots, = axe.plot(wllist, data_dict[classification][downsample]["mean_score"], 
                 label=f"{downsample}X Downsample",
                 color=downsample_color_dict[downsample],
                 marker=downsample_shapes_dict[downsample],
                 markersize=downsample_sizes_dict[downsample],
                 linewidth=0,
                 zorder=30,
                 )
        legend_artists.append((dashes, outlines, dots))
        legend_labels.append(f"{downsample}X downsample")
        
    plt.suptitle(f"{method_display_names_dict[classification]} Mean F1 Score +/- 1 std. dev.")
    axe.set_title("70:30 test:train split, N=40")
    axe.set_ylim([0.5, 1])
    if "K5N" in classification:
      axe.legend(legend_artists, legend_labels, ncol=1 , loc=(0.61, 0.48))
    else:
      axe.legend(legend_artists, legend_labels, ncol=1 , loc=(0.61, 0.03))
    axe.set_xlabel("Window Length (s)")
    axe.set_ylabel("F1 Score out of 1.00")
    plt.xticks(wllist)
    axe.grid(True, which="minor", axis='both')
    axe.minorticks_on()
    axe.tick_params(which="minor", bottom=False, left=False)
    axe.grid(True, which="major", axis='both', linewidth=2, color='k')
    axe.set_axisbelow(True)
 
plt.show(block=False)
input("Press enter to close.")

# Use win_lens as x axis




