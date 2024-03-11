# This script plots the results from the experiment concluded
# on March 10, 2024

import glob
import pandas as pd
import matplotlib.pyplot as plt

import pdb

# Set figures
fontsize = 30
legendsize = 20
plt.rc('font', size=fontsize, family='sans')
plt.rc('axes', titlesize=fontsize)
plt.rc('axes', labelsize=fontsize)
plt.rc('legend', fontsize=legendsize)
plot_width = 3.6 # pt

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
dflist.sort(reverse=True)

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

downsample_colors_dict = {}

freq_colors_dict = {"FFT_Mag":"red", "FFT_Rt":"blue", "FFT_Sq": "green", "FreqControl1": "orange"}
method_shapes_list = [".", "+", "^"]

audio_fs_from_dsf_dict = {df:af for df, af in zip(dflist, aflist)}
family_method_cols_dict = {"svm": 3, "knn": 3, "ffnn": 3}
method_display_names_dict = {"rbf_svm": "SVM RBF", 
                             "K5N": "KNN(5)",
#                             "K10N": "KNN(10)",
#                             "K15N": "KNN(15)",
                             "MLPClass1": "MLP A",
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
    for downsample in dflist:
        data_dict[classification][downsample] = {met: [] for met in metrics}
        # Select value for chosen classifier method and frequency method and window len
        for wlen in wllist:
          for met in metrics:
            data_dict[classification][downsample][met].append(
              plot_frame[(plot_frame["downsample_factor"] == downsample) 
              & (plot_frame["window_duration"] == wlen)][met].values[0]
            )
        print(f"Downsampling: {downsample}> For {classification}, with {freq}:")
        print("Mean F1  : " + str(data_dict[classification][downsample]["mean_score"]))
        print("Std. dev : " + str(data_dict[classification][downsample]["std_dev"]))
        # Plot trace
        
        # Plot classification with different shape for different hyperparameters
        # Use diferent color for different downsampling 
        axe.plot(wllist, data_dict[classification][downsample]["mean_score"], 
                 #c=freq_colors_dict[freq], 
                 linestyle='--', linewidth=plot_width)
        axe.scatter(wllist, data_dict[classification][downsample]["mean_score"], 
                 label=f"{downsample}X downsample",
                 #c=freq_colors_dict[freq], 
                 #marker=method_shapes_list[idx],
                 s=78)
        axe.set_title(f"{classification} with {freq}")
        axe.set_ylim([0, 1])
        axe.legend(ncol=1 , loc="lower right")
        axe.set_xlabel("Window Len (s)")
        axe.set_ylabel("F1 Score, out of 1.00")
        plt.xticks(wllist)
        axe.grid(True, which="minor", axis='both')
        axe.minorticks_on()
        axe.tick_params(which="minor", bottom=False, left=False)
        axe.grid(True, which="major", axis='both', linewidth=2, color='k')
        axe.set_axisbelow(True)
 
plt.show(block=False)
input("Press enter to close.")

# Use win_lens as x axis




