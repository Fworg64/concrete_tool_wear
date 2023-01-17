# This script plots the results from the experiment concluded
# on Jan 7 2023

import glob
import pandas as pd
import matplotlib.pyplot as plt

import pdb

# Set figures
fontsize = 20
plt.rc('font', size=fontsize, family='sans')
plt.rc('axes', titlesize=fontsize)
plt.rc('axes', labelsize=fontsize)
plt.rc('legend', fontsize=fontsize)

# load all data in folder

results_path = './server_results_jan7_2023/'

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

freq_colors_dict = {"FFT_Mag":"red", "FFT_Rt":"blue", "FFT_Sq": "green", "FreqControl1": "orange"}
method_shapes_list = [".", "+", "^"]

audio_fs_from_dsf_dict = {df:af for df, af in zip(dflist, aflist)}
family_method_cols_dict = {"svm": 2, "knn": 3, "ffnn": 3}
method_display_names_dict = {"rbf_svm": "SVM RBF", 
                             "K5N": "KNN(5)",
                             "K10N": "KNN(10)",
                             "K15N": "KNN(15)",
                             "MLPClass1": "MLP A",
                             "MLPClass2": "MLP B",
                             "MLPClass3": "MLP C"}
family_display_names_dict = {"svm": "SVM", "knn": "KNN", "ffnn": "MLP"}


for downsample in dflist:
  classification_group_members_dict = {
    "svm":{"names":["rbf_svm"], "data":[]}, 
    "knn":{"names":['K5N', 'K10N', 'K15N'], "data":[]},
   "ffnn":{"names":['MLPClass1', 'MLPClass2', 'MLPClass3'], "data":[]}
  }
  for classification in cmlist:
    # Sort classification groups
    for key in classification_group_members_dict.keys():
      if classification in classification_group_members_dict[key]["names"]:
        classification_group_members_dict[key]["data"].append(
          mega_frame.loc[(mega_frame['downsample_factor'] == downsample) 
          & (mega_frame['classifier'] == classification)]
        )


  # Make figure for each group
  for family_method in classification_group_members_dict.keys():
    plot_frame = pd.concat(classification_group_members_dict[family_method]["data"])
    print(plot_frame.columns)
    plot_frame = plot_frame[["mean_score", "std_dev", "acc", "acc_dev", 
                             "window_duration", "classifier", "freq1"]]

    # Make figure
    fig, axe = plt.subplots()

    data_dict = {}
    # Organize by classifier in family
    for idx, classifier in enumerate(classification_group_members_dict[family_method]["names"]):
      classifier_frame = plot_frame.loc[plot_frame["classifier"] == classifier]
      # Pull out each frequency method
      data_dict[classifier] = {}
      metrics = ["mean_score", "std_dev", "acc", "acc_dev"]
      for freq in fmlist:
        data_dict[classifier][freq] = {met: [] for met in metrics}
        # Select value for chosen classifier method and frequency method and window len
        for wlen in wllist:
          for met in metrics:
            data_dict[classifier][freq][met].append(
              classifier_frame[(classifier_frame["freq1"] == freq) 
              & (classifier_frame["window_duration"] == wlen)][met].values[0]
            )
        print(f"Downsampling: {downsample}> For {family_method}: {classifier}, with {freq}:")
        print("Mean F1  : " + str(data_dict[classifier][freq]["mean_score"]))
        print("Std. dev : " + str(data_dict[classifier][freq]["std_dev"]))
        # Plot trace
        
        # Plot classification with different shape for different hyperparameters
        # Use diferent color for frequncy method
        axe.plot(wllist, data_dict[classifier][freq]["mean_score"], 
                 c=freq_colors_dict[freq], 
                 linestyle='--')
        axe.scatter(wllist, data_dict[classifier][freq]["mean_score"], 
                 label=f"{method_display_names_dict[classifier]} with {freq}",
                 c=freq_colors_dict[freq], 
                 marker=method_shapes_list[idx],
                 s=78)
        axe.set_title(f"{family_display_names_dict[family_method]} with Fs = {audio_fs_from_dsf_dict[downsample]} Hz [{downsample}X downsample]")
        axe.set_ylim([0, 1])
        axe.legend(ncol=family_method_cols_dict[family_method] , loc="lower right")
        axe.set_xlabel("Window Len (s)")
        axe.set_ylabel("F1 Score, out of 1.00")
        plt.grid(True)
 
plt.show(block=False)
input("Press enter to close.")

# Use win_lens as x axis




