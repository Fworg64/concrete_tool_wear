# This script plots the results from the experiment concluded
# on Jan 7 2023

import glob
import pandas as pd
import matplotlib.pyplot as plt

import pdb

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

dflist = downsample_factors.tolist()
dflist.sort(reverse=True)

cmlist = classification_methods.tolist()
cmlist.sort(reverse=True)

fmlist = frequency_methods.tolist()
fmlist.sort()

wllist = win_lens.tolist()
wllist.sort()

print("downsamplings: ")
print(dflist)
print("classification: ")
print(cmlist)
print("freq trans : ")
print(fmlist)
print("window len: ")
print(wllist)


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
    #plot_frame.plot()
    data_dict = {}
    # Organize by classifier in family
    for classifier in classification_group_members_dict[family_method]["names"]:
      classifier_frame = plot_frame.loc[plot_frame["classifier"] == classifier]
      # Pull out each frequency method
      data_dict[classifier] = {}
      for freq in fmlist:
        data_dict[classifier][freq] = []
        # Select value for chosen classifier method and frequency method and window len
        for wlen in wllist:
          data_dict[classifier][freq].append(
            classifier_frame[(classifier_frame["freq1"] == freq) 
            & (classifier_frame["window_duration"] == wlen)]
          )
        print(f"Downsampling: {downsample}> For {family_method}: {classifier}, with {freq}:")
        print(data_dict[classifier][freq])
# Plot classification with different shape for different hyperparameters
# Use diferent color for frequncy method
# Use win_lens as x axis
plt.show()




