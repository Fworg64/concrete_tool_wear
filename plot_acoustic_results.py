# Plot the acoustic results

import pandas as pd
import matplotlib.pyplot as plt

import pdb

# Path to experiment data
filename = "20220710_120343results.csv" # this is the important one
path_to_data = "./out/server/"

# load files into data frame
data_table = pd.read_csv(path_to_data + filename)

# Pull relevant data
methods = data_table["fft"].to_list()
f1_scores = data_table["mean_score"].to_list()
f1_std_dev = data_table["std_dev"].to_list()

# Set figure sizes
fontsize = 30
legendsize = 20
plt.rc('font', size=fontsize, family='sans')
plt.rc('axes', titlesize=fontsize)
plt.rc('axes', labelsize=fontsize)
plt.rc('legend', fontsize=legendsize)

# Make bar graph with error bars
ind = [1,2,3,4]
fig, ax = plt.subplots(1,1)
ax.bar(ind, f1_scores, yerr=f1_std_dev, 
       width=0.4, capsize=65, ecolor="#DF2094", error_kw={"elinewidth":2, "capthick":3},
       color="#2094DF")
ax.set_title("Support-Vector Machine F1 scores +/- 1 std. dev.")

ax.set_xticks(ind)
ax.set_xticklabels(methods)
ax.set_ylim(0,1)
ax.set_xlabel("Preprocessing Method")
ax.set_ylabel("F1 score, out of 1.00")

plt.show(block=False)
input("Press Enter to close...")
