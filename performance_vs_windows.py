import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import numpy as np
import pdb


#load in all data and put into one dataframe
#path to data files
path = './data'
files = glob.glob(os.path.join(path + "/*.csv"))

df = pd.DataFrame()
for f in files:
    csv = pd.read_csv(f)
    df = df.append(csv)

applications = df.groupby('application')
app_data = {}
for app_name, app_group in applications:
  windows_groups = app_group.groupby('window_duration')
  app_data[app_name] = {}
  for win_name, win_group in windows_groups:
    best_row = win_group[win_group.mean_score == win_group.mean_score.max()]
    app_data[app_name][win_name] = (float(best_row.iloc[0]["mean_score"]),
                                    float(best_row.iloc[0]["std_dev"]))

yfont = {'family': 'Sans','color':  'k','weight': 'normal','size': 15,}
xfont= {'family':'Sans', 'color':'k','weight':'normal','size': 15,}
labelfont = {'family': 'Sans','color':  'k','weight': 'bold','size': 15,}

#fig = plt.figure(figsize=(10, 6))

#fig.text(0.5, 0.02,'window duration',fontdict=xfont, ha='center', va='center')
#fig.text(0.06, 0.5, 'mean score',fontdict=yfont, ha='center', va='center', rotation='vertical')


applications = ["cap mat.", "cap wear", "sg mat.", "sg wear"]

fig, ax = plt.subplots(2,2, sharex='col')

for idx, app in enumerate(applications):
  ax = plt.subplot(2,2, idx+1)
  exes = list(app_data[app].keys())
  wyes = [app_data[app][x][0] for x in exes]
  errs = [app_data[app][x][1]/2.0 for x in exes] # plot is +/- this value
  ax.bar(x=exes, height=wyes, yerr=errs, width=0.02, capsize=3, ecolor='gray')
  ax.set_title(app)
  ax.set_ylim([0,1]) 
  ax.set_ylabel("F1 Score with 1 SD error bars")
  if idx >= 2:
    ax.set_xlabel("Window duration")

fig.text(0.5, 0.96,'Performance vs. Window Width for Applications',fontdict=labelfont, ha='center',va='center')

# check signifigance of performance for each app
print("Computing Z scores for each app...")
for app in applications:
  exes = list(app_data[app].keys())
  wyes = [app_data[app][x][0] for x in exes]
  errs = [app_data[app][x][1] for x in exes]
  z_scores = {}
  for idx, x in enumerate(exes[:-1]):
    for idx2, x2 in enumerate(exes[idx+1:]):
      z_scores[(x,x2)] = (wyes[idx]-wyes[idx2+idx+1]) / np.sqrt(errs[idx]*errs[idx]/100.0 +
                                  errs[idx2+idx+1]*errs[idx2+idx+1]/100.0)
  print(app)
  [print(item) for item in z_scores.items()]

plt.show(block=True)
