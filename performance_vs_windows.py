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
#fig.text(0.5, 0.96,'Performance vs. Window Width',fontdict=labelfont, ha='center',va='center')

applications = ["cap mat.", "cap wear", "sg mat.", "sg wear"]

fig, ax = plt.subplots(2,2, sharex='col')

for idx, app in enumerate(applications):
  ax = plt.subplot(2,2, idx+1)
  exes = list(app_data[app].keys())
  wyes = [app_data[app][x][0] for x in exes]
  errs = [app_data[app][x][1] for x in exes]
  ax.bar(x=exes, height=wyes, yerr=errs, width=0.02, capsize=3, ecolor='gray')
  ax.set_title(app)
  ax.set_ylim([0,1]) 
  ax.set_ylabel("F1 Score")
  if idx >= 2:
    ax.set_xlabel("Window duration")

#ax = plt.subplot(2,2,1)
#ax.bar(x1,y1,width=0.02,color ='b')
#ax.set_title("cap mat")
#ax.set_ylim([0, 1])

#ax = plt.subplot(2,2,2)
#ax.bar(x2,y2,width=0.02,color ='r')
#ax.set_title("cap wear")
#ax.set_ylim([0, 1])

#ax = plt.subplot(2,2,3)
#ax.bar(x3,y3,width=0.02,color ='k')
#ax.set_title("sg mat", y=-0.22)
#ax.set_ylim([0, 1])

#ax = plt.subplot(2,2,4)
#ax.bar(x4,y4,width=0.02,color ='g')
#ax.set_title("sg wear", y=-0.22)
#ax.set_ylim([0, 1])




plt.show(block=True)
