import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import numpy as np
import pdb


#load in all data and put into one dataframe
#path to data files
path = r'C:\Users\hardi\OneDrive\Documents\GitRepos\concrete_tool_wear\data' 
files = glob.glob(os.path.join(path + "/*.csv"))

df = pd.DataFrame()
for f in files:
    csv = pd.read_csv(f)
    df = df.append(csv)


value = 'mean_score'
df2 = (df.groupby(['application','window_duration'], as_index=False)[value].max())

new_frame = [d for _, d in df2.groupby(['application'])]

print(new_frame)

x_var1 = new_frame[0]["window_duration"]
y_var1 = new_frame[0]["mean_score"]
x1 = list(x_var1)
y1 = list(y_var1)
#error1 = new_frame[0]['std_dev']
#yerr1 = list(error1)

x_var2 = new_frame[1]["window_duration"]
y_var2 = new_frame[1]["mean_score"]
x2 = list(x_var2)
y2 = list(y_var2)


x_var3 = new_frame[2]["window_duration"]
y_var3 = new_frame[2]["mean_score"]
x3 = list(x_var3)
y3 = list(y_var3)


x_var4 = new_frame[3]["window_duration"]
y_var4 = new_frame[3]["mean_score"]
x4 = list(x_var4)
y4 = list(y_var4)





yfont = {'family': 'Arial','color':  'k','weight': 'normal','size': 15,}
xfont= {'family':'Arial', 'color':'k','weight':'normal','size': 15,}
labelfont = {'family': 'Arial','color':  'k','weight': 'bold','size': 15,}

fig = plt.figure(figsize=(10, 6))

fig.text(0.5, 0.02,'window duration',fontdict=xfont, ha='center', va='center')
fig.text(0.06, 0.5, 'mean score',fontdict=yfont, ha='center', va='center', rotation='vertical')
fig.text(0.5, 0.96,'Performance vs. Window Width',fontdict=labelfont, ha='center',va='center')

ax = plt.subplot(2,2,1)
ax.bar(x1,y1,width=0.02,color ='b')
ax.set_title("cap mat")
ax.set_ylim([0, 1])

ax = plt.subplot(2,2,2)
ax.bar(x2,y2,width=0.02,color ='r')
ax.set_title("cap wear")
ax.set_ylim([0, 1])

ax = plt.subplot(2,2,3)
ax.bar(x3,y3,width=0.02,color ='k')
ax.set_title("sg mat", y=-0.22)
ax.set_ylim([0, 1])

ax = plt.subplot(2,2,4)
ax.bar(x4,y4,width=0.02,color ='g')
ax.set_title("sg wear", y=-0.22)
ax.set_ylim([0, 1])




plt.show(block=True)
