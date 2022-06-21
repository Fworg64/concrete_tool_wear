import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb

file_names =  ['20220618_150936results.csv',  '20220618_170458results.csv']

data_bank = pd.read_csv('./data/' + file_names[1])

norm1 = data_bank["mean_score"][::4]
norm2 = data_bank["mean_score"][1::4]
norm3 = data_bank["mean_score"][2::4]
norm4 = data_bank["mean_score"][3::4]

norm5 = data_bank["std_dev"][::4]
norm6 = data_bank["std_dev"][1::4]
norm7 = data_bank["std_dev"][2::4]
norm8 = data_bank["std_dev"][3::4]
plot_points = np.array([6.*float(x) + 10.*int(x/4) for x in range(len(norm1))])

#change window size of graph when displayed
width = 10
height = 5

fig1 = plt.figure(figsize =(width, height))

plt.bar(plot_points+2, norm1.tolist(),yerr= norm5.tolist(), capsize = 3, ecolor = 'gray')
plt.bar(plot_points+1, norm2.tolist(), yerr= norm6.tolist(),  capsize = 3, ecolor = 'gray')
plt.bar(plot_points+3, norm3.tolist(),yerr= norm7.tolist(),  capsize = 3, ecolor = 'gray')
plt.bar(plot_points, norm4.tolist(),yerr= norm8.tolist(),  capsize = 3, ecolor = 'gray')



#labels the y axis
yname = "mean_score"
yfont = {'family': 'Arial','color':  'k','weight': 'normal','size': 15,}
plt.ylabel(yname, fontdict=yfont, labelpad=15)
#the title
titlename = "Normalizing and Frequency Data Results"
labelfont = {'family': 'Arial','color':  'k','weight': 'bold','size': 15,}
plt.title(titlename,fontdict=labelfont, pad=25)
#labels the x axis
xname = "applications"
xfont= {'family':'Arial', 'color':'k','weight':'normal','size': 12,}
plt.xlabel(xname, fontdict=xfont, labelpad=10)

#group names of applications
plt.tick_params(bottom=False, top=True, left=True, right=False)
plt.tick_params(labelbottom=False, labeltop=True, labelleft=True, labelright=False)
application=['for','for','for','for']
x = 10
pos = [x,(x +35),(x+35+35),(x+35+35+35)]
plt.xticks(pos, application)

#legend of the colors of the bars
colors = { 'after':'green','both':'blue','before':'orange','none':'red'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
locationlegend = 'upper left'
#lables the legend
titlelegend =  'FFT applied'
plt.legend(handles, labels,bbox_to_anchor=(1.02, 1), loc= locationlegend, borderaxespad=0,title =titlelegend)




plt.show(block=True)

