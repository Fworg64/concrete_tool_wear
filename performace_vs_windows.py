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

#print (df)

pd.set_option('display.width', 500)
#groups by application and window size then sorts through that data set to find
#the max mean_score

value = 'mean_score'
df2 = (df.groupby(['application','window_duration'], as_index=False)[value].max())

df_app = [d for _, d in df2.groupby(['application'])]

print(df_app)

#take that data and create 4 bar graphs with them
#New_Colors = ['green','blue','purple','brown','teal']
#plt.bar(color=New_Colors)


# x is window size

#y is score





#have 4 plots on one window, for each application



#label titles with the approperate number application 1-4

#label x axis

#label y axis

#label x tick markes. the window sizes




#grab the parameters for the winners in each plot
#parameters as the normalization procees and FFT



plt.show(block=True)
