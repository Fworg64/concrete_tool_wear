import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb

file_names =  ['20220618_150936results.csv',  '20220618_170458results.csv']

data_bank = pd.read_csv('./data/' + file_names[1])

pdb.set_trace()

norm1 = data_bank["mean_score"][::4]
norm2 = data_bank["mean_score"][1::4]
norm3 = data_bank["mean_score"][2::4]
norm4 = data_bank["mean_score"][3::4]

plot_points = np.array([6.*float(x) + 10.*int(x/4) for x in range(len(norm1))])


fig1 = plt.figure()

plt.bar(plot_points, norm1.tolist())
plt.bar(plot_points+1, norm2.tolist())
plt.bar(plot_points+2, norm3.tolist())
plt.bar(plot_points+3, norm4.tolist())



plt.show(block=True)

