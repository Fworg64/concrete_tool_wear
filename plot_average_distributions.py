## Python script to plot average distributions after loading files

import pdb

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from loader import load_audio_files
from windowizer import Windowizer, window_maker
from custom_pipeline_elements import FFTMag

# Audio sample rate of recordings
audio_fs = 44100

# Hyper-parameters
window_shape = "hamming"
window_duration = 0.2
window_overlap = 0.5

# Load data
downsample_factor = 16
raw_audio_data, metadata = load_audio_files(
  "./raw_audio/classifications.txt", integer_downsample=downsample_factor)

# Adjust sampling from downsampling
audio_fs = int(audio_fs / downsample_factor)

# Apply windowing
window_len = int(window_duration*audio_fs)
audio_window = Windowizer(window_maker(window_shape, int(window_duration*audio_fs)), window_overlap)
windowed_audio_data, windowed_audio_labels = audio_window.windowize(raw_audio_data, metadata)
      
# Define classes
wear_classes2ints = {"New":0, "Moderate":1, "Worn":2}
wear_ints2classes = {v: k for k,v in wear_classes2ints.items()}

# Process windowed data using FFT
fft_windowed_audio_data = []
fft_transform = FFTMag()
for sample in windowed_audio_data:
  fft_windowed_audio_data.append(fft_transform.transform(sample))

# Sort data by wear level
time_domain_data_dict = {"New": [], "Moderate": [], "Worn": []}
freq_domain_data_dict = {"New": [], "Moderate": [], "Worn": []}
for (td, fd, cat) in zip(windowed_audio_data, fft_windowed_audio_data, windowed_audio_labels):
  time_domain_data_dict[cat].append(td)
  freq_domain_data_dict[cat].append(fd)

# Find average and variance of each coefficient for each wear level
new_td = np.array(time_domain_data_dict["New"])
mod_td = np.array(time_domain_data_dict["Moderate"])
worn_td = np.array(time_domain_data_dict["Worn"])
new_fd = np.array(freq_domain_data_dict["New"])
mod_fd = np.array(freq_domain_data_dict["Moderate"])
worn_fd = np.array(freq_domain_data_dict["Worn"])

names = ["New Time Domain [16x downsample]",
         "Mod. Time Domain [16x downsample]",
	 "Worn Time Domain [16x downsample]",
	 "New Freq. Domain [16x downsample]",
	 "Mod. Freq. Domain [16x downsample]",
	 "Worn Freq. Domain [16x downsample]"]
avgs = {}
devs = {}
for val, name in zip([new_td, mod_td, worn_td, new_fd, mod_fd, worn_fd], names):
  avgs[name] = np.mean(val, axis=0)
  devs[name] = np.std(val, axis=0)

# Plot distribution vectors for each wear level for frequency
exes = list(range(len(avgs[names[0]])))
freqs = list(range(len(avgs[names[3]])))
freq_vals = np.linspace(0, audio_fs/2, len(avgs[names[3]]))
time_vals = np.linspace(0, window_duration, len(avgs[names[0]]))

# Set figure sizes
fontsize = 22
plt.rc('font', size=fontsize, family='sans')
plt.rc('axes', titlesize=fontsize)
plt.rc('axes', labelsize=fontsize)
plt.rc('legend', fontsize=fontsize)
plot_width = 3.6 # pt

fig, axs = plt.subplots(3,2)

for index in range(3):
  # Plot time domain
  axs[index][0].fill_between(time_vals, 
    avgs[names[index]] + devs[names[index]],
    avgs[names[index]] - devs[names[index]],
    alpha=0.5, color='tab:purple')
  axs[index][0].plot(time_vals, avgs[names[index]], 
	color='tab:green', linewidth=plot_width)
  axs[index][0].set_title(names[index])
  axs[index][0].set_ylim(-6000, 6000)
  axs[index][0].legend(["Mean", r"$\pm$ 1 Std. Dev."], loc="upper right", prop = {"size":18})
  axs[index][0].set_xlabel("Time (s)")
  axs[index][0].set_ylabel("Amplitude")
  axs[index][0].ticklabel_format(axis='y', style='sci', scilimits=(3,3))

  # Plot freq domain
  axs[index][1].fill_between(freq_vals,
    avgs[names[index+3]] + devs[names[index+3]],
    avgs[names[index+3]] - devs[names[index+3]],
    alpha=0.5, color='tab:purple')
  axs[index][1].fill_between([0, 200], 0, 500000, color='tab:red', alpha=0.2)
  axs[index][1].fill_between([400, 600], 0, 500000, color='tab:red', alpha=0.2)
  axs[index][1].fill_between([800, 1000], 0, 500000, color='tab:red', alpha=0.2)
  axs[index][1].fill_between([1200, 1400], 0, 500000, color='tab:red', alpha=0.2)
  #axs[index][1].fill_between([1200, 1400], 0, 500000, color='tab:red', alpha=0.2)
  axs[index][1].plot(freq_vals, avgs[names[index+3]], 
	color='tab:green', linewidth=plot_width)
  axs[index][1].set_title(names[index+3])
  axs[index][1].set_ylim(0, 3.5e5)
  axs[index][1].legend(["Mean", r"$\pm$ 1 Std. Dev."], loc="upper right", prop = {"size":18})
  axs[index][1].set_xlabel("Freq. (Hz)")
  axs[index][1].set_ylabel("Spectra Magnitude")
  axs[index][1].ticklabel_format(axis='y', style='sci', scilimits=(3,3))


plt.show(block=False)
input("Press Enter to close...")

# Plot distribution vectors for each wear level for time domain


