## Python script to plot average distributions after loading files

import pdb

import numpy as np
import matplotlib.pyplot as plt

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

names = ["New TD", "Mod. TD", "Worn TD", "New FD", "Mod. FD", "Worn FD"]
avgs = {}
devs = {}
for val, name in zip([new_td, mod_td, worn_td, new_fd, mod_fd, worn_fd], names):
  avgs[name] = np.mean(val, axis=0)
  devs[name] = np.std(val, axis=0)

# Plot distribution vectors for each wear level for frequency
exes = list(range(len(avgs[names[0]])))
freqs = list(range(len(avgs[names[3]])))

#pdb.set_trace()

fig, axs = plt.subplots(3,2)
axs[0][0].plot(avgs[names[0]])
axs[1][0].plot(avgs[names[1]])
axs[2][0].plot(avgs[names[2]])

axs[0][1].plot(freqs, avgs[names[3]])
axs[0][1].fill_between(freqs, avgs[names[3]]+devs[names[3]], avgs[names[3]]-devs[names[3]], alpha=0.5)

axs[1][1].plot(freqs, avgs[names[4]])
axs[1][1].fill_between(freqs, avgs[names[4]]+devs[names[4]], avgs[names[4]]-devs[names[4]], alpha=0.5)

axs[2][1].plot(freqs, avgs[names[5]])
axs[2][1].fill_between(freqs, avgs[names[5]]+devs[names[5]], avgs[names[5]]-devs[names[5]], alpha=0.5)
plt.show(block=False)

input("Press Enter to close...")

# Plot distribution vectors for each wear level for time domain


