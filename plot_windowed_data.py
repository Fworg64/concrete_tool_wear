import numpy as np
import time

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from loader import load_audio_files_from_dir, load_audio_files
from windowizer import Windowizer, window_maker
from custom_pipeline_elements import FFTMag

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec

import scipy
from scipy.signal.windows import hamming

import pdb

print("Loading Audio from files...")
# Load audio from audio files classifications.txt
# filename, wear, spacing, start, end, notes

widths = [0.2] #[0.05, 0.1, 0.2, 0.4, 0.8]
scores = []
test_cmats = []

test_new_loader = True

for sample_width_s in widths:
    if test_new_loader:
      audio_fs = 44100
      downsample_factor = 1
      raw_audio_data, metadata = load_audio_files(
        "./raw_audio/classifications.txt", integer_downsample=downsample_factor)
      audio_fs = int(audio_fs/downsample_factor)

      # Apply windowing
      window_duration = 0.2 # seconds
      window_overlap  = 0.5 # ratio of overlap [0,1)
      window_shape    = "hamming" #"boxcar" # from scipy.signal.windows
      audio_window = Windowizer(window_maker(window_shape, int(window_duration*audio_fs)), window_overlap)
      windowed_audio_data, windowed_audio_labels = audio_window.windowize(raw_audio_data, metadata)
      # Process chunks
      fft_proc = FFTMag(1)
      pdb.set_trace()
      data_procd = fft_proc.transform(windowed_audio_data)
      data_vectors = windowed_audio_data
      sample_time = np.arange(0, sample_width_s, 1./audio_fs)
      sample_freq = np.fft.rfftfreq(len(windowed_audio_data[0]), 1./audio_fs)
      classifications = windowed_audio_labels
    else:
      data_vectors_pre, classifications = load_audio_files_from_dir("./raw_audio/classifications.txt", sample_width_s)#, overlap_frac)
    
      # Hyperparameters: Window width (sec), Audio prefiltering (?)
      m = len(data_vectors_pre[0])
      my_window = hamming(m)
      print(my_window)

      data_vectors = (my_window * data_vectors_pre[:])
      data_vectors = data_vectors.tolist()
      dt = sample_width_s / m
      sample_time = np.arange(0, sample_width_s, dt)
      sample_freq = np.fft.rfftfreq(m, dt)

      # Process chunks (FFT)
      # Use np real input FFT for fast computation
      data_procd = [np.abs(np.fft.rfft(data_vector)) for data_vector in data_vectors];
      data_procd_unwindowed = [np.abs(np.fft.rfft(data_vector)) for data_vector in data_vectors_pre];

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5,1)
    if not test_new_loader: 
      ax0.plot(sample_time, data_vectors_pre[0])
      ax1.plot(sample_time, my_window)
      ax3.plot(sample_freq, data_procd_unwindowed[0])

    ax2.plot(sample_time, data_vectors[0])
    ax4.plot(sample_freq, data_procd[0])

    ax0.set_title('Raw Audio')
    ax0.set_ylabel('Decibels')
    ax0.set_xlabel('Time (s)')

    ax1.set_title("Hamming Window Curve")
    ax1.set_ylabel('Amplitude')
    ax1.set_xlabel('Time (s)')

    ax2.set_title('Audio With Window')
    ax2.set_ylabel('Decibels')
    ax2.set_xlabel('Time (s)')

    ax3.set_title('FFT of un-windowed data')
    ax3.set_ylabel('Spectra Magnitude')
    ax3.set_xlabel('Frequency')

    ax4.set_title('FFT of windowed data')
    ax4.set_ylabel('Spectra Magnitude')
    ax4.set_xlabel('Frequency')

    plt.show(block=False)

    #continue # skip classification

    classes2ints = {"New":0, "Moderate":1, "Worn":2}
    integer_classes = [classes2ints[classi] for classi in classifications]
    # print("Assigning classes to integers as" + str(classes2ints))

    # print(str(len(data_vectors)) + " Samples")
    #print(classifications)

    # Sort into training and testing/validation (K-fold?)
    x_train, x_test, y_train, y_test = train_test_split(data_procd, integer_classes,
                                                   test_size=0.25, random_state=24000)

    # Print data statistics
    vals, counts = np.unique(y_train, return_index=False,
                             return_inverse=False, return_counts=True)
    # print(f"train data has vals: {vals} with counts: {counts}")
    vals, counts = np.unique(y_test, return_index=False,
                             return_inverse=False, return_counts=True)
    # print(f"test data has vals: {vals} with counts: {counts}")
    #pdb.set_trace()
    # Train SVM
    clf = svm.SVC()
    tic = time.perf_counter()
    clf.fit(x_train, y_train)
    toc = time.perf_counter()
    print(f"Fit the svm in {toc - tic:0.4f} seconds")
    print(f"Found {clf.classes_} for classes")

    # Display accuracy, scores, etc
    y_train_pred = clf.predict(x_train)
    y_test_pred  = clf.predict(x_test)
    train_cmat = confusion_matrix(y_test, y_test_pred, normalize="true")
    test_cmat  = confusion_matrix(y_train, y_train_pred, normalize="true")
    clf_score = f1_score(y_test, y_test_pred, average='macro')
    scores.append(clf_score)
    test_cmats.append(test_cmat)
    with np.printoptions(precision=3, suppress=True):
        print("Training data confusion matrix: \n")
        print(train_cmat)
        print('\n')
        print("Test data confusion matrix: \n")
        print(test_cmat)
        print(clf_score)

#tells us where/which part of the list gives us the most accurate/largest f1_score
#max_score_index = np.argmax(scores)
#print("Best score, best width, Testing confusion matrix")
#print(scores[max_score_index])
#print(widths[max_score_index])
#print(test_cmats[max_score_index])

input("Press enter to close.")
plt.close(fig)
