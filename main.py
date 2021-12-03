import numpy as np
import time

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from loader import load_audio_files_from_dir

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec

import scipy
from scipy.signal.windows import hamming

import pdb

print("Loading Audio from files...")
# Load audio from audio files classifications.txt
# filename, wear, spacing, start, end, notes

widths = [0.05, 0.1, 0.2, 0.4, 0.8]
scores = []
test_cmats = []

for sample_width_s in widths:
    #overlap_frac = 0.5
    data_vectors_pre, classifications = load_audio_files_from_dir("./raw_audio/classifications.txt", sample_width_s)#, overlap_frac)
    #get lenght of a data vector. Store in variable
    # use that lenght to gerate a hamming window
    #convert hamming window / data vector to same type (numpy v.s. list)
    # elment wise multiple each data vector by the hamming window
    #make sure data vectors is a list of lists

    # Hyperparameters: Window width (sec), Audio prefiltering (?)
    m = len(data_vectors_pre[0])
    my_window = hamming(m)
    print(my_window)

    data_vectors = (my_window * data_vectors_pre[:])
    data_vectors = data_vectors.tolist()
    dt = sample_width_s / m
    sample_time = np.arange(0, sample_width_s, dt)
    fig, (ax0, ax1, ax2) = plt.subplots(3,1)
    ax0.plot(sample_time, data_vectors_pre[0])
    ax1.plot(sample_time, my_window)
    ax2.plot(sample_time, data_vectors[0])

    ax0.set_title('Raw Audio')
    ax0.set_ylabel('Decibels')
    ax0.set_xlabel('Time (s)')

    ax1.set_title("Hamming Window Curve")
    ax1.set_ylabel('Amplitube')
    ax1.set_xlabel('Time (s)')

    ax2.set_title('Audio With Window')
    ax2.set_ylabel('Decibels')
    ax2.set_xlabel('Time (s)')

    plt.show(block=False)

    classes2ints = {"New":0, "Moderate":1, "Worn":2}
    integer_classes = [classes2ints[classi] for classi in classifications]
    # print("Assigning classes to integers as" + str(classes2ints))

    # print(str(len(data_vectors)) + " Samples")
    #print(classifications)

    # Process chunks (FFT)
    # Use np real input FFT for fast computation
    data_procd = [np.abs(np.fft.rfft(data_vector)) for data_vector in data_vectors];

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
    # with np.printoptions(precision=3, suppress=True):
    #     print("Training data confusion matrix: \n")
    #     print(train_cmat)
    #     print('\n')
    #     print("Test data confusion matrix: \n")
    #     print(test_cmat)
    #     print(clf_score)

max_score_index = np.argmax(scores)
#tells us where/which part of the list gives us the most accurate/largest f1_score
print("Best score, best width, Testing confusion matrix")
print(scores[max_score_index])
print(widths[max_score_index])
print(test_cmats[max_score_index])

input("Press enter to close.")
plt.close(fig)
