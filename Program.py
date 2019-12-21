from sklearn import tree as tree
import numpy as np
import pandas as pd
import stft
import scipy.signal as sp
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift

table = pd.read_csv("tableofnotes.csv", sep=";")

big = [i for i in range(1, 12)]
little = [i for i in range(13, 24)]
first = [i for i in range(25, 36)]
second = [i for i in range(37, 48)]
third = [i for i in range(49, 60)]

amp = 2 * np.sqrt(2)

classifier = tree.DecisionTreeClassifier()
classifier.fit(np.array(table["freq"]).reshape(-1, 1), np.array(table["note"]))

(fs, file) = wav.read("twt.wav")
print(file)
print(file.shape)

(f, t, Zxx) = sp.stft(file, fs, nperseg=256, return_onesided=True)
SAMPLE_RATE = 44100
WINDOW_SIZE = 2048
WINDOW_STEP = 512

(spectrum, frequencies, s, im) = plt.specgram(Zxx, NFFT=WINDOW_SIZE, noverlap=WINDOW_SIZE - WINDOW_STEP, Fs=SAMPLE_RATE, scale_by_freq=False)
plt.show()
for f in frequencies:
    print(f)



