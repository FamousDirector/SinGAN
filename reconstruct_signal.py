from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal

import numpy as np
import json

with open('Input/Spectrograms/data.json') as f:
    signal_info = json.load(f)

sampling_freq = signal_info['sampling_freq']

signal_num = 3
img = Image.open('Output/RandomSamples/spectrogram/gen_start_scale=0/' + str(signal_num) + '.png').convert('F')
img_array = np.array(img)

img_array = (img_array / 255.0) * signal_info['max_intensity']
t, xrec = signal.istft(img_array, sampling_freq,
                       nperseg=int(sampling_freq / 4),
                       noverlap=int(sampling_freq / 4) - 2
                       )

plt.plot(t, xrec)
plt.savefig('Input/Spectrograms/reconstructed_signal_' + str(signal_num) + '.png')
