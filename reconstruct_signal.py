import matplotlib.pyplot as plt
from scipy import signal

import numpy as np


start_scale = 2

sampling_freq = 1024
number_of_samples = 200
number_of_channels = 5

num_of_sig = 3

fig, ax = plt.subplots(num_of_sig, number_of_channels)

for i, row in enumerate(ax):

    npzfile = np.load('Output/RandomSamples/PSU_Data_200ms_part1/gen_start_scale=' + str(start_scale) + '/' + str(i) + '.npz')
    key = sorted(npzfile.files)[0]
    spectral_array = npzfile[key].transpose(2, 0, 1)

    for j, col in enumerate(row):
        img_array = np.array(spectral_array[j])

        t, xrec = signal.istft(img_array, sampling_freq,
                               nperseg=int(number_of_samples / 1),
                               noverlap=int(number_of_samples / 1) - 1
                               )

        col.plot(t, xrec)

plt.savefig('Output/GeneratedSignals/reconstructed_signals.png', dpi=300)

