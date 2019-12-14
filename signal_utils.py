import numpy as np
from scipy import signal
from PIL import Image
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt


def get_channel_array(opt):
    multi_channel_spectral = []

    # parse signals
    data = pd.read_csv('%s%s' % (opt.input_dir, opt.input_name), nrows=opt.row+1)
    x = data.values[opt.row, 1:]
    channels = np.split(x, opt.num_of_channels)

    class_label = data.values[opt.row, 0]
    print("***This signal is of class: " + str(int(class_label)))

    # for each temporal channel
    for i, ch in enumerate(channels):

        # calculate spectrogram image
        signal_length = len(ch)
        _, _, Zxx = signal.stft(ch, opt.samp_freq,
                                nperseg=int(signal_length / 1),
                                noverlap=int(signal_length / 1) - 1
                                )

        # generate spectrogram image
        min_intensity = np.min(Zxx.real)
        max_intensity = np.max(Zxx.real)

        img_array = ((Zxx.real / max_intensity) * 255)

        img = Image.fromarray(img_array).convert('L')
        img_array = np.array(img)
        multi_channel_spectral.append(img_array)

    multi_channel_spectral = np.array(multi_channel_spectral)  # convert to np array
    multi_channel_spectral = multi_channel_spectral.transpose((1, 2, 0))  # convert to NHWC for conformity

    return multi_channel_spectral


def reconstruct_signals(opt, dir2save):
    """
    This function creates a csv file that contains the reconstructed temporal signals that were generated with SinGAN
    """
    row_index = 37
    start_scale = 0

    directory = ('Output/RandomSamples/PSU_Data_200ms_part1_row' + str(row_index) +
                 '/gen_start_scale=' + str(start_scale) +
                 '/')

    signals = []

    for filename in os.listdir(dir2save):
        if filename.endswith(".npz"):

            npzfile = np.load(os.path.join(directory, filename))
            key = sorted(npzfile.files)[0]
            spectral_array = npzfile[key].transpose(2, 0, 1)

            s = np.array([])

            for j in range(opt.num_of_channels):
                img_array = np.array(spectral_array[j])

                _, xrec = signal.istft(img_array, opt.samp_freq,
                                       nperseg=int(opt.num_samples / 1),
                                       noverlap=int(opt.num_samples / 1) - 1
                                       )

                s = np.append(s, xrec)

            signals.append(s)

    with open((dir2save + '/signals.csv'), 'w', newline='') as csv_file:
        wr = csv.writer(csv_file, delimiter=',')
        for sig in signals:
            wr.writerow(sig)


def plot_signals(opt, path):
    # read in signals
    signals = np.genfromtxt(path + '/signals.csv', delimiter=',')

    # plot figures
    for i in range(signals.shape[0]):
        channels = np.split(signals[i], opt.num_of_channels)
        fig, ax = plt.subplots(1, opt.num_of_channels)

        for j, col in enumerate(ax):
            x = list(range(len(channels[j])))
            y = channels[j]
            col.plot(x, y)

        plt.savefig(path + '/reconstructed_signals_' + str(i) + '.png', dpi=300)
        plt.close()

