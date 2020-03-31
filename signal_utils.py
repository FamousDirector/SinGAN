import numpy as np
from scipy import signal
from PIL import Image
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
import pycwt
import SinGAN.functions as functions


def get_multi_channel_spectral_array(opt):
    multi_channel_spectral = []
    min_max_values = []
    dir2save = functions.generate_dir2save(opt)  # for saving signal values

    # parse signals
    data = pd.read_csv('%s%s' % (opt.input_dir, opt.input_name), nrows=opt.row+1, header=None)
    x = data.values[opt.row, opt.data_col_start_index:]
    channels = np.split(x, opt.num_of_channels)

    class_label = data.values[opt.row, 0]
    print("***This signal is of class: " + str(int(class_label)))

    # for each temporal channel
    for i, ch in enumerate(channels):

        if i % opt.channel_skip_count:
            continue

        min_max_values.append([np.min(ch), np.max(ch)])

        # calculate spectrogram image
        signal_length = len(ch)
        if opt.spectral_type == "stft":
            _, _, spectral = signal.stft(ch, opt.sample_freq,
                                         nperseg=int(signal_length / 1),
                                         noverlap=int(signal_length / 1) - 1
                                         )
        elif opt.spectral_type == "cwt":
            spectral, sj, _, _, _, _ = pycwt.cwt(ch, 1 / opt.sample_freq,
                                                 dj=1/32,
                                                 wavelet=pycwt.Morlet(100)
                                                 )
            np.savetxt('%s/sj_values.csv' % dir2save, sj, delimiter=',')
        else:
            raise Exception('Spectral type %s is not supported' % opt.spectral_type)

        # generate spectrogram image
        min_intensity = np.min(spectral.real)
        max_intensity = np.max(spectral.real)

        img_array = ((spectral.real - min_intensity) / (max_intensity - min_intensity) * 255)

        img = Image.fromarray(img_array).convert('L')
        img_array = np.array(img)
        multi_channel_spectral.append(img_array)

    multi_channel_spectral = np.array(multi_channel_spectral)  # convert to np array
    multi_channel_spectral = multi_channel_spectral.transpose((1, 2, 0))  # convert to NHWC for conformity

    np.savetxt('%s/min_max_values.csv' % dir2save,
               np.array(min_max_values), delimiter=',')  # save min/max values for signals

    return multi_channel_spectral


def reconstruct_signals(opt, dir2save):
    """
    This function creates a csv file that contains the reconstructed temporal signals that were generated with SinGAN
    """
    signals = []

    # load directory path of trained signal information
    if opt.mode == 'animation':
        opt.mode = 'animation_train'
        d = functions.generate_dir2save(opt)
        opt.mode = 'animation'
    else:
        opt.mode = 'train'
        d = functions.generate_dir2save(opt)
        opt.mode = 'random_samples'

    min_max_values = np.loadtxt('%s/min_max_values.csv' % d, delimiter=',')

    for filename in os.listdir(dir2save):
        if filename.endswith(".npz"):

            npzfile = np.load(os.path.join(dir2save, filename))
            key = sorted(npzfile.files)[0]
            spectral_array = npzfile[key].transpose(2, 0, 1)

            s = np.array([])

            for j in range(spectral_array.shape[0]):
                img_array = np.array(spectral_array[j])

                if opt.spectral_type == "stft":
                    _, xrec = signal.istft(img_array, opt.sample_freq,
                                           nperseg=int(opt.sample_length / 1),
                                           noverlap=int(opt.sample_length / 1) - 1
                                           )

                elif opt.spectral_type == "cwt":
                    sj = np.loadtxt('%s/sj_values.csv' % d, delimiter=',')
                    xrec = pycwt.icwt(img_array, sj, 1 / opt.sample_freq,
                                      dj=1/32,
                                      wavelet=pycwt.Morlet(100)
                                      )
                    xrec = xrec.real
                else:
                    raise Exception('Spectral type %s is not supported' % opt.spectral_type)

                x_norm = ((xrec - np.min(xrec)) / (np.max(xrec) - np.min(xrec)))
                x_renorm = (x_norm * (min_max_values[j][1] - min_max_values[j][0])) + min_max_values[j][0]
                s = np.append(s, x_renorm)

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

