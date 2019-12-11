import numpy as np
from scipy import signal
from PIL import Image
import pandas as pd


def get_channel_array(opt):
    multi_channel_spectral = []

    # parse signals
    data = pd.read_csv('%s%s' % (opt.input_img, opt.input_name), nrows=opt.row+1)
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

