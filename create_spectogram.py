from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from PIL import Image

# data info
# TODO make args
sampling_freq = 1000
row = 15

data = pd.read_csv("emg_data/PSU_Data_200ms_part1.csv", nrows=100)
x = data.values[row, 1:]
class_label = data.values[row, 0]

# plot temporal signal
plt.plot(np.arange(len(x)), x)
plt.savefig('Input/Spectrograms/input_signal.png')
plt.close()

# calculate spectogram image
f, t, Zxx = signal.stft(x, sampling_freq,
                        nperseg=int(sampling_freq / 4),
                        noverlap=int(sampling_freq / 4) - 2
                        )

# generate spectogram image
min_intensity = np.min(Zxx.real)
max_intensity = np.max(Zxx.real)

img_array = ((Zxx.real / max_intensity) * 255)

img = Image.fromarray(img_array).convert('RGB')
img.save("Input/Spectrograms/spectrogram.png")

# print min and max values
min_freq = min(f)
max_freq = max(f)
min_t = min(t)
max_t = max(t)

print("Class: " + str(class_label))
print("Time range is from " + str(min_t) + " to " + str(max_t) + " seconds")
print("Frequency range is from " + str(min_freq) + " to " + str(max_freq) + " Hz")
print("Intensity range is from " + str(min_intensity) + " to " + str(max_intensity))

# generate JSON with values
info = dict()

info['min_freq'] = min_freq
info['max_freq'] = max_freq
info['min_t'] = min_t
info['max_t'] = max_t
info['max_intensity'] = max_intensity
info['min_intensity'] = min_intensity
info['sampling_freq'] = sampling_freq
info['class_label'] = int(class_label)

with open('Input/Spectrograms/data.json', 'w', encoding='utf-8') as json_file:
    json.dump(info, json_file, ensure_ascii=False, indent=4)
