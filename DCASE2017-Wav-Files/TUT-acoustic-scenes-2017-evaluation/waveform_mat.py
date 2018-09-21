import sys
import numpy as np

from madmom.audio.spectrogram import LogarithmicFilterbank, LogarithmicFilteredSpectrogramProcessor
from madmom.processors import SequentialProcessor
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor, Signal
from madmom.audio.filters import LogFilterbank
# import matplotlib.pyplot as plt
from scipy.io import wavfile

SAMPLE_RATE = 22050
FRAME_SIZE = 2048
FPS = 50
max_length = 220501
channel = 2

my_map = {'bus':0, 'cafe/restaurant':1, 'car':2, 'city_center':3, 'forest_path':4, 'grocery_store':5,
'home':6, 'beach':7, 'library':8, 'metro_station':9, 'office':10, 'residential_area':11, 
'train':12, 'tram':13, 'park':14}

# sig_proc = SignalProcessor(num_channels=2, norm=True, sample_rate=SAMPLE_RATE, att=0)
# fsig_proc = FramedSignalProcessor(frame_size=FRAME_SIZE, fps=FPS, online=False)
# seq_proc = SequentialProcessor((sig_proc,fsig_proc))
data = '1024.wav'

def get_wav(filename):
	dmy = Signal(filename, sample_rate=SAMPLE_RATE, num_channels=2, start=None, stop=None, norm=True, gain=0.0, dtype=np.float32)
	# dmy = np.mean(dmy,axis=1)
	# print dmy.shape
	return dmy

# wavfile.write('tmp.wav',SAMPLE_RATE,dmy)
# get_wav(data)
# print dmy.shape

# wave = get_wav(data)
# print wave.shape
# wavfile.write('tmp.wav',SAMPLE_RATE,wave)

file = open(sys.argv[1])
num_data = sum(1 for line in file)
file = open(sys.argv[1])
images = np.empty(shape = [num_data,max_length,channel])
labels = np.empty(num_data)
counter = 0
for iter,line in enumerate(file.readlines()): # first line of traininData.csv is header (only for trainingData.csv)
    filename = line.split()[0]
    label = line.split()[1]
    wave = get_wav(filename)
    wavfile.write('tmp.wav',SAMPLE_RATE,wave)
    # plt.imshow(spectrogram)
    # plt.show()
    images[counter] = wave
    labels[counter] = my_map[label]
    counter += 1
    print (counter)

print(str(counter))  
if num_data != counter:
    print('num_data and counter dont match')                     
np.save('waves/2channel2017_madmom_wave_' + sys.argv[1][:-4] + '_images.npy',images)
np.save('waves/2channel2017_madmom_wave_' + sys.argv[1][:-4] + '_labels.npy', labels)

