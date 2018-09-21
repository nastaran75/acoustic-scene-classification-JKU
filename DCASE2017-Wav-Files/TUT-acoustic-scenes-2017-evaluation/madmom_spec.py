import sys
import numpy as np

from madmom.audio.spectrogram import LogarithmicFilterbank, LogarithmicFilteredSpectrogramProcessor
from madmom.processors import SequentialProcessor
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.filters import LogFilterbank


my_map = {'bus':0, 'cafe/restaurant':1, 'car':2, 'city_center':3, 'forest_path':4, 'grocery_store':5,
'home':6, 'beach':7, 'library':8, 'metro_station':9, 'office':10, 'residential_area':11, 
'train':12, 'tram':13, 'park':14}


SAMPLE_RATE = 22050
FRAME_SIZE = 2048
FPS = 50

sig_proc = SignalProcessor(num_channels=2, norm=True, sample_rate=SAMPLE_RATE, att=0)
fsig_proc = FramedSignalProcessor(frame_size=FRAME_SIZE, fps=FPS, online=False)
spec_proc = LogarithmicFilteredSpectrogramProcessor(filterbank=LogFilterbank, num_bands=24, fmin=20, fmax=8000,norm_filters=True)
seq_proc = SequentialProcessor((sig_proc,fsig_proc))

def get_spec(file_name):
    dmy = seq_proc(file_name)
    dmy.signal = dmy.signal.mean(1)
    S = spec_proc(dmy)
    return np.array(S, dtype=np.float32)


height = 501
width = 137
file = open(sys.argv[1])
num_data = sum(1 for line in file)
file = open(sys.argv[1])
images = np.empty(shape = [num_data,1,height,width])
labels = np.empty(num_data)
counter = 0
for iter,line in enumerate(file.readlines()): # first line of traininData.csv is header (only for trainingData.csv)
    filename = line.split()[0]
    label = line.split()[1]
    spectrogram = get_spec(filename)
    print spectrogram.shape
    # plt.imshow(spectrogram)
    # plt.show()
    images[counter] = spectrogram
    labels[counter] = my_map[label]
    counter += 1
    print counter

print str(counter)
if num_data != counter:
    print('num_data and counter dont match')                     
np.save('new2_2017_' + sys.argv[1][:-4] + '_images.npy',images)
np.save('new2_2017_' + sys.argv[1][:-4] + '_labels.npy', labels)
