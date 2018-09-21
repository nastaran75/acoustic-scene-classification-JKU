import numpy as np
from madmom.audio.spectrogram import LogarithmicFilterbank, LogarithmicFilteredSpectrogramProcessor
from madmom.processors import SequentialProcessor
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor, Signal
from madmom.audio.filters import LogFilterbank
from scipy.io import wavfile



SAMPLE_RATE = 22050
FRAME_SIZE = 2048
FPS = 50
max_length = 220501

def add_noise(data):
        noise = np.random.randn(len(data))
        data_noise = data + 0.005 * noise
        return data_noise

def shift(data):
        return np.roll(data, 16000)

def get_wav(data):
	dmy = Signal(data, sample_rate=SAMPLE_RATE, num_channels=2, start=None, stop=None, norm=False, gain=0.0, dtype=np.float32)
	dmy = np.mean(dmy,axis=1)
	# print dmy.shape
	return dmy

data = '1024.wav'
orig = get_wav(data)
print orig.shape
wavfile.write('orig.wav',SAMPLE_RATE,orig)
shifted = shift(orig)
wavfile.write('shifted.wav',SAMPLE_RATE,shifted)
noisy = add_noise(orig)
wavfile.write('noisy.wav',SAMPLE_RATE,noisy)

