
import subprocess
import numpy as np

from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import LogarithmicFilteredSpectrogramProcessor, \
    FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor
from madmom.audio.filters import MelFilterbank, LogarithmicFilterbank
from madmom.processors import SequentialProcessor, ParallelProcessor
from madmom.processors import Processor

import librosa
import soundfile as sf


def get_file_info(infile, cmd='ffprobe'):
    """
    Extract and return information about audio files.

    Parameters
    ----------
    infile : str
        Name of the audio file.
    cmd : {'ffprobe', 'avprobe'}, optional
        Probing command (defaults to ffprobe, alternatively supports avprobe).

    Returns
    -------
    dict
        Audio file information.

    """
    # check input file type
    if not isinstance(infile, str):
        raise ValueError("only file names are supported as `infile`, not %s."
                         % infile)
    # init dictionary
    info = {'num_channels': None, 'sample_rate': None}
    # call ffprobe
    output = subprocess.check_output([cmd, "-v", "quiet", "-show_streams",
                                      infile])
    # parse information
    for line in output.split():
        if line.startswith(b'channels='):
            info['num_channels'] = int(line[len('channels='):])
        if line.startswith(b'sample_rate='):
            # the int(float(...)) conversion is necessary because
            # avprobe returns sample_rate as floating point number
            # which int() can't handle.
            info['sample_rate'] = int(float(line[len('sample_rate='):]))
        if line.startswith(b'duration='):
            info['duration'] = float(line[len('duration='):])
    # return the dictionary
    return info


class RandomAudioSliceProcessor(Processor):

    def __init__(self, n_seconds=20, num_target_frames=None):
        self.n_seconds = n_seconds
        self.length_cache = dict()


    def process(self, file_path):

        # get file info
        if file_path not in self.length_cache:
            info = get_file_info(file_path)
            self.length_cache[file_path] = int(info['duration'])
        duration = self.length_cache[file_path]

        # get random sub-signal
        max_start = np.max([1, duration - self.n_seconds])
        start = np.random.randint(0, max_start)
        stop = start + self.n_seconds

        # load signal signal
        sig = sig_proc.process(file_path, start=start, stop=stop)

        # compute spectrogram
        fsig = fsig_proc.process(sig)
        spec = spec_proc.process(fsig)

        return spec


class MultiChannelSpectrogramProcessor(Processor):

    def __init__(self, signal_processor, spectrogram_processor):
        self.signal_processor = signal_processor
        self.spectrogram_processor = spectrogram_processor

    def process(self, data, **kwargs):
        spectrograms = []
        signals = self.signal_processor.process(data)
        for i_sig in xrange(signals.shape[1]):
            spectrogram = self.spectrogram_processor.process(signals[:, i_sig])
            spectrograms.append(spectrogram)

        return np.stack(spectrograms)


class SpectrogramSliceProcessor(Processor):

    def __init__(self, num_target_frames):
        self.num_target_frames = num_target_frames

    def process(self, spectrogram):
        n_frames = spectrogram.shape[0]

        # pad spectrogram
        if n_frames < self.num_target_frames:
            n_missing = self.num_target_frames - n_frames
            spectrogram = np.pad(spectrogram, ((0, n_missing), (0, 0)), mode='reflect')

        # get random slice of spectrogram
        elif n_frames > self.num_target_frames:
            max_start = np.max([1, n_frames - self.num_target_frames])
            start = np.random.randint(0, max_start)
            stop = start + self.num_target_frames

            spectrogram = spectrogram[start:stop, :]

        # do nothing
        else:
            pass

        return spectrogram


class RawAndSpectrogramProcessor(Processor):

    def __init__(self, signal_processor, spectrogram_processor):
        self.signal_processor = signal_processor
        self.spectrogram_processor = spectrogram_processor

    def process(self, data, **kwargs):
        signal = self.signal_processor.process(data)
        spectrogram = self.spectrogram_processor.process(data)

        return signal, spectrogram


class LibrosaProcessor(Processor):

    def __init__(self):
        pass

    def process(self, file_path, **kwargs):
        y, sr = sf.read(file_path)
        spectrogram = librosa.feature.melspectrogram(y=y.mean(1), sr=sr, n_mels=128)
        spectrogram = librosa.logamplitude(spectrogram, ref_power=np.max).T
        return spectrogram


# default processor
sig_proc = SignalProcessor(num_channels=1, sample_rate=22050, norm=True)
fsig_proc = FramedSignalProcessor(frame_size=2048, fps=31.25, origin='future')
spec_proc = LogarithmicFilteredSpectrogramProcessor(num_bands=24, fmin=20, fmax=16000)
processor = SequentialProcessor([sig_proc, fsig_proc, spec_proc])

# tut17 processors
tut17_sig_proc = SignalProcessor(num_channels=1, sample_rate=22050, norm=True)
tut17_fsig_proc = FramedSignalProcessor(frame_size=2048, fps=50, origin='future')
tut17_spec_proc = FilteredSpectrogramProcessor(filterbank=LogarithmicFilterbank, num_bands=24, fmin=20, fmax=8000)
tut17_slice_proc = SpectrogramSliceProcessor(num_target_frames=500)
tut17_processor = SequentialProcessor([tut17_sig_proc, tut17_fsig_proc, tut17_spec_proc, tut17_slice_proc])

tut17_processor_mc = MultiChannelSpectrogramProcessor(SignalProcessor(num_channels=None, sample_rate=22050, norm=True),
                                                      SequentialProcessor([tut17_fsig_proc, tut17_spec_proc]))

tut17_raw_and_spec_proc = RawAndSpectrogramProcessor(tut17_sig_proc, tut17_processor)

tut17_sig_proc_44k = SignalProcessor(num_channels=1, sample_rate=44100, norm=True)
tut17_multi_framesize = ParallelProcessor([])
for frame_size in [2048, 4096, 8192]:
    frames = FramedSignalProcessor(frame_size=frame_size, fps=50)
    filt = FilteredSpectrogramProcessor(
        filterbank=LogarithmicFilterbank, num_bands=24, fmin=20, fmax=8000,
        norm_filters=True, unique_filters=False)
    spec = LogarithmicSpectrogramProcessor()
    # process each frame size with spec and diff sequentially
    tut17_multi_framesize.append(SequentialProcessor((frames, filt, spec)))
tut17_processor_mfs = SequentialProcessor((tut17_sig_proc_44k, tut17_multi_framesize, np.stack))


# librosa processor
tut17_librosa_proc = LibrosaProcessor()


if __name__ == '__main__':
    """ main """
    print get_file_info("/media/rk0/shared/datasets/jamendo/audiofiles/12345.mp3")
