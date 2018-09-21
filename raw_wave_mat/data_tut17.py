
import os
import numpy as np

# from lasagne_wrapper.data_pool import DataPool
from data_pools import AugmentedAudioFileClassificationDataPool
from audio_processors import tut17_processor, tut17_processor_mc, tut17_sig_proc, tut17_processor_mfs
from settings import DATA_ROOT_TUT17 as DATA_ROOT

N_WORKERS = 10

SPEC_CONTEXT = 500  # number of spectrogram frames
SPEC_BINS = 137     # number of frequency bins

CLASS_ID_MAPPING = dict()
CLASS_ID_MAPPING["beach"] = 0
CLASS_ID_MAPPING["bus"] = 1
CLASS_ID_MAPPING["cafe/restaurant"] = 2
CLASS_ID_MAPPING["car"] = 3
CLASS_ID_MAPPING["city_center"] = 4
CLASS_ID_MAPPING["forest_path"] = 5
CLASS_ID_MAPPING["grocery_store"] = 6
CLASS_ID_MAPPING["home"] = 7
CLASS_ID_MAPPING["library"] = 8
CLASS_ID_MAPPING["metro_station"] = 9
CLASS_ID_MAPPING["office"] = 10
CLASS_ID_MAPPING["park"] = 11
CLASS_ID_MAPPING["residential_area"] = 12
CLASS_ID_MAPPING["train"] = 13
CLASS_ID_MAPPING["tram"] = 14

ID_CLASS_MAPPING = dict(zip(CLASS_ID_MAPPING.values(), CLASS_ID_MAPPING.keys()))


def get_files_2_labels(txt_file):
    """
    Load files of split
    """
    with open(txt_file, 'r') as fp:
        file_list = fp.read()

    files2labels = dict()
    for line in file_list.split("\n"):
        # print line
        split_line = line.split("\t")

        if len(split_line) == 2:
            file_name = split_line[0].strip()
            files2labels[file_name] = CLASS_ID_MAPPING[split_line[1].strip()]

    # print files2labels
    return files2labels


def get_files_and_labels(txt_file, f2l):
    """
    Load files of split
    """
    with open(txt_file, 'r') as fp:
        train_list = fp.read()

    files = []
    labels = []
    for line in train_list.split("\n"):
        split_line = line.split("\t")

        if split_line[0] != '':
            file_name = split_line[0].strip()
            file_path = os.path.join(DATA_ROOT, file_name)

            files.append(file_path)
            labels.append(f2l[file_name])

        else:
            pass

    return np.asarray(files, dtype=np.string_), np.asarray(labels, dtype=np.int32)

def get_files_and_labels_test(txt_file, f2l):
    """
    Load files of split
    """
    with open(txt_file, 'r') as fp:
        train_list = fp.read()

    files = []
    labels = []
    for line in train_list.split("\n"):
        split_line = line.split("\t")

        if split_line[0] != '':
            file_name = split_line[0].strip()
            file_path = os.path.join('../DCASE2016-Wav-Files/TUT-acoustic-scenes-2016-evaluation', file_name)

            files.append(file_path)
            labels.append(f2l[file_name])

        else:
            pass

    return np.asarray(files, dtype=np.string_), np.asarray(labels, dtype=np.int32)



def load_data(fold=1, n_workers=N_WORKERS, multi_channel=False, raw_audio=False, multi_framesize=False):
    """ load data """

    split_dir = os.path.join(DATA_ROOT, "evaluation_setup")

    tr_file = '../DCASE2016-Wav-Files/TUT-acoustic-scenes-2016-development/evaluation_setup/fold1_train.txt'
    va_file = '../DCASE2016-Wav-Files/TUT-acoustic-scenes-2016-development/evaluation_setup/fold1_evaluate.txt'
    te_file = '../DCASE2016-Wav-Files/TUT-acoustic-scenes-2016-evaluation/evaluation_setup/test.txt'

    # meta_data_file = os.path.join(DATA_ROOT, 'meta.txt')
    meta_data_file = '../DCASE2016-Wav-Files/TUT-acoustic-scenes-2016-development/meta.txt'
    meta_data_file_test = '../DCASE2016-Wav-Files/TUT-acoustic-scenes-2016-evaluation/meta.txt'

    f2l = get_files_2_labels(meta_data_file)
    f2l_test = get_files_2_labels(meta_data_file_test)

    tr_files, tr_labels = get_files_and_labels(tr_file, f2l)
    va_files, va_labels = get_files_and_labels(va_file, f2l)
    te_files, te_labels = get_files_and_labels_test(te_file, f2l_test)

    # select spectrogram processor
    if multi_channel:
        proc = tut17_processor_mc
    elif multi_framesize:
        proc = tut17_processor_mfs
    elif raw_audio:
        proc = tut17_sig_proc
    else:
        proc = tut17_processor

    pool = AugmentedAudioFileClassificationDataPool

    train_pool = pool(tr_files, tr_labels, proc, n_workers=n_workers, shuffle=True, use_cache=True)
    valid_pool = pool(va_files, va_labels, proc, n_workers=n_workers, shuffle=False, use_cache=True)
    test_pool = pool(te_files, te_labels, proc, n_workers=n_workers, shuffle=False, use_cache=True)

    return {'train': train_pool, 'valid': valid_pool, 'test': test_pool}


def load_data_test(n_workers=N_WORKERS):
    test_root = DATA_ROOT.replace("TUT-acoustic-scenes-2016-development", "TUT-acoustic-scenes-2016-evaluation")
    split_dir = os.path.join(test_root, "evaluation_setup")
    te_file = os.path.join(split_dir, "test.txt")

    # load test set file list
    with open(te_file, 'r') as fp:
        file_list = fp.read()

    files, labels = [], []
    for line in file_list.split("\n"):
        split_line = line.split("\t")

        if split_line[0] != '':
            file_name = split_line[0].strip()
            file_path = os.path.join(test_root, file_name)

            files.append(file_path)
            labels.append(-1)

        else:
            pass

    te_files = np.asarray(files, dtype=np.string_)
    te_labels = np.asarray(labels, dtype=np.int32)

    # select spectrogram processor
    proc = tut17_processor

    # init test set pool
    pool = AugmentedAudioFileClassificationDataPool
    test_pool = pool(te_files, te_labels, proc, n_workers=n_workers, shuffle=False, use_cache=True)

    return {'test': test_pool}


def load_mfcc(fold=1):
    """ load data """

    split_dir = os.path.join(DATA_ROOT, "evaluation_setup")

    tr_file = os.path.join(split_dir, "fold%d_train.txt" % fold)
    va_file = os.path.join(split_dir, "fold%d_evaluate.txt" % fold)
    te_file = os.path.join(split_dir, "fold%d_test.txt" % fold)

    meta_data_file = os.path.join(DATA_ROOT, 'meta.txt')

    f2l = get_files_2_labels(meta_data_file)

    tr_files, tr_labels = get_files_and_labels(tr_file, f2l)
    va_files, va_labels = get_files_and_labels(va_file, f2l)
    te_files, te_labels = get_files_and_labels(te_file, f2l)

    def load_features(files):
        import scipy.io as sio
        X = []
        for f in files:
            f = f.replace("audio", "mfcc/DCASEv4left").replace(".wav", ".mat")
            feat = sio.loadmat(f)
            feat_l = feat["mfcc_features"].flatten()
            f = f.replace("audio", "mfcc/DCASEv4right").replace(".wav", ".mat")
            feat = sio.loadmat(f)
            feat_r = feat["mfcc_features"].flatten()
            feat = np.concatenate([feat_l, feat_r])
            X.append(feat)

        X = np.asarray(X, dtype=np.float32)

        return X

    # load features
    X_tr = load_features(tr_files)
    X_va = load_features(va_files)
    X_te = load_features(te_files)

    train_pool = DataPool(X_tr, tr_labels)
    valid_pool = DataPool(X_va, va_labels)
    test_pool = DataPool(X_te, te_labels)

    return {'train': train_pool, 'valid': valid_pool, 'test': test_pool}


def prepare(X, y):
    """ Prepare data for network processing """
    shape = X.shape
    if len(shape) == 3:
        X = X[:, np.newaxis]

    return X, y


def prepare_concat(X, y):
    X, y = prepare(X, y)
    X = np.concatenate((X[:, 0:1], X[:, 1:2]), axis=2)
    X = X[:, :, 0:2*SPEC_CONTEXT, :]
    return X, y


def prepare_random_slice(X, y):
    X, y = prepare(X, y)

    # apply random cyclic shift
    N_FRAMES = 100
    n_frames = X.shape[2] - N_FRAMES
    X_new = np.zeros((X.shape[0], X.shape[1], N_FRAMES, X.shape[3]), dtype=np.float32)
    for i in xrange(X.shape[0]):
        start = np.random.randint(0, n_frames)
        stop = start + N_FRAMES
        X_new[i] = X[i, :, start:stop, :]

    return X_new, y


def prepare_cyclic_shift(X, y):
    """ Prepare data for network processing
        (cyclic spectrogram shift)
    """
    X, y = prepare(X, y)

    # apply random cyclic shift
    n_frames = X.shape[2]
    for i in xrange(X.shape[0]):
        shift = np.random.randint(0, n_frames)
        X[i] = np.roll(X[i], shift=shift, axis=1)

    return X, y


def prepare_random_shift(X, y):
    """ Prepare data for network processing
        (cyclic spectrogram shift)
    """
    X, y = prepare(X, y)

    # apply random shift
    X_new = np.zeros_like(X, dtype=np.float32)
    n_slices = 5
    slice_length = X.shape[2] // n_slices
    borders = np.arange(0, n_slices) * slice_length
    for i in xrange(X.shape[0]):
        np.random.shuffle(borders)
        for j, b in enumerate(borders):
            s = j * slice_length
            e = s + slice_length
            X_new[i, 0, s:e, :] = X[i, 0, b:b+slice_length, :]

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(X[i, 0].T)
        plt.subplot(2, 1, 2)
        plt.imshow(X_new[i, 0].T)
        plt.show(block=True)

    return X_new, y


if __name__ == '__main__':
    """ main """
    # import time
    # import matplotlib.pyplot as plt
    # from lasagne_wrapper.batch_iterators import BatchIterator, threaded_generator_from_iterator
    #
    # # load data
    # data = load_data(n_workers=1)
    #
    # # init batch iterator
    # bi = BatchIterator(batch_size=10, k_samples=None, shuffle=True, prepare=prepare)
    #
    # print "Train Files:", data['test'].shape
    #
    # # iterate train data
    # for e in xrange(10):
    #     start_time = time.time()
    #     generator = threaded_generator_from_iterator(bi(data['test']))
    #     for i, (X, y) in enumerate(generator):
    #         print e, X.shape, y.shape
    #
    #         # get timing
    #         end_time = time.time()
    #         print "%.2f seconds required for batch." % (end_time - start_time)
    #         start_time = end_time
    #
    #         # show data
    #         plt.figure("spectrogram")
    #         plt.imshow(X[0, 0].T, aspect='auto', origin='lower')
    #         plt.ylabel("%d bins" % X[0, 0].shape[1])
    #         plt.xlabel("%d frames" % X[0, 0].shape[0])
    #         plt.title(ID_CLASS_MAPPING[y[0]])
    #         plt.show(block=True)

    # import time
    # import matplotlib.pyplot as plt
    # from lasagne_wrapper.batch_iterators import BatchIterator, threaded_generator_from_iterator
    #
    # # load data
    # data = load_data(fold=1, multi_channel=True)
    #
    # # init batch iterator
    # bi = BatchIterator(batch_size=10, k_samples=None, shuffle=True, prepare=prepare_concat)
    #
    # print "Train Files:", data['test'].shape
    #
    # # iterate train data
    # for e in xrange(10):
    #     start_time = time.time()
    #     generator = threaded_generator_from_iterator(bi(data['test']))
    #     for i, (X, y) in enumerate(generator):
    #         print e, X.shape, y.shape

    data = load_data_test(n_workers=N_WORKERS)
