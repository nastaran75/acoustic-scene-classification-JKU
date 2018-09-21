
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from lasagne_wrapper.network import Network

from config.settings import EXP_ROOT
from train import select_model, load_data, get_dump_file_paths
from lasagne_wrapper.batch_iterators import BatchIterator


def print_result(fold, y, y_predicted, id_class_mapping):
    """ print result matrix """

    n_classes = len(np.unique(y))

    p, r, f, s = precision_recall_fscore_support(y, y_predicted, labels=None, pos_label=1, average=None)
    a = [(accuracy_score(y[y == c], y_predicted[y == c])) for c in xrange(n_classes)]

    # count occurrences of classes
    count = Counter(y)

    print("\n")
    print("Results on fold %d" % fold)
    print("\n")
    print("%20s  |  %s  |  %5s  |  %4s  |  %4s  |   %4s   |" % ("LABEL", "CNT", "ACC ", "PR ", "RE ", "F1 "))
    print('-' * 70)
    for c in xrange(n_classes):
        print("%20s  |  %03d  |  %0.3f  |  %.2f  |  %.2f  |  %.3f   |" % (id_class_mapping[c], count[c], a[c], p[c], r[c], f[c]))
    print('-' * 70)
    print("%20s  |  %03d  |  %0.3f  |  %.2f  |  %.2f  |  %.3f   |" % ('average', len(y), np.mean(a), np.mean(p), np.mean(r), np.mean(f)))
    print('=' * 70)
    print("Overall Accuracy: %.3f %%" % (100.0 * accuracy_score(y, y_predicted)))
    print('=' * 70)


if __name__ == '__main__':
    """ main """
    # add argument parser
    parser = argparse.ArgumentParser(description='Train multi-modality model.')
    parser.add_argument('--model', help='select model to evaluate.')
    parser.add_argument('--params', help='select model parameters to evaluate.', default=None)
    parser.add_argument('--data', help='select model to train.')
    parser.add_argument('--fold', help='train split.', type=int, default=0)
    parser.add_argument('--dump_results', help='dump results (.pkl, .mat).', type=str, default=None)
    args = parser.parse_args()

    # select model
    model = select_model(args.model)

    # load data
    print("Loading data ...")
    data, id_class_mapping = load_data(args.data, args.fold, n_workers=1)

    # set model dump file
    print("Loading model parameters ...")
    out_path = os.path.join(os.path.join(EXP_ROOT), model.EXP_NAME)
    dump_file, log_file = get_dump_file_paths(out_path, args.fold)

    # overwrite model parameters
    if args.params:
        dump_file = args.params
        print("overwriting parameter dump file", os.path.basename(dump_file))

    # compile network
    net = model.build_model(batch_size=1)

    # initialize neural network
    my_net = Network(net, print_architecture=False)

    # load model parameters network
    my_net.load(dump_file)

    # init batch iterator
    bi = BatchIterator(batch_size=1, k_samples=None, shuffle=False, prepare=model.prepare)

    # iterate train data
    print("Predicting on test set ...")
    y_true = []
    y_probs = []
    y_predicted = []
    for i, (X, y) in enumerate(bi(data['test'])):
        print("Processing file %d / %d" % (i + 1, data['test'].shape[0]), end='\r')
        sys.stdout.flush()

        # predict on file
        p_pred = my_net.predict_proba(X)[0]
        y_pred = np.argmax(p_pred)
        y_predicted.append(y_pred)
        y_probs.append(p_pred)
        y_true.append(y[0])

    # convert to array
    y_predicted = np.asarray(y_predicted, dtype=np.int32)
    y_probs = np.asarray(y_probs, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int32)

    # present results
    print_result(args.fold, y_true, y_predicted, id_class_mapping)

    # dump matlab result matrices
    if args.dump_results:

        if args.dump_results == ".mat":
            import scipy.io as spio
            file_name = os.path.basename(dump_file).replace("params", "probs_%s" % args.data)
            mat_file = log_file = os.path.join(out_path, file_name)
            spio.savemat(mat_file, {'y_probs': y_probs, 'y_true': y_true, 'files': data['test'].files})

        elif args.dump_results == ".pkl":
            print("Not implemented yet!")
