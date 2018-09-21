
from __future__ import print_function

import os
from os.path import basename

import numpy as np
import argparse
import time
from sklearn.metrics import accuracy_score

from lasagne_wrapper.batch_iterators import BatchIterator
from lasagne_wrapper.network import Network

from train import select_model, load_data
from config.settings import EXP_ROOT

from lasagne_wrapper.utils import BColors
col = BColors()

# seed seed for consistency
np.random.seed(4711)


if __name__ == '__main__':
    """ main """
    # add argument parser
    parser = argparse.ArgumentParser(description='Train multi-modality model.')
    parser.add_argument('--model', help='select model to train.')
    parser.add_argument('--data', help='select model to train.')
    parser.add_argument('--fold', help='train split.', type=int, default=None)
    parser.add_argument('--n_workers', help='train split.', type=int, default=1)
    parser.add_argument('--keep_best_k', help='keep best k models.', type=int, default=5)
    parser.add_argument('--max_runs', help='Max number of training runs', type=int, default=50)
    parser.add_argument('--store_details', help='store detailed results to file', action='store_true')
    args = parser.parse_args()

    # set number of threads for spectrogram processing
    os.environ["OMP_NUM_THREADS"] = "1"

    # select model
    model = select_model(args.model)

    # change model name
    model.EXP_NAME += "_model_search"

    # load data
    print("Loading data ...")
    data, _ = load_data(args.data, args.fold, args.n_workers)

    # prepare book keeping
    best_accuracies = dict()
    all_accs = []

    if args.store_details:
        import pandas as pd
        details = {'model': [],
                   'data': [],
                   'fold': [],
                   'run': [],
                   'file': [],
                   'y_true': [],
                   'y_pred': []}

    # search forever
    run_count = 0
    while run_count < args.max_runs:
        run_count += 1
        print(col.print_colored("\nStarting run %d ..." % run_count, col.WARNING))

        # get time stamp
        tstamp = str(time.ctime())

        # set model dump file
        out_path = os.path.join(os.path.join(EXP_ROOT), model.EXP_NAME)
        par = 'params_%d_%s.pkl' % (args.fold, tstamp)
        log = 'results_%d.pkl' % args.fold
        dump_file = os.path.join(out_path, par)
        log_file = os.path.join(out_path, log)

        report_file = os.path.join(out_path, 'report_%d.txt' % args.fold)
        details_file = os.path.join(out_path, 'detailed_report_%d.pkl' % args.fold)

        # compile network
        net = model.build_model()

        # initialize neural network
        my_net = Network(net, print_architecture=False)

        # train network
        acc = my_net.fit(data, model.get_train_strategy(), log_file=log_file, dump_file=None)

        if args.store_details:
            # init batch iterator
            bi = BatchIterator(batch_size=1, k_samples=None, shuffle=False, prepare=model.prepare)
            for i, (X, y_true) in enumerate(bi(data['test'])):
                f = data['test'].files[i]
                y_pred = my_net.predict(X)[0]
                details['model'].append(args.model)
                details['data'].append(args.data)
                details['fold'].append(args.fold)
                details['run'].append(run_count - 1)
                details['file'].append(basename(f))
                details['y_pred'].append(y_pred)
                details['y_true'].append(y_true[0])

            df = pd.DataFrame(details)
            df.to_pickle(details_file)

            # recompute accuracy
            acc = accuracy_score(details['y_true'], details['y_pred'])

        # book keeping
        all_accs.append(acc)

        # keep model
        if len(best_accuracies) < args.keep_best_k:
            best_accuracies[par] = acc
            my_net.save(dump_file)

        # replace model if current one is better
        else:

            # find worst model so far
            keys = best_accuracies.keys()
            accs = best_accuracies.values()
            sorted_idx = np.argsort(accs)
            worst_model = keys[sorted_idx[0]]

            # check if current model is better
            if acc > accs[sorted_idx[0]]:
                print(col.print_colored("Found better model ...", col.OKBLUE))

                # delete worst model
                best_accuracies.pop(worst_model)
                worst_path = os.path.join(out_path, worst_model)
                if os.path.exists(worst_path):
                    print(col.print_colored("Replacing old one!", col.OKBLUE))
                    print(worst_path)
                    os.remove(worst_path)

                # keep current model
                best_accuracies[par] = acc
                my_net.save(dump_file)

        # print report current status
        print(col.print_colored("\nLogging Results ...", col.UNDERLINE))
        with open(report_file, "wb") as fp:
            for key, acc in best_accuracies.iteritems():
                txt = "%s: %.3f" % (key, acc)
                print(txt)
                fp.write(txt + "\n")

            txt = "Runs: %d | Mean: %.3f | Std: %.3f | Min: %.3f | Max: %.3f" % (len(all_accs), np.mean(all_accs), np.std(all_accs), np.min(all_accs), np.max(all_accs))
            print(txt)
            fp.write(txt + "\n")
