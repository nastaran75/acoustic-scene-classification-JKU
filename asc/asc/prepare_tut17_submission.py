
import argparse
import numpy as np
import scipy.io as spio
from utils.data_tut17 import ID_CLASS_MAPPING

if __name__ == "__main__":
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Generate DACASE submission file.')
    parser.add_argument('--prediction_file', help='prediction file.')
    parser.add_argument('--out_file', help='name of submission.', default=None)
    parser.add_argument('--plot_predictions', help='plot prediction matrix.', action='store_true')
    args = parser.parse_args()

    # check if outfile is specified
    if args.out_file is None:
        print "Stopping script: Outfile not specified!"
        exit(0)

    # load prediction
    prediction = spio.loadmat(args.prediction_file)

    # get file names and predictions
    files_names = prediction['files']
    y_probs = prediction['y_probs']

    if args.plot_predictions:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(y_probs, aspect="auto", interpolation="nearest")
        plt.show(block=True)

    # iterate files
    with open(args.out_file, 'wb') as fp:
        for i in xrange(len(files_names)):
            start_idx = files_names[i].find("audio")
            file_name = files_names[i][start_idx:]
            y_pred = np.argmax(y_probs[i])
            label = ID_CLASS_MAPPING[y_pred]

            entry = "%s %s\n" % (file_name, label)
            fp.write(entry)
