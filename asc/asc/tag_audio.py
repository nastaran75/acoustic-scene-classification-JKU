
from __future__ import print_function

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from lasagne_wrapper.network import Network

from config.settings import EXP_ROOT
from train import select_model, get_dump_file_paths, load_data

from utils.audio_processors import processor


if __name__ == '__main__':
    """ main """
    # example call: "python tag_audio.py --model models/jamendo156_nin_full.py --audio_file /media/rk0/shared/datasets/jamendo/audiofiles/120073.mp3"

    # add argument parser
    parser = argparse.ArgumentParser(description='Predict tags for given music audio.')
    parser.add_argument('--model', help='select tagger model.')
    parser.add_argument('--params', help='select model parameter file.', type=str, default=None)
    parser.add_argument('--audio_file', help='select audio file to be tagged.')
    parser.add_argument('--top_k', help='top k tags to print.', type=int, default=5)
    parser.add_argument('--window_size', help='slide window over audio for prediction.', type=int, default=None)

    args = parser.parse_args()

    # init audio processor
    audio_processor = processor

    # get index to tag mapping
    data, id_to_tag = load_data("jamendo156_tags", fold=None, n_workers=1)

    # select model
    model = select_model(args.model)

    # set model dump file
    # todo: this should be moved to a more general place
    print("\nLoading model parameters ...")
    # out_path = os.path.join(os.path.join(EXP_ROOT), model.EXP_NAME)
    # dump_file, _ = get_dump_file_paths(out_path, None)
    dump_file = args.params if args.params else "/home/matthias/experiments/asc/jamendo156_nin_full/params.pkl"

    # compile network
    net = model.build_model(batch_size=1)

    # initialize neural network
    my_net = Network(net, print_architecture=False)

    # load model parameters network
    my_net.load(dump_file)

    print("\nTagging song: %s" % args.audio_file)

    # process audio
    spec = audio_processor.process(args.audio_file)

    # prepare spectrogram for network
    spec = spec[np.newaxis, np.newaxis]

    # predict tags on full audio
    y_pred = my_net.predict_proba(spec)[0]

    # print to k tags
    prob_sorted_idxs = np.argsort(y_pred)[::-1]
    for i in xrange(args.top_k):
        tag = id_to_tag[prob_sorted_idxs[i]].ljust(25)
        prob = y_pred[prob_sorted_idxs[i]]
        print("%s: %.3f" % (tag, prob))

    # compute latent representation (activation of layer by name, "l0" of "l1")
    # latent = my_net.compute_layer_output(spec, layer="l0")[0]

    # run sliding window over audio
    if args.window_size:
        print("\nSliding window tagging ...")

        # iterate over spectrogram
        tags = []
        pred_times = []
        steps = np.arange(0, spec.shape[2] - args.window_size, args.window_size)
        print("Number of steps: %d" % len(steps))
        for i_step, f0 in enumerate(steps):

            # predict on sliding window
            f1 = f0 + args.window_size
            y_pred = my_net.predict_proba(spec[:, :, f0:f1, :])[0]

            # get tags
            local_tags = []
            prob_sorted_idxs = np.argsort(y_pred)[::-1]
            for i in xrange(args.top_k):
                local_tags.append(id_to_tag[prob_sorted_idxs[i]])

            # book keeping
            tags.append(local_tags)
            pred_times.append(f0 + args.window_size // 2)

        # visualize results
        print("\nCreateing plot ... ", end="")
        plt.figure("Audio Tagging", figsize=(30, 10))

        plt.subplot(2, 1, 1)
        plt.imshow(spec[0, 0].T, cmap='viridis', origin='lower', aspect='auto')

        plt.subplot(2, 1, 2)
        for p_time, l_tags in zip(pred_times, tags):
            for j, t in enumerate(l_tags):
                plt.plot(p_time, j, 'wo', markersize=0)
                plt.text(p_time, j, t, va="center", ha="center", fontsize=12, rotation=45)
        plt.xlim([0, spec.shape[2]])
        plt.axis("off")

        plt.savefig("tmp.png")
        print("saved to tmp.png.")
