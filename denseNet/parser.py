from argparse import ArgumentParser


def opts_parser():
    usage = "Trains and tests sampleCNN on the given dataset"
    parser = ArgumentParser(description=usage)
    parser.add_argument(
            '--dataset_name', type=str, default = '2channel2016_madmom',
            help='the name of the dataset')
    parser.add_argument(
            '--fold', type=int, default=1,
            help='the fold number')
    parser.add_argument(
            '--division', type=int, default=9,
            help='divide each file to how many pieces')
    parser.add_argument(
            '--augment', action='store_true', default=True,
            help='Perform data augmentation (enabled by default)')
    parser.add_argument(
            '--patience', action='store_true', default = True,
            help='patience training')
    parser.add_argument(
            '--representation', type = str, default='Mid',
            help='it can be Mid, Side, Left or Right')
    parser.add_argument(
            '--no-augment', action='store_false', dest='augment',
            help='Disable augmentation')
    # parser.add_argument(
    #         '--validate-test', action='store_const', dest='validate',
    #         const='test', help='Perform validation on test set')
    parser.add_argument(
            '--epochs', type=int, default=150,
            help='Number of training epochs (default: %(default)s)')
    parser.add_argument(
            '--learning_rate', type=float, default=0.02,
            help='Initial learning rate (default: %(default)s)')
    # parser.add_argument(
    #         '--save-weights', type=str, default=None, metavar='FILE',
    #         help='If given, save network weights to given .npz file')
    # parser.add_argument(
    #         '--save-errors', type=str, default=None, metavar='FILE',
    #         help='If given, save train/validation errors to given .npz file')
    return parser