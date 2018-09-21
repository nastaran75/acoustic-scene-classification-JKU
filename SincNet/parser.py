from argparse import ArgumentParser


def opts_parser():
    usage = "Trains and tests sampleCNN on the given dataset"
    parser = ArgumentParser(description=usage)
    parser.add_argument(
            '--depth', type=int, default=36,
            help='Network depth in layers (default: %(default)s)')
    parser.add_argument(
            '--dataset_name', type=str, default = '2channel2016_madmom',
            help='the name of the dataset')
    parser.add_argument(
            '--nonL', type=str, default = 'rectify',
            help='nonlinearity')
    parser.add_argument(
            '--fold', type=int, default=1,
            help='the fold number')
    parser.add_argument(
            '--division', type=int, default=10,
            help='divide each file to how many pieces')
    parser.add_argument(
            '--augment', action='store_true', default=True,
            help='Perform data augmentation (enabled by default)')
    parser.add_argument(
            '--no-patience', action='store_false', dest='patience',
            help='Disable patience')
    parser.add_argument(
            '--patience', action='store_true', default = True,
            help='patience training')
    parser.add_argument(
            '--representation', type = str, default='Mid',
            help='it can be Mid, Side, Left or Right')
    parser.add_argument(
            '--no-augment', action='store_false', dest='augment',
            help='Disable augmentation')
    parser.add_argument(
            '--batch_size', type=int, default=25,
            help='batch-size')
    parser.add_argument(
            '--epochs', type=int, default=200,
            help='Number of training epochs (default: %(default)s)')
    parser.add_argument(
            '--learning_rate', type=float, default=0.001,
            help='Initial learning rate (default: %(default)s)')
    parser.add_argument(
            '--dropout', type=float, default=0,
            help='Dropout rate (default: %(default)s)')
    parser.add_argument(
            '--growth_rate', type=int, default=12,
            help='Growth rate in dense blocks (default: %(default)s)')
    # parser.add_argument(
    #         '--save-errors', type=str, default=None, metavar='FILE',
    #         help='If given, save train/validation errors to given .npz file')
    return parser