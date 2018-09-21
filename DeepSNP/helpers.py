from __future__ import print_function

import types
import lasagne
import theano
import theano.tensor as T
import collections
# from drum_o_gan import SAMPLES_PATH, PROJECT_PATH
# from drum_o_gan.data import get_patterns_as_numpy, NOTES_PER_BAR, drum_insts, render_drum_pattern
import os
import datetime
import time
from shutil import copy2

try:
    from plot_settings import setup_plot_style, COLUMN_WIDTH, PALETTE

    setup_plot_style(headless=False)
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    rc_xelatex = {'pgf.rcfonts': False}
    mpl.rcParams.update(rc_xelatex)
    PLOTTING = True
except ImportError:
    save_plots_default = False
    import numpy as np
    mpl = None
    plt = None
    setup_plot_style = None
    COLUMN_WIDTH = 2
    PLOTTING = False
    print("error loading matplotlib, plotting disabled.")

NUM_BARS_DEFAULT = 4
THRESHOLD = 0.5
RENDER_TEMPO = 140


# init color printer
class BColors:
    """
    Colored command line output formatting
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def __init__(self):
        """ Constructor """
        pass

    @staticmethod
    def print_colored(string, color):
        """ Change color of string """
        return color + string + BColors.ENDC


def pb(s):
    col = BColors()
    return col.print_colored(s, BColors.OKBLUE)


def print_net_architecture(net, tag=None, detailed=False):
    """ Print network architecture """
    import lasagne

    col = BColors()
    print('\n')

    if tag is not None:
        print(col.print_colored('Net-Architecture: %s' % tag, BColors.UNDERLINE))
    else:
        print(col.print_colored('Net-Architecture:', BColors.UNDERLINE))

    layers = lasagne.layers.helper.get_all_layers(net)
    max_len = np.max([len(l.__class__.__name__) for l in layers]) + 7
    for l in layers:
        class_name = l.__class__.__name__
        output_shape = str(l.output_shape)

        if isinstance(l, lasagne.layers.DropoutLayer):
            class_name += "(%.2f)" % l.p

        class_name = class_name.ljust(max_len)
        output_shape = output_shape.ljust(25)

        if isinstance(l, lasagne.layers.InputLayer):
            class_name = col.print_colored(class_name, BColors.OKBLUE)

        if isinstance(l, lasagne.layers.MergeLayer):
            class_name = col.print_colored(class_name, BColors.WARNING)

        layer_details = ""
        if detailed:
            layer_details = []

            # add nonlinearity
            if hasattr(l, "nonlinearity"):
                if isinstance(l.nonlinearity, types.FunctionType):
                    layer_details.append(pb("NL: ") + str(l.nonlinearity.__name__))
                else:
                    layer_details.append(pb("NL: ") + str(l.nonlinearity.__class__.__name__))

            # print weight shape if possible
            if hasattr(l, 'W'):
                weight_shape = str(l.W.get_value().shape)
                layer_details.append(pb("W: ") + weight_shape)

            # print bias shape if possible
            if hasattr(l, 'b'):
                bias_shape = str(l.b.get_value().shape) if l.b is not None else "None"
                layer_details.append(pb("b: ") + bias_shape)

            # print scaler shape if possible
            if hasattr(l, 'gamma'):
                bias_shape = str(l.beta.get_value().shape) if l.beta is not None else "None"
                layer_details.append(pb("gamma: ") + bias_shape)

            # print bias shape if possible
            if hasattr(l, 'beta'):
                bias_shape = str(l.beta.get_value().shape) if l.beta is not None else "None"
                layer_details.append(pb("beta: ") + bias_shape)

            layer_details = ", ".join(layer_details)

        print(class_name, output_shape, layer_details)


# def amsgrad(loss_or_grads, params, learning_rate=0.001, beta1=0.9,
#             beta2=0.999, epsilon=1e-8):
#     all_grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
#     t_prev = theano.shared(lasagne.utils.floatX(0.))
#     updates = collections.OrderedDict()

#     # Using theano constant to prevent upcasting of float32
#     one = T.constant(1)

#     t = t_prev + 1
#     a_t = learning_rate*T.sqrt(one-beta2**t)/(one-beta1**t)

#     for param, g_t in zip(params, all_grads):
#         value = param.get_value(borrow=True)
#         m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
#                                broadcastable=param.broadcastable)
#         v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
#                                broadcastable=param.broadcastable)
#         v_hat_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
#                                    broadcastable=param.broadcastable)

#         m_t = beta1*m_prev + (one-beta1)*g_t
#         v_t = beta2*v_prev + (one-beta2)*g_t**2
#         v_hat_t = T.maximum(v_hat_prev, v_t)
#         step = a_t*m_t/(T.sqrt(v_hat_t) + epsilon)

#         updates[m_prev] = m_t
#         updates[v_prev] = v_t
#         updates[v_hat_prev] = v_hat_t
#         updates[param] = param - step

#     updates[t_prev] = t
#     return updates


# def clip(x):
#     return T.clip(x, 1e-7, 1-1e-7)


# def roll_the_dice(x_b, y_b, x_fake_b, batch_size, supervised=False, sample_y=None):
#     # x_fake_b = generator_fn(sample_Y(batch_size), sample_Z(batch_size, NU))
#     is_real = np.random.randint(0, 2, size=batch_size)
#     x_mixed = []
#     y_mixed = []
#     for i in range(batch_size):
#         if is_real[i]:
#             x_mixed.append(x_b[i])
#             # if supervised, we use real labels for real images
#             if supervised:
#                 y_mixed.append(y_b[i])
#         else:
#             x_mixed.append(x_fake_b[i])
#             if supervised:
#                 y_mixed.append(sample_y(1)[0])

#     x_mixed = np.array(x_mixed, dtype='float32')
#     is_real = is_real.reshape((-1, 1)).astype('float32')
#     return x_mixed, y_mixed, is_real


def onehoter(y, n_class):
    n_samp = len(y)
    if isinstance(y[0], list):
        y = [cur[0] for cur in y]
    oh = np.zeros((n_samp, n_class))
    oh[np.arange(n_samp), y] = 1
    return oh.astype('float32')


# def multihoter(y, n_class):
#     n_samp = len(y)
#     oh = np.zeros((n_samp, n_class))
#     idxs = [(samp_nr, elem) for samp_nr in range(n_samp) for elem in y[samp_nr]]
#     oh[zip(*idxs)] = 1
#     return oh.astype('float32')


# def select_model(model_path):
#     """ select model and train function """

#     model_str = os.path.basename(model_path)
#     model_str = model_str.split('.py')[0]
#     model = None
#     exec('from models import ' + model_str + ' as model')

#     return model, model_str


# def iterate_minibatches(sparse_data, batchsize, shuffle=False, k_samples=-1, bars_per_sample=NUM_BARS_DEFAULT,
#                         genre_list=None):
#     num_bars_per_file = [(file_idx, len(cur_file['patterns'])) for file_idx, cur_file in enumerate(sparse_data)]
#     num_samples_per_file = [(num_bars[0], int(num_bars[1] / bars_per_sample)) for num_bars in num_bars_per_file]

#     indices = [(f_idx, bar_idx * bars_per_sample)
#                for (f_idx, bar_num) in num_samples_per_file for bar_idx in range(bar_num)]
#     if shuffle:
#         np.random.shuffle(indices)

#     if k_samples < 0:
#         k_samples = len(indices)

#     for start_idx in range(0, k_samples - batchsize + 1, batchsize):
#         excerpt = indices[slice(start_idx, start_idx + batchsize)]

#         yield [get_patterns_as_numpy(sparse_data, sample[0], sample[1], bars_per_sample, genre_list=genre_list)
#                for sample in excerpt]


# def plot_samples(data, name, num_steps_per_bar=NOTES_PER_BAR, columns=2, pgf=False):
#     if not PLOTTING:
#         return
#     num_samples = len(data)
#     # num_bars_per_sample = len(data[0]) / num_steps_per_bar
#     num_insts, num_notes = data[0].shape
#     # num_bars_per_sample = num_notes / num_steps_per_bar

#     plt.figure(figsize=(COLUMN_WIDTH*2, COLUMN_WIDTH * 4))
#     for sample_idx in range(num_samples):
#         sample = data[sample_idx]
#         plt.subplot(int(np.ceil(num_samples/float(columns))), columns, sample_idx+1)
#         plt.imshow(sample, origin='lower', cmap='viridis')
#         x_range = range(0, num_notes, num_steps_per_bar/4)
#         plt.xticks([x-0.5 for x in x_range], [str(int(x/4)+1)+'.'+str(int(x % 4)+1) for x in range(len(x_range))])
#         # y_range = range(2, num_insts, 2)
#         # plt.yticks([y-0.5 for y in y_range], [drum_insts[y] for y in y_range])
#         y_range = range(num_insts)
#         plt.yticks([y for y in y_range], [drum_insts[y] for y in y_range])
#         plt.grid(b=None, axis='y')

#     f = plt.gcf()
#     f.tight_layout(pad=0, rect=(0.01, 0.01, 1, 1))
#     if name is None:
#         plt.show()
#     else:
#         if pgf:
#             f.savefig(name + '.pgf')
#         f.savefig(name + '.png')
#         plt.close()


# def render_samples(data, name, num_steps_per_bar=NOTES_PER_BAR):
#     data = np.asarray(data)
#     num_sampes, num_insts, total_steps = data.shape
#     nonzeros = np.transpose(np.nonzero(data > THRESHOLD))
#     num_bars = total_steps / num_steps_per_bar
#     for track_idx in range(num_sampes):
#         onsets = nonzeros[nonzeros[:, 0] == track_idx][:, 1:]
#         patterns = np.asarray(onsets[:, [1, 0]], dtype='float32')
#         patterns[:, 0] = patterns[:, 0] / 32.0
#         render_drum_pattern(patterns, RENDER_TEMPO, num_bars=num_bars, output_file=name+'_'+str(track_idx))


# def generate_samples_for_each_class(CAT_SIZE, CONT_SIZE, NOISE_SIZE, num_samples_generate, timestamp, gen_fn, save_plots, render_wav):

#     noise_c = np.repeat(np.asarray([0.5, 0.5])[:, None], num_samples_generate, 1).T.astype('float32')
#     #  np.random.rand(CAT_SIZE, CONT_SIZE).astype('float32')  # TODO we can set these later (CONT_SIZE)

#     for per_genre_ix in range(CAT_SIZE):
#         category = np.zeros(CAT_SIZE)
#         category[per_genre_ix] = 1
#         noise_y = np.repeat(category[:, None], num_samples_generate, 1).T.astype('float32')
#         noise_z = np.random.rand(num_samples_generate, NOISE_SIZE).astype('float32')

#         samples = gen_fn(noise_z, noise_c, noise_y)
#         np.save(os.path.join(SAMPLES_PATH, timestamp, 'samples_{}.npy'.format(per_genre_ix)), samples)
#         if save_plots:
#             plot_samples(samples, os.path.join(SAMPLES_PATH, timestamp, 'samples_{}'.format(per_genre_ix)))
#         if render_wav:
#             render_samples(samples, os.path.join(SAMPLES_PATH, timestamp, 'samples_{}'.format(per_genre_ix)))


# def create_dir_and_write_params(timestamp, param_dict):
#     exp_folder = os.path.join(SAMPLES_PATH, timestamp)
#     if not os.path.exists(exp_folder):
#         os.makedirs(exp_folder)
#         copy2(os.path.join(PROJECT_PATH, param_dict['code']+'.py'), exp_folder)
#         copy2(os.path.join(PROJECT_PATH, 'models', param_dict['model']+'.py'), exp_folder)
#         with open(os.path.join(exp_folder, param_dict['code']+'_settings.txt'), 'w') as outfile:
#             for key in param_dict:
#                 outfile.write(key + ': ' + str(param_dict[key]) + '\n')


# def get_default_model(num_bars):
#     if num_bars == 1:
#         default_model = 'models/conv_v2.py'
#     elif num_bars == 4:
#         default_model = 'models/conv_v1.py'
#     else:
#         default_model = 'models/dense.py'

#     return default_model


# def get_timestamp():
#     return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')


# def map_genre(genre_idx):
#     # map genre index into points on a circle with radius 1
#
#     assert all(genre_idx < BP_GENRE_COUNT)
#     angle = genre_idx * 2.0 * np.pi / BP_GENRE_COUNT
#
#     genre_x = np.sin(angle)
#     genre_y = np.cos(angle)
#     return genre_x, genre_y
