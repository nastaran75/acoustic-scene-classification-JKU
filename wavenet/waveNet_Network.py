import theano
import lasagne
from lasagne.layers import InputLayer, DenseLayer, batch_norm

def DilatedConv1D(name, input_dim, output_dim, filter_size, inputs, dilation, mask_type=None, apply_biases=True):
    """
    inputs.shape: (batch size, length, input_dim)
    mask_type: None, 'a', 'b'
    output.shape: (batch size, length, output_dim)
    """
    def uniform(stdev, size):
        """uniform distribution with the given stdev and size"""
        return numpy.random.uniform(
            low=-stdev * numpy.sqrt(3),
            high=stdev * numpy.sqrt(3),
            size=size
        ).astype(theano.config.floatX)

    filters_init = uniform(
        1./numpy.sqrt(input_dim * filter_size),
        # output dim, input dim, height, width
        (output_dim, input_dim, filter_size, 1)
    )

    if mask_type is not None:
        filters_init *= lib.floatX(numpy.sqrt(2.))

    filters = lib.param(
        name+'.Filters',
        filters_init
    )

    if mask_type is not None:
        mask = numpy.ones(
            (output_dim, input_dim, filter_size, 1),
            dtype=theano.config.floatX
        )

        center = filter_size//2
        for i in xrange(filter_size):
            if (i > center):
                mask[:, :, i, :] = 0.
            # if (mask_type=='a' and i == center):
            #     mask[:, :, center] = 0.
        filters = filters * mask

    inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], 1, inputs.shape[2]))
    # conv2d takes inputs as (batch size, input channels, height[?], width[?])
    inputs = inputs.dimshuffle(0, 3, 1, 2)
    result = T.nnet.conv2d(inputs, filters, border_mode='half', filter_flip=False, filter_dilation=(dilation, 1))

    if apply_biases:
        biases = lib.param(
            name+'.Biases',
            numpy.zeros(output_dim, dtype=theano.config.floatX)
        )
        result = result + biases[None, :, None, None]

    result = result.dimshuffle(0, 2, 3, 1)
    return result.reshape((result.shape[0], result.shape[1], result.shape[3]))

def build_cnn(data_len,input_var = None, division=1):

	length = data_len
	channel = 1
	dropout_ratio = .5
	filter_size = 3
	pool_length = 3
	stride = 1
	dropout_internal = .5

	# input
	my_input = lasagne.layers.InputLayer(shape=(None, channel, length), input_var=input_var)

	################################################################
	#0
	conv0 = batch_norm(lasagne.layers.DilatedConv1D(my_input, num_filters=128, filter_size=3,
	 nonlinearity=lasagne.nonlinearities.rectify,pad='valid', stride=3, W=lasagne.init.HeNormal()))
	###########################################################################
	#1
	conv1 = batch_norm(lasagne.layers.DilatedConv1D(conv0, num_filters=128, filter_size=filter_size,
	nonlinearity=lasagne.nonlinearities.rectify, pad='same', stride=stride, W=lasagne.init.HeNormal()))	

	# max pool
	pool1 = lasagne.layers.MaxPool1DLayer(conv1, pool_size=pool_length)

	# pool1 = lasagne.layers.GaussianNoiseLayer(pool1, sigma=0.1)

	# pool1 = lasagne.layers.dropout(pool1, p=dropout_internal)

	################################################################

	#2
	conv2 = batch_norm(lasagne.layers.DilatedConv1D(pool1, num_filters=128, filter_size=filter_size,
	nonlinearity=lasagne.nonlinearities.rectify, pad='same', stride=stride, W=lasagne.init.HeNormal()))

	# max pool
	pool2 = lasagne.layers.MaxPool1DLayer(conv2, pool_size=pool_length)

	# pool2 = lasagne.layers.GaussianNoiseLayer(pool2, sigma=0.1)

	################################################################

	#3
	conv3 = batch_norm(lasagne.layers.DilatedConv1D(pool2, num_filters=256, filter_size=filter_size,
	nonlinearity=lasagne.nonlinearities.rectify, pad='same', stride=stride, W=lasagne.init.HeNormal()))

	# max pool
	pool3 = lasagne.layers.MaxPool1DLayer(conv3, pool_size=pool_length)

	# pool3 = lasagne.layers.GaussianNoiseLayer(pool3, sigma=0.1)

	# pool3 = lasagne.layers.dropout(pool3, p=dropout_internal)

	####################################################################

	#4
	conv4 = batch_norm(lasagne.layers.DilatedConv1D(pool3, num_filters=256, filter_size=filter_size,
	nonlinearity=lasagne.nonlinearities.rectify, pad='same', stride=stride, W=lasagne.init.HeNormal()))

	# max pool
	pool4 = lasagne.layers.MaxPool1DLayer(conv4, pool_size=pool_length)

	# pool4 = lasagne.layers.GaussianNoiseLayer(pool4, sigma=0.1)

	####################################################################

	#5
	conv5 = batch_norm(lasagne.layers.DilatedConv1D(pool4, num_filters=256, filter_size=filter_size,
	nonlinearity=lasagne.nonlinearities.rectify, pad='same', stride=stride, W=lasagne.init.HeNormal()))

	# max pool
	pool5 = lasagne.layers.MaxPool1DLayer(conv5, pool_size=pool_length)

	# pool5 = lasagne.layers.GaussianNoiseLayer(pool5, sigma=0.1)

	# pool5 = lasagne.layers.dropout(pool5, p=dropout_internal)

	####################################################################

	#6
	conv6 = batch_norm(lasagne.layers.DilatedConv1D(pool5, num_filters=256, filter_size=filter_size,
	nonlinearity=lasagne.nonlinearities.rectify, pad='same', stride=stride, W=lasagne.init.HeNormal()))

	# max pool
	pool6 = lasagne.layers.MaxPool1DLayer(conv6, pool_size=pool_length)

	# pool6 = lasagne.layers.GaussianNoiseLayer(pool6, sigma=0.1)

	####################################################################

	#7
	conv7 = batch_norm(lasagne.layers.DilatedConv1D(pool6, num_filters=256, filter_size=filter_size,
	nonlinearity=lasagne.nonlinearities.rectify, pad='same', stride=stride, W=lasagne.init.HeNormal()))

	# max pool
	pool7 = lasagne.layers.MaxPool1DLayer(conv7, pool_size=pool_length)

	# pool7 = lasagne.layers.GaussianNoiseLayer(pool7, sigma=0.1)

	# pool7 = lasagne.layers.dropout(pool7, p=dropout_internal)

	####################################################################

	#8
	conv8 = batch_norm(lasagne.layers.DilatedConv1D(pool7, num_filters=256, filter_size=filter_size,
	nonlinearity=lasagne.nonlinearities.rectify, pad='same', stride=stride, W=lasagne.init.HeNormal()))

	# max pool
	pool8 = lasagne.layers.MaxPool1DLayer(conv8, pool_size=pool_length)

	# pool8 = lasagne.layers.GaussianNoiseLayer(pool8, sigma=0.1)

	####################################################################

	#9
	conv9 = batch_norm(lasagne.layers.DilatedConv1D(pool8, num_filters=512, filter_size=filter_size,
	nonlinearity=lasagne.nonlinearities.rectify, pad='same', stride=stride, W=lasagne.init.HeNormal()))

	# max pool
	pool9 = lasagne.layers.MaxPool1DLayer(conv9, pool_size=pool_length)

	# pool9 = lasagne.layers.GaussianNoiseLayer(pool9, sigma=0.1)

	####################################################################

	# #10
	# conv10 = batch_norm(lasagne.layers.Conv1DLayer(pool9, num_filters=256, filter_size=filter_size,
	# nonlinearity=lasagne.nonlinearities.rectify, pad='same', stride=1, W=lasagne.init.HeNormal()))

	# # max pool
	# pool10 = lasagne.layers.MaxPool1DLayer(conv10, pool_size=pool_length)

	# ####################################################################
	# #11
	# conv11 = batch_norm(lasagne.layers.Conv1DLayer(pool10, num_filters=256, filter_size=filter_size,
	# nonlinearity=lasagne.nonlinearities.rectify, pad='same', stride=1, W=lasagne.init.HeNormal()))

	# # max pool
	# pool11 = lasagne.layers.MaxPool1DLayer(conv11, pool_size=pool_length)

	####################################################################
	# #13
	# conv13 = batch_norm(lasagne.layers.Conv1DLayer(pool9, num_filters=256, filter_size=filter_size,
	# nonlinearity=lasagne.nonlinearities.rectify, pad='same', stride=1, W=lasagne.init.HeNormal()))

	# # max pool
	# pool13 = lasagne.layers.MaxPool1DLayer(conv13, pool_size=pool_length)
	# #############################################################################

	# #14
	# conv14 = batch_norm(lasagne.layers.Conv1DLayer(pool9, num_filters=512, filter_size=filter_size,
	# nonlinearity=lasagne.nonlinearities.rectify, pad='same', stride=1, W=lasagne.init.HeNormal()))

	# # max pool
	# pool14 = lasagne.layers.MaxPool1DLayer(conv14, pool_size=pool_length)
	# ######################################################################

	#15
	conv15 = batch_norm(lasagne.layers.DilatedConv1D(pool9, num_filters=512, filter_size=1,
	nonlinearity=lasagne.nonlinearities.rectify, pad='same', stride=stride, W=lasagne.init.HeNormal()))

	dropout1 = lasagne.layers.dropout(conv15, p=dropout_ratio)

	####################################################################

	flattened = lasagne.layers.FlattenLayer(dropout1)

	#####################################################################

	output = lasagne.layers.DenseLayer(flattened, num_units=15, nonlinearity=lasagne.nonlinearities.softmax)

	return output