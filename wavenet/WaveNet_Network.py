import theano
import lasagne
from lasagne.layers import InputLayer, DenseLayer, batch_norm
import numpy
import lib
import theano.tensor as T

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng = RandomStreams(seed=4884)


DIM = 64 # Model dimensionality.
N_CHANNELS = 1
DEPTH = 1 
DILATION_LEVEL = 4 

def relu(x):
    # Using T.nnet.relu gives me NaNs. No idea why.
	return T.switch(x > lib.floatX(0), x, lib.floatX(0))

def Softmax(output):
    # output.shape: (batch size, length, output_dim, 256)
    softmax_output = T.nnet.softmax(output.reshape((-1,output.shape[output.ndim-1])))
    return softmax_output.reshape(output.shape)

def sample_from_softmax(softmax_var):
    #softmax_var assumed to be of shape (batch_size, num_classes)
    old_shape = softmax_var.shape

    softmax_var_reshaped = softmax_var.reshape((-1,softmax_var.shape[softmax_var.ndim-1]))

    return T.argmax(
        T.cast(
            srng.multinomial(pvals=softmax_var_reshaped),
            theano.config.floatX
            ).reshape(old_shape),
        axis = softmax_var.ndim-1
)



def DilatedConv1D(name,input_dim, output_dim, filter_size, inputs, dilation, mask_type=None, apply_biases=True):
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

def DilatedConvBlock(name,input_dim, output_dim, filter_size, inputs):
    result = inputs
    for i in xrange(DILATION_LEVEL):
        d = numpy.left_shift(2, i)
        result = DilatedConv1D(name+'Dilation'+str(d), DIM, DIM, 5, result, d, mask_type='b')
        result = relu(result)
	return result

def build_cnn(input_var = None, division=1):

	# length = data_len
	# channel = 1
	# dropout_ratio = .5
	# filter_size = 3
	# pool_length = 3
	# stride = 1
	# dropout_internal = .5

	output = DilatedConv1D('InputConv',N_CHANNELS, DIM, 5, input_var, 1, mask_type='a')

	for i in xrange(DEPTH):
		output = DilatedConvBlock('DilatedConvBlock'+str(i),DIM, DIM, 5, output)
		output = relu(output)
	
	# dropout1 = lasagne.layers.dropout(conv15, p=dropout_ratio)
	output = DilatedConv1D('OutputConv1',DIM, DIM, 1, output, 1, mask_type='b')
	output = relu(output)

	output = DilatedConv1D('OutputConv2',DIM, DIM, 1, output, 1, mask_type='b')
	output = relu(output)

	output = DilatedConv1D('OutputConv3',DIM, 15, 1, output, 1, mask_type='b')
	print output.shape

	####################################################################

	# flattened = lasagne.layers.FlattenLayer(output)

	#####################################################################

	# output = lasagne.layers.DenseLayer(output, num_units=15, nonlinearity=lasagne.nonlinearities.softmax)
	output = Softmax(output.reshape((output.shape[0], output.shape[1], 1, output.shape[2])))
	# output = sample_from_softmax(output)

	return output