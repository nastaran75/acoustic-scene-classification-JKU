import numpy as np
import theano
import theano.tensor as T
import lasagne
import cPickle, gzip
import os
import random

# height = 128
# width = 431
# length = 220501
# channel = 1
directory = '../dataset/'

# def load_testset(dataset_name,fold_number,division,representation,augment):

# 	shifts = []
# 	# print len(shifts)
# 	if augment:
# 		num_data = (len(shifts)+2)*division
# 	else:
# 		num_data=division

# 	def add_noise(data):
# 		# noise = np.random.randn(len(data))
# 		noise = np.random.normal(0.0,1.0,len(data))
# 		data_noise = data + noise
# 		return data_noise

# 	def shift(data,val):
# 		return np.roll(data, val)

# 	#load the images
# 	def load_images(image_filename,labels_filename):
# 		if not os.path.exists(image_filename):
# 			print filename + ' not exists'

# 		if not os.path.exists(labels_filename):
# 			print filename + ' not exists'

# 		data = np.load(image_filename)
# 		data_labels = np.load(labels_filename)
# 		length = data.shape[1]
# 		# print length
# 		triple_length = length/division
# 		triple = np.empty([data.shape[0]*num_data,triple_length])
# 		labels = np.empty([data.shape[0]*num_data])
# 		counter_triple = 0
# 		for i,d in enumerate(data):
# 			for cnt in range(division):
# 				left = d[cnt*triple_length:(cnt+1)*triple_length][:,0]
# 				right = d[cnt*triple_length:(cnt+1)*triple_length][:,1]
# 				if representation=='Left':
# 					triple[counter_triple] = left
					

# 				elif representation=='Right':
# 					triple[counter_triple] = right


# 				elif representation=='Mid':
# 					triple[counter_triple] = np.mean(d[cnt*triple_length:(cnt+1)*triple_length],axis=1)
 

# 				elif representation=='Side':
# 					triple[counter_triple] = np.subtract(left,right)
					

# 				else:
# 					print '----------------not a valid representation----------------'


# 				labels[counter_triple] = data_labels[i]
# 				wave = triple[counter_triple]
# 				counter_triple += 1 

# 				if(augment==True):
# 					#add noise to the wave
# 					triple[counter_triple] = add_noise(wave)
# 					labels[counter_triple] = data_labels[i]
# 					counter_triple += 1 

# 					#shift the left channel
# 					for val in shifts:
# 						triple[counter_triple] = shift(wave,val)
# 						labels[counter_triple] = data_labels[i]
# 						counter_triple += 1

# 		normalized = (triple - np.mean(triple)) / np.std(triple)
# 		normalized = np.expand_dims(normalized,axis=1)
# 		labels = labels.astype(np.uint8)
# 		return normalized,labels

# 	X_test,y_test = load_images(directory + dataset_name + '_wave_meta_images.npy',
# 		directory + dataset_name + '_wave_meta_labels.npy')
# 	# y_test = load_labels(directory + dataset_name + '_wave_meta_labels.npy')

# 	return X_test, y_test, num_data

# def load_valset(dataset_name,fold_number,division,representation,augment):

# 	shifts = []
# 	if augment:
# 		num_data = (len(shifts)+2)*division
# 	else:
# 		num_data=division

# 	def add_noise(data):
# 		noise = np.random.randn(len(data))
# 		data_noise = data + 0.005 * noise
# 		return data_noise

# 	def shift(data,val):
# 		return np.roll(data, val)

# 	#load the images
# 	def load_images(image_filename,labels_filename):
# 		if not os.path.exists(image_filename):
# 			print filename + ' not exists'

# 		if not os.path.exists(labels_filename):
# 			print filename + ' not exists'

# 		data = np.load(image_filename)
# 		data_labels = np.load(labels_filename)
# 		length = data.shape[1]
# 		# print length
# 		triple_length = length/division
# 		triple = np.empty([data.shape[0]*num_data,triple_length])
# 		labels = np.empty([data.shape[0]*num_data])
# 		counter_triple = 0
# 		for i,d in enumerate(data):
# 			for cnt in range(division):
# 				left = d[cnt*triple_length:(cnt+1)*triple_length][:,0]
# 				right = d[cnt*triple_length:(cnt+1)*triple_length][:,1]
# 				if representation=='Left':
# 					triple[counter_triple] = left
					

# 				elif representation=='Right':
# 					triple[counter_triple] = right


# 				elif representation=='Mid':
# 					triple[counter_triple] = np.mean(d[cnt*triple_length:(cnt+1)*triple_length],axis=1)
 

# 				elif representation=='Side':
# 					triple[counter_triple] = np.subtract(left,right)
					

# 				else:
# 					print '----------------not a valid representation----------------'


# 				labels[counter_triple] = data_labels[i]
# 				wave = triple[counter_triple]
# 				counter_triple += 1 

# 				if(augment==True):
# 					#add noise to the wave
# 					triple[counter_triple] = add_noise(wave)
# 					labels[counter_triple] = data_labels[i]
# 					counter_triple += 1 

# 					#shift the left channel
# 					for val in shifts:
# 						triple[counter_triple] = shift(wave,val)
# 						labels[counter_triple] = data_labels[i]
# 						counter_triple += 1

# 		normalized = (triple - np.mean(triple)) / np.std(triple)
# 		normalized = np.expand_dims(normalized,axis=1)
# 		labels = labels.astype(np.uint8)
# 		return normalized,labels

# 	# #load the labels
# 	# def load_labels(filename):
# 	# 	if not os.path.exists(filename):
# 	# 		print filename + ' not exists'
# 	# 	data = np.load(filename)
# 	# 	triple = np.empty([data.shape[0]*division])
# 	# 	counter_triple = 0
# 	# 	for d in data:
# 	# 		for cnt in range(division):
# 	# 			triple[counter_triple] = d
# 	# 			counter_triple += 1
			


# 	# 	triple = triple.astype(np.uint8)
# 	# 	return triple

# 	# X_train,y_train = load_images(directory + dataset_name + '_wave_fold' + str(fold_number) + '_train_images.npy',
# 	# 	directory + dataset_name + '_wave_fold' + str(fold_number) + '_train_labels.npy')
# 	# y_train = load_labels(directory + dataset_name + '_wave_fold' + str(fold_number) + '_train_labels.npy')
# 	X_val, y_val = load_images(directory + dataset_name + '_wave_fold' + str(fold_number) + '_evaluate_images.npy',
# 		directory + dataset_name + '_wave_fold' + str(fold_number) + '_evaluate_labels.npy')
# 	# y_val = load_labels(directory + dataset_name + '_wave_fold' + str(fold_number) + '_evaluate_labels.npy')
# 	# X_test,y_test = load_images(directory + dataset_name + '_wave_meta_images.npy',
# 	# 	directory + dataset_name + '_wave_meta_labels.npy')
# 	# y_test = load_labels(directory + dataset_name + '_wave_meta_labels.npy')


# 	return X_val, y_val, num_data

def load_dataset(dataset_name,fold_number,division,representation,augment):

	shifts = []
	if augment:
		num_data = (len(shifts)+2)*division
	else:
		num_data=division

	def add_noise(data):
		noise = np.random.randn(len(data))
		data_noise = data + 0.005 * noise
		return data_noise

	def shift(data,val):
		return np.roll(data, val)

	#load the images
	def load_images(image_filename,labels_filename):
		if not os.path.exists(image_filename):
			print filename + ' not exists'

		if not os.path.exists(labels_filename):
			print filename + ' not exists'

		data = np.load(image_filename)
		data_labels = np.load(labels_filename)
		length = data.shape[1]
		# print length
		triple_length = length/division
		triple = np.empty([data.shape[0]*num_data,triple_length])
		labels = np.empty([data.shape[0]*num_data])
		counter_triple = 0
		for i,d in enumerate(data):
			for cnt in range(division):
				left = d[cnt*triple_length:(cnt+1)*triple_length][:,0]
				right = d[cnt*triple_length:(cnt+1)*triple_length][:,1]
				if representation=='Left':
					triple[counter_triple] = left
					

				elif representation=='Right':
					triple[counter_triple] = right


				elif representation=='Mid':
					triple[counter_triple] = np.mean(d[cnt*triple_length:(cnt+1)*triple_length],axis=1)
 

				elif representation=='Side':
					triple[counter_triple] = np.subtract(left,right)
					

				else:
					print '----------------not a valid representation----------------'


				labels[counter_triple] = data_labels[i]
				wave = triple[counter_triple]
				counter_triple += 1 

				if(augment==True):
					#add noise to the wave
					triple[counter_triple] = add_noise(wave)
					labels[counter_triple] = data_labels[i]
					counter_triple += 1 

					#shift the left channel
					for val in shifts:
						triple[counter_triple] = shift(wave,val)
						labels[counter_triple] = data_labels[i]
						counter_triple += 1

		normalized = (triple - np.mean(triple)) / np.std(triple)
		normalized = np.expand_dims(normalized,axis=1)
		labels = labels.astype(np.uint8)
		return normalized,labels

	# #load the labels
	# def load_labels(filename):
	# 	if not os.path.exists(filename):
	# 		print filename + ' not exists'
	# 	data = np.load(filename)
	# 	triple = np.empty([data.shape[0]*division])
	# 	counter_triple = 0
	# 	for d in data:
	# 		for cnt in range(division):
	# 			triple[counter_triple] = d
	# 			counter_triple += 1
			


	# 	triple = triple.astype(np.uint8)
	# 	return triple

	X_train,y_train = load_images(directory + dataset_name + '_wave_fold' + str(fold_number) + '_train_images.npy',
		directory + dataset_name + '_wave_fold' + str(fold_number) + '_train_labels.npy')
	# y_train = load_labels(directory + dataset_name + '_wave_fold' + str(fold_number) + '_train_labels.npy')
	X_val, y_val = load_images(directory + dataset_name + '_wave_fold' + str(fold_number) + '_evaluate_images.npy',
		directory + dataset_name + '_wave_fold' + str(fold_number) + '_evaluate_labels.npy')
	# # y_val = load_labels(directory + dataset_name + '_wave_fold' + str(fold_number) + '_evaluate_labels.npy')
	X_test,y_test = load_images(directory + dataset_name + '_wave_meta_images.npy',
		directory + dataset_name + '_wave_meta_labels.npy')
	# # y_test = load_labels(directory + dataset_name + '_wave_meta_labels.npy')


	return X_train, y_train, X_val, y_val, X_test, y_test, num_data


def iterate_minibatches(inputs, targets, batch_size, shuffle = False):

	indices = np.arange(len(inputs))
	if shuffle:
		np.random.shuffle(indices)
	
	for start_index in range(0, len(inputs) - batch_size + 1 , batch_size):
		if shuffle:
			excerpt = indices[start_index: start_index + batch_size]
		else:
			excerpt = slice(start_index,start_index + batch_size)

		yield inputs[excerpt], targets[excerpt], indices[start_index: start_index + batch_size]

def augment_minibatches(minibatches, noise=0.5, trans=4):
    """
    Randomly augments images by horizontal flipping with a probability of
    `flip` and random translation of up to `trans` pixels in both directions.
    """
    for inputs, targets, indices in minibatches:
        batchsize, c, h= inputs.shape
        # print batchsize,c,h
        if noise:
        	noise_vector = np.random.standard_normal(size=(batchsize,c,h))
        	coins = np.random.rand(batchsize) < noise
        	inputs = [inp + 0.005*noise_v if coin else inp for inp, coin, noise_v in zip(inputs, coins, noise_vector)]
        	if not trans:
        		inputs = np.asarray(inputs)
        outputs = inputs
        if trans:
        	outputs = [np.roll(inp,np.random.randint(0,h)) for inp in inputs]

        #     outputs = np.empty((batchsize, c, h), inputs[0].dtype)
        #     shifts = np.random.randint(0, h, (batchsize, 1))
        #     for outp, inp, x in zip(outputs, inputs, shifts):
        #     	outp[:] = np.roll(inp,x)
                # if x > 0:
                #     outp[:, :x] = 0
                #     outp = outp[:, x:]
                #     inp = inp[:, :-x]
                # elif x < 0:
                #     outp[:, x:] = 0
                #     outp = outp[:, :x]
                #     inp = inp[:, -x:]
                # if y > 0:
                #     outp[:, :, :y] = 0
                #     outp = outp[:, :, y:]
                #     inp = inp[:, :, :-y]
                # elif y < 0:
                #     outp[:, :, y:] = 0
                #     outp = outp[:, :, :y]
                #     inp = inp[:, :, -y:]
                # outp[:] = inp
        yield outputs, targets, indices