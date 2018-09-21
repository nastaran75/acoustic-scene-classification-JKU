import numpy as np
import theano
import theano.tensor as T
import lasagne
import cPickle, gzip
import os
import random
import scipy.io.wavfile
import helpers
# import scikits.audiolab

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

def load_dataset(dataset_name,fold_number,division,representation,random_sample=False):

	# shifts = []
	# if augment:
	# 	num_data = (len(shifts)+2)*division
	# else:
	# 	num_data=division
	num_data = division
	num_random_samples=15
	# def add_noise(data):
	# 	noise = np.random.randn(len(data))
	# 	data_noise = data + 0.005 * noise
	# 	return data_noise

	# def shift(data,val):
	# 	return np.roll(data, val)

	#load the images
	def load_images(image_filename,labels_filename,random_sample=False):
		if not os.path.exists(image_filename):
			print filename + ' not exists'

		if not os.path.exists(labels_filename):
			print filename + ' not exists'

		data = np.load(image_filename)
		# print data.shape
		new_data = np.zeros(shape=(data.shape[0],data.shape[1]))
		for i,d in enumerate(data):
			new_data[i] = np.mean(d,axis=1)
		data_labels = np.load(labels_filename)
		# length = data.shape[1]
		# # print length
		# triple_length = length/division
		# triple = np.empty([data.shape[0]*num_data,triple_length])
		# labels = np.empty([data.shape[0]*num_data])
		# counter_triple = 0

		# if not random_sample:
		# 	for i,d in enumerate(data):
		# 		for cnt in range(division):
		# 			left = d[cnt*triple_length:(cnt+1)*triple_length][:,0]
		# 			right = d[cnt*triple_length:(cnt+1)*triple_length][:,1]
		# 			if representation=='Left':
		# 				triple[counter_triple] = left
						

		# 			elif representation=='Right':
		# 				triple[counter_triple] = right


		# 			elif representation=='Mid':
		# 				triple[counter_triple] = np.mean(d[cnt*triple_length:(cnt+1)*triple_length],axis=1)
	 

		# 			elif representation=='Side':
		# 				triple[counter_triple] = np.subtract(left,right)
						

		# 			else:
		# 				print '----------------not a valid representation----------------'


		# 			labels[counter_triple] = data_labels[i]
		# 			wave = triple[counter_triple]
		# 			counter_triple += 1 

		# 		# if(augment==True):
		# 		# 	#add noise to the wave
		# 		# 	triple[counter_triple] = add_noise(wave)
		# 		# 	labels[counter_triple] = data_labels[i]
		# 		# 	counter_triple += 1 

		# 		# 	#shift the left channel
		# 		# 	for val in shifts:
		# 		# 		triple[counter_triple] = shift(wave,val)
		# 		# 		labels[counter_triple] = data_labels[i]
		# 		# 		counter_triple += 1
		# else:
			
		# 	for i,d in enumerate(data):
		# 		for cnt in range(num_random_samples):
		# 			rnd = np.random.randint(0,length-triple_length)
		# 			left = d[rnd:rnd+triple_length][:,0]
		# 			right = d[rnd:rnd+triple_length][:,1]
		# 			if representation=='Left':
		# 				triple[counter_triple] = left
						

		# 			elif representation=='Right':
		# 				triple[counter_triple] = right


		# 			elif representation=='Mid':
		# 				triple[counter_triple] = np.mean(d[rnd:rnd+triple_length],axis=1)
	 

		# 			elif representation=='Side':
		# 				triple[counter_triple] = np.subtract(left,right)
						

		# 			else:
		# 				print '----------------not a valid representation----------------'


		# 			labels[counter_triple] = data_labels[i]
		# 			wave = triple[counter_triple]
		# 			counter_triple += 1 




		normalized = (new_data - np.mean(new_data)) / np.std(new_data)
		# normalized = triple
		# normalized = np.expand_dims(normalized,axis=1)
		# normalized = np.expand_dims(normalized,axis=1)
		data_labels = data_labels.astype(np.uint8)
		return normalized,data_labels

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
		directory + dataset_name + '_wave_fold' + str(fold_number) + '_train_labels.npy',random_sample=False)
	# y_train = load_labels(directory + dataset_name + '_wave_fold' + str(fold_number) + '_train_labels.npy')
	X_val, y_val = load_images(directory + dataset_name + '_wave_fold' + str(fold_number) + '_evaluate_images.npy',
		directory + dataset_name + '_wave_fold' + str(fold_number) + '_evaluate_labels.npy',random_sample=False)
	# # y_val = load_labels(directory + dataset_name + '_wave_fold' + str(fold_number) + '_evaluate_labels.npy')
	X_test,y_test = load_images(directory + dataset_name + '_wave_meta_images.npy',
		directory + dataset_name + '_wave_meta_labels.npy',random_sample=False)
	# # y_test = load_labels(directory + dataset_name + '_wave_meta_labels.npy')

	if random_sample:
		num_data = num_random_samples
	return X_train, y_train, X_val, y_val, X_test, y_test, num_data


def iterate_minibatches(inputs, targets, batch_size,num_samples,division, shuffle = False,noise=False,mixup=False,val=False):
	# print inputs.shape
	# mixup_length = 0
	# if mixup:
	# 	mixup_length = 2
	# mixup_length = 2000
	new_inputs = np.zeros(shape=(inputs.shape[0]*(num_samples),inputs.shape[1]/division))
	new_targets = np.zeros(targets.shape[0]*(num_samples),dtype=np.uint8)
	new_counter = 0
	length = inputs.shape[1]
	divided_length=length/division
	if not val:
		for i,d in enumerate(inputs):
			for cnt in range(num_samples):
				rnd = np.random.randint(0,length - divided_length)
				new_inputs[new_counter] = d[rnd:rnd+divided_length]
				new_targets[new_counter] = targets[i]
				new_counter += 1	

	if val:
		for i,d in enumerate(inputs):
			for cnt in range(num_samples):
				# rnd = np.random.randint(0,length - divided_length)
				new_inputs[new_counter] = d[cnt*divided_length:cnt*divided_length+divided_length]
				new_targets[new_counter] = targets[i]
				new_counter += 1
	

	new_targets = helpers.onehoter(new_targets,15)
	# print new_targets
	
	
	if mixup:
		# print 'going to mixup...'
		for i in range(inputs.shape[0]*3):
			# print i
			rnd1 = np.random.randint(0,new_inputs.shape[0])
			rnd2 = np.random.randint(0,new_inputs.shape[0])
			mul = np.random.beta(0.5,0.5)
			# print mul
			new_inputs[rnd2] = np.sum(np.array([new_inputs[rnd1]*mul, new_inputs[rnd2]*(1.0-mul)]), axis=0)
			new_targets[rnd2] = np.sum(np.array([new_targets[rnd1]*mul, new_targets[rnd2]*(1.0-mul)]), axis=0)
			# print new_targets[rnd1]*mul
			# print new_targets[rnd2]*(1.0-mul)
			# print new_targets[i]

	if noise:
		for d in new_inputs:
			if np.random.rand()<0.5:
				noise_vector = np.random.standard_normal(size=d.shape)
				d += 0.005*noise_vector

	new_inputs = np.expand_dims(new_inputs,axis=1)
	new_inputs = np.expand_dims(new_inputs,axis=1)
	# print new_inputs.shape
	indices = np.arange(len(new_inputs))
	if shuffle:
		np.random.shuffle(indices)

	for start_index in range(0, len(new_inputs) - batch_size + 1 , batch_size):
		if shuffle:
			excerpt = indices[start_index: start_index + batch_size]
		else:
			excerpt = slice(start_index,start_index + batch_size)

		yield new_inputs[excerpt], new_targets[excerpt], indices[start_index: start_index + batch_size]

def augment_minibatches(minibatches, noise=0.5, cyclic_shift=0.3, random_shift=0):
    """
    Randomly augments images by horizontal flipping with a probability of
    `flip` and random translation of up to `trans` pixels in both directions.
    """
    for inputs, targets, indices in minibatches:
        batchsize, c, h= inputs.shape
        # scipy.io.wavfile.write('before_noise.wav',22050,inputs[0].T)
        # print batchsize,c,h
        # noise checked :D
        if noise:
        	noise_vector = np.random.standard_normal(size=(batchsize,c,h))
        	coins = np.random.rand(batchsize) < noise
        	inputs = [inp + 0.005*noise_v if coin else inp for inp, coin, noise_v in zip(inputs, coins, noise_vector)]
        	if not (cyclic_shift or random_shift):
        		inputs = np.asarray(inputs)
        outputs = inputs

        ### shifting checked :D
        if cyclic_shift:
        	coins = np.random.rand(batchsize) < cyclic_shift
        	for out,inp,coin in zip(outputs,inputs,coins):
        		if coin:
        		# print inp.shape
	        		shift=np.random.randint(0,h)
	        		out = np.roll(inp,shift=shift,axis=1)


        # apply random shift
        if random_shift:
        	coins = np.random.rand(batchsize) < random_shift
	        X_new = outputs
	        n_slices = 5
	        slice_length = h // n_slices
	        borders = np.arange(0, n_slices) * slice_length
	        for rnd,out,coin in zip(X_new,outputs,coins):
	        	if coin:
		        	np.random.shuffle(borders)
		        	for j, b in enumerate(borders):
		        		s = j * slice_length
		        		e = s + slice_length
		        		rnd[0, s:e] = out[0, b:b+slice_length]
		        	# scipy.io.wavfile.write('after_random_shift.wav',22050,rnd.T)
		        # break
			outputs = X_new
	
        yield outputs, targets, indices