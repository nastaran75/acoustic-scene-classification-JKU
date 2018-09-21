import numpy as np
import theano
import theano.tensor as T
import lasagne
import cPickle, gzip
import os


# height = 128
# width = 431
# length = 220501
channel = 2
directory = 'dataset/'


# def load_testset(division=1):

	#load the images
	# def load_images(filename):
	# 	if not os.path.exists(filename):
	# 		print filename + ' not exists'

	# 	data = np.load(filename)
	# 	triple_length = length/division
	# 	triple = np.empty([data.shape[0]*division,triple_length])
	# 	counter_triple = 0
	# 	for d in data:
	# 		for cnt in range(division):
	# 			triple[counter_triple] = d[cnt*triple_length:(cnt+1)*triple_length]
	# 			counter_triple += 1 



	# 	triple = np.expand_dims(triple,axis=1)
	# 	return triple
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

	# X_test = load_images(directory + 'test_wave_images.npy')
	# y_test = load_labels(directory + 'test_wave_labels.npy')

	# return X_test, y_test



def load_dataset(dataset_name,fold_number,division=1):

	shifts = range(10000,220000,70000)
	num_aug = (len(shifts)+2)*(channel+1)

	def add_noise(data):
		noise = np.random.randn(len(data))
		data_noise = data + 0.005 * noise
		return data_noise

	def shift(data,val):
		return np.roll(data, val)

	#load the images
	def load_images(image_filename, label_filename):
		if not os.path.exists(image_filename):
			print image_filename + ' not exists'

		if not os.path.exists(label_filename):
			print label_filename + ' not exists'

		data = np.load(image_filename)
		data_labels = np.load(label_filename)
		length = data.shape[1]
		print length
		triple_length = length/division
		triple = np.empty([data.shape[0]*division*num_aug,triple_length])
		labels = np.empty([data.shape[0]*division*num_aug])
		print data.shape[0],division,num_aug
		counter_triple = 0
		for i,d in enumerate(data):
			for cnt in range(division):
				#left image
				print counter_triple
				# print d[cnt*triple_length:(cnt+1)*triple_length].shape
				triple[counter_triple] = d[cnt*triple_length:(cnt+1)*triple_length][:,0]
				left = triple[counter_triple]
				labels[counter_triple] = data_labels[i]
				counter_triple += 1 

				#add noise to left image
				triple[counter_triple] = add_noise(left)
				labels[counter_triple] = data_labels[i]
				counter_triple += 1 

				#shift the left channel
				for val in shifts:
					triple[counter_triple] = shift(left,val)
					labels[counter_triple] = data_labels[i]
					counter_triple += 1

				#right image
				triple[counter_triple] = d[cnt*triple_length:(cnt+1)*triple_length][:,1]
				right = triple[counter_triple]
				labels[counter_triple] = data_labels[i]
				counter_triple += 1 

				#add noise to right image
				triple[counter_triple] = add_noise(right)
				labels[counter_triple] = data_labels[i]
				counter_triple += 1 

				#shift the right channel
				for val in shifts:
					triple[counter_triple] = shift(right,val)
					labels[counter_triple] = data_labels[i]
					counter_triple += 1

				#mean image
				triple[counter_triple] = np.mean(d[cnt*triple_length:(cnt+1)*triple_length],axis=1)
				mean = triple[counter_triple]
				labels[counter_triple] = data_labels[i]
				counter_triple += 1 

				#add noise to mean image
				triple[counter_triple] = add_noise(mean)
				labels[counter_triple] = data_labels[i]
				counter_triple += 1 

				#shift the mean 
				for val in shifts:
					triple[counter_triple] = shift(mean,val)
					labels[counter_triple] = data_labels[i]
					counter_triple += 1

		labels= labels.astype(np.uint8)
		# normalized = (triple - np.mean(triple)) / np.std(triple)
		triple = np.expand_dims(triple,axis=1)
		return triple,labels
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
	# y_val = load_labels(directory + dataset_name + '_wave_fold' + str(fold_number) + '_evaluate_labels.npy')
	X_test,y_test = load_images(directory + dataset_name + '_wave_meta_images.npy',
		directory + dataset_name + '_wave_meta_labels.npy')
	# y_test = load_labels(directory + dataset_name + '_wave_meta_labels.npy')


	return X_train, y_train, X_val, y_val, X_test, y_test,num_aug


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