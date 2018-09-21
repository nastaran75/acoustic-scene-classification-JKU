import numpy as np
import theano
import theano.tensor as T
import lasagne
import cPickle, gzip
import os

directory = 'dataset/'


# def load_testset(dataset_name):
# 	#load the images
# 	def load_images(filename):
# 		if not os.path.exists(filename):
# 			print filename + ' not exists'

# 		data = np.load(filename)
# 		# normalized =  (data - np.mean(data)) / np.std(data)

# 		return data


# 	#load the labels
# 	def load_labels(filename):
# 		if not os.path.exists(filename):
# 			print filename + ' not exists'
# 		data = np.load(filename)
# 		data = data.astype(np.uint8)
# 		return data

# 	X_test = load_images(directory + dataset_name + '_test_images.npy')
# 	y_test = load_labels(directory + dataset_name + '_test_labels.npy')

# 	return X_test, y_test


def load_dataset(dataset_name, fold_number):

	#load the images
	def load_images(filename):
		if not os.path.exists(filename):
			print filename + ' not exists'

		data = np.load(filename)
		normalized =  (data - np.mean(data)) / np.std(data)

		# print np.mean(normalized)
		# print np.std(normalized)
		return normalized

	#load the labels
	def load_labels(filename):
		if not os.path.exists(filename):
			print filename + ' not exists'
		data = np.load(filename)
		data = data.astype(np.uint8)
		return data

	# def compute_mean(filename):
	# 	data = np.load(filename)
	# 	return np.mean(data),np.std(data)

	# mean_image,std = compute_mean(directory + dataset_name + '_fold' + str(fold_number) + '_train_images.npy')
	X_train = load_images(directory + dataset_name + '_fold' + str(fold_number) + '_train_images.npy')
	y_train = load_labels(directory + dataset_name + '_fold' + str(fold_number) + '_train_labels.npy')
	X_val = load_images(directory + dataset_name + '_fold' + str(fold_number) + '_evaluate_images.npy')
	y_val = load_labels(directory + dataset_name + '_fold' + str(fold_number) + '_evaluate_labels.npy')
	X_test = load_images(directory + dataset_name + '_test_images.npy')
	y_test = load_labels(directory + dataset_name + '_test_labels.npy')


	return X_train, y_train, X_val, y_val, X_test, y_test


def iterate_minibatches(inputs, targets, batch_size, shuffle = False):
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)

	
	for start_index in range(0, len(inputs) - batch_size + 1 , batch_size):
		if shuffle:
			excerpt = indices[start_index: start_index + batch_size]
		else:
			excerpt = slice(start_index,start_index + batch_size)

		yield inputs[excerpt], targets[excerpt]