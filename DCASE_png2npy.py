import os
import scipy.misc
import numpy as np

my_map = {'bus':0, 'cafe/restaurant':1, 'car':2, 'city_center':3, 'forest_path':4, 'grocery_store':5,
'home':6, 'beach':7, 'library':8, 'metro_station':9, 'office':10, 'residential_area':11, 
'train':12, 'tram':13, 'park':14}

directory = 'TUT-Spec-Files/'
counter = 0

def find(name, path):
    for file in os.listdir(path):
        # print file, name
        if name.endswith(file):
            return os.path.join(path, file)

width = 768
height = 256


for text_filename in os.listdir(directory):
	if text_filename.endswith(".txt"):
		path = os.path.join(directory, text_filename)
		file = open(path)
		num_data = sum(1 for line in file)
		file = open(path)
		cnt = 0
		images = np.empty(shape = [num_data,1,height,width])
		labels = np.empty(num_data)
		for line in file:
			filename = line.split()[0]
			filename = filename[:-4]
			filename = filename + '.png'
			label = line.split()[1]
			# print filename,label
			image_folder = text_filename[:-4]
			image_path = find(filename,os.path.join(directory,image_folder))
			if(image_path):
				image = scipy.misc.imread(image_path)
				images[cnt] = image
				labels[cnt] = my_map[label]
				cnt += 1
		print cnt
		np.save(directory+image_folder+'_images.npy',images)
		np.save(directory+image_folder+'_labels.npy', labels)
						
print counter

