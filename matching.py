# matching

import os
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


model = load_model('PATH/MODEL-NAME.h5')

# remove last layer since it is for the classification
model = keras.Model(model.layers[0].input, model.layers[-2].output)

# get all images paths
# data_dir = main folder containing all images
def get_image_paths(data_dir):
    # init empty list for paths
    out = []
    # loop through folder
    for imgs in os.listdir(data_dir):
            # add to list the image path
            out.append(data_dir + imgs)
    return out


# check that the model has the same shape in the input -- first layer in model.summary()
img_height = 120
img_width = 120

# extract the features for a given image
# inp_img_pth = path of a given image -- e.g. /home/a/animals/cat/cat1.jpeg
def extract_features(inp_img_pth):
	# load image
	img = tf.keras.utils.load_img(inp_img_pth, target_size=(img_height, img_width))
	# transform the image in array
	img_array = tf.keras.utils.img_to_array(img)
	# normalize the image
	img_array = img_array/255.0
	# add dimension to image array
	img_array = tf.expand_dims(img_array, 0)
	# extract image features
	return model.predict(img_array)[0]


# get query and gallery paths
query_paths, query_classes = get_image_paths('QUERY-PATH/')
print('found %s images for the queries' % (len(query_paths)))

gallery_paths, gallery_classes = get_image_paths('GALLERY-PATH/')
print('found %s images in gallery' % (len(gallery_paths)))


# extract gallery and query features

# init empty list to save features
gallery_features = []
# loop through the gallery paths
for p in gallery_paths:
	# for each image in the gallery extract the features and them to the list
	gallery_features.append(extract_features(p))

# init empty list to save features
query_features = []
# loop through the query paths
for p in query_paths:
	# for each image in the query extract the features and them to the list
	query_features.append(extract_features(p))


# for each query image compute the euclidean distance between the given query image and all the gallery images
# init empty dict where we'll save the matching between query and gallery images
# keys are the query paths
# values are lists of top10 most similar images
matching = {}
# loop through query features and query paths
for query, q in zip(query_features, query_paths):
	# L2 norm - Minkowski distance with r=2
	# compute euclidean distance between the query image and all gallery images
	euc_dist = [norm(query-b) for b in gallery_features]
	# get the sorted indeces of gallery images based on euclidean distance
	ids = np.argsort(euc_dist)[::-1]
	# add to the dictionary a key-value pair
	# key = gallery path
	# value = sorted list of paths -- top10
	matching[q] = [gallery_paths[ids[-i]] for i in range(1, 11)]


# get some query paths
qtest = list(matching.keys())[:5]

# plot some query images with top10 most similar gallery images
for queries in qtest:
	plt.figure(figsize=(20, 20))
	qpth = queries
	qimg = plt.imread(qpth)
	ax = plt.subplot(3,4,1)
	plt.imshow(qimg)
	plt.title('query image')
	for i in range(10):
	    pth = matching[qpth][i]
	    img = plt.imread(pth)
	    ax = plt.subplot(3,4,i+2)
	    plt.imshow(img)
	    plt.axis("off")
	plt.show()



# since we have not to send the complete path but only the image names, it is necessary to edit the dictionary
# we remove the initial path from both keys and list values
# init empty dictionary to be sent
out = {}
for match in matching:
	# get query image name
	# LAST-THING-BEFORE-IMAGE-NAME -- e.g. 'query', ')' (if animal name), etc.
	q_path = match.split('LAST-THING-BEFORE-IMAGE-NAME/')[1]
	# get gallery image names
	g_paths = [g.split('LAST-THING-BEFORE-IMAGE-NAME/')[1] for g in matching[match]]
	# add to the dictionary the key-value pair
	# key = query name
	# value = list of gallery names -- top10
	out[q_path] = g_paths


### send out
