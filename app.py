# first step is importing all the important modules

import pandas as pd
import numpy as np
import tensorflow
import tensorflow.keras as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# our resent model is pre-trained on the imagenet dataset,
# so we have to use this resnet model for the feature extraction
# create a model variable
# we are not including the top layer in our model
model = ResNet50(include_top = False, weights='imagenet',input_shape=(224,224,3))
# we don't have to train the model
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    # we add our own layer
    GlobalMaxPooling2D()
])

# this gives the summary of our resnet model
print(model.summary)

imag = image.load_img('/Users/rohittiwari/Documents/bodega-products (2)/images/1529.jpg',target_size=(224,224))       #from the prerpocessing we uses image to load images
imag_array = image.img_to_array(imag)          # convert image file into numpy array formate
print(imag_array)
print('shape of image is: ',imag_array.shape)

# we have to expand the dimensions of the image array,
# because keras accept the batch of the images
image_expand = np.expand_dims(imag_array,axis=0)
print(image_expand)
print('shape of the image is: ',image_expand.shape)

# using preprocess_input, which convert the image input array into resnet considerable manner
processed_image = preprocess_input(image_expand)
print(processed_image.shape)
print(processed_image)

result = model.predict(processed_image).flatten()
norm_result = result/norm(result)
print(norm_result)


# Make a function which extract the feature from image
# we combine all the steps that perform above
def extract_feature(image_path, model):
    imag = image.load_img(image_path, target_size=(224, 224))
    imag_array = image.img_to_array(imag)
    image_expand = np.expand_dims(imag_array, axis=0)
    processed_image = preprocess_input(image_expand)
    result = model.predict(processed_image).flatten()
    norm_result = result / np.linalg.norm(result)

    return norm_result

# create a list of filenames with the required path
filenames = []
for file in os.listdir('/Users/rohittiwari/Documents/bodega-products (2)/images'):
    filenames.append(os.path.join('/Users/rohittiwari/Documents/bodega-products (2)/images',file))


# create a list of features name
features_list = []
for files in tqdm(filenames):
    features_list.append(extract_feature(files,model))

# we will use this list of filenames and features so we make a pickle file of both
# this files used in the recommendation while testing

with open ("embiddings.pkl" , 'wb') as f:
    pickle.dump (features_list , f)

with open ("filenames.pkl" , 'wb') as f:
    pickle.dump (filenames , f)
