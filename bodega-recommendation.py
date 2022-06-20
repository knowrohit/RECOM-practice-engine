import streamlit
import numpy as np
import tensorflow
import tensorflow.keras as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
import os
from PIL import Image
import pickle
from sklearn.neighbors import NearestNeighbors
import json
import base64
import streamlit.components.v1 as components

features_list = np.array(pickle.load(open('embiddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))
for i in filenames:
    print(i)


def fetching_json(index):
    with open(' /Users/rohittiwari/Documents/bodega-products (2)/styles.csv {}.json'.format(index)) as f:
        json_file = json.load(f)
        return  json_file


def save_file(file):
    try:
        with open(os.path.join('uploaded',file.name),'wb') as f:
            f.write(file.getbuffer())
        return 1
    except:
        return 0

model = ResNet50(include_top = False, weights='imagenet',input_shape=(224,224,3))          # use ResNet50 model
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    # we add our own layer
    GlobalMaxPooling2D()
])

# we create a function which extract the features from image
def extract_function(model, image_path):
    imag = image.load_img(image_path, target_size=(224, 224))
    imag_array = image.img_to_array(imag)
    imag_expand = np.expand_dims(imag_array,axis=0)
    processed_image = preprocess_input(imag_expand)
    result = model.predict(processed_image).flatten()
    result_norm = result/norm(result)

    return result_norm

def recommend(features_list, feature):
    neighbor = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
    neighbor.fit(features_list)
    distance, index = neighbor.kneighbors([feature])
    return index


if __name__ =='__main__':
    # steps
    # upload image
    # load image
    # extract features from uploaded file
    # recommend 5 images same as uploaded image....



    streamlit.sidebar.title('Bodega Recommendation Engine')
    file = streamlit.sidebar.file_uploader('')
    if file is not None:
        if save_file(file):
            # display image
            i = Image.open(file)
            streamlit.image(i,width=400)

            feature = extract_function(model,os.path.join('uploaded',file.name))
            index = recommend(features_list,feature)
            print(index)
            data_json = fetching_json(index[0][0])

            #create a columns for each image recommendation
            for each_index in index:
                col1, col2, col3, col4, col5 = streamlit.columns(5)
                with col1:
                    print(each_index[0])
                    streamlit.image(filenames[each_index[0]],caption=data_json['data']['productDisplayName'])                   #image1
                with col2:
                    streamlit.image(filenames[each_index[1]])                   #image2
                with col3:
                    streamlit.image(filenames[each_index[2]])                   #image3
                with col4:
                    streamlit.image(filenames[each_index[3]])                   #image4
                with col5:
                    streamlit.image(filenames[each_index[4]])                   #image5


