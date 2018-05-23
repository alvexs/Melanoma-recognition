import os.path
import numpy as np
import matplotlib.image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing import image
from keras import optimizers
import h5py
from keras.models import Model
from keras import applications
from skimage.io import imsave, imread
# dimensions of images.
img_width, img_height = [256] * 2

train_data_dir = 'new_data/train'
validation_data_dir = 'new_data/validation'
test_data_dir = 'data/test'
nb_train_samples = 1766
nb_validation_samples = 689
epochs = 10
batch_size = 16


def create_model():

       # build the VGG16 network
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    # add the model on top of the convolutional base
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    # set the first 15 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:15]:
        layer.trainable = False

    return model

def recognize(target):
    model = create_model()

    #loading weights
    model.load_weights('/Users/alexivannikov/recognizeMelanoma/recognizeApp/networks/weights_new1.h5')


    if os.path.isfile(target):
        img = image.load_img(target, target_size=(256, 256))
        x = image.img_to_array(img)
        x = x[np.newaxis, ...]
        img2 = imread(target)
        prediction = model.predict(x)
        print(int(prediction))
        print(prediction[0][0])
        K.clear_session()
        return prediction[0][0]



#nevus_model = create_model()
#train(nevus_model)
#recognize('Figura4-805x560.jpg')
