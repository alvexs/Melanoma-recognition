from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave, imread
import numpy as np

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from .v1 import recognize
from .createSamples import create_samples
from .data import load_train_data, load_test_data

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 256
img_cols = 256

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.float32)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    # imgs_train = preprocess(imgs_train)
    # imgs_mask_train = preprocess(imgs_mask_train)


    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train = imgs_train.astype('float32')
    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')

    #imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train, imgs_mask_train, batch_size=32, epochs=50, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    # imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')

    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test2.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0]).astype(np.float32)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)


    return mean, std

def load_model():

    model = get_unet()

    model.load_weights('/Users/alexivannikov/recognizeMelanoma/recognizeApp/networks/weights.h5')
    imgs_test, imgs_id_test = load_test_data()
    #imgs_test = preprocess(imgs_test)
    imgs_train, imgs_mask_train = load_train_data()
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    imgs_mask_test = model.predict(imgs_test, verbose=1)

    np.save('imgs_mask_test.npy', imgs_mask_test)
    pred_dir = 'validation/melanoma_mask'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0]).astype(np.float32)
        imsave(os.path.join(pred_dir, str(image_id) + '_mask.png'), image)

def save_image():
    imgs_test, imgs_id_test = load_test_data()
    #imgs_test = preprocess(imgs_test)
    imgs_train, imgs_mask_train = load_train_data()
    #mean = np.mean(imgs_train)  # mean for data centering
    #std = np.std(imgs_train)  # std for data normalization

    imgs_test = imgs_test.astype('uint8')

    pred_dir = 'melanoma_img_color'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_test, imgs_id_test):
        #image = (image[:, :, 3]).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_image.png'), image)

def load_test_data_for_one(path):

    imgs = np.ndarray((1, img_rows, img_cols), dtype=np.float32)
    imgs_id = np.ndarray((1, ), dtype=np.int32)
    imgs_color = np.ndarray((1, img_rows, img_cols, 3), dtype=np.uint8)
    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)


    img = imread(path, as_grey=True)
    img_color = imread(path)

    img_color = resize(img_color, (img_rows, img_cols, 3), preserve_range=True)
    img = resize(img, (img_rows, img_cols), preserve_range=True)

    imgs_color[i] = img_color
    imgs[i] = img


    if i % 10 == 0:
        print('Done: {0}/{1} images'.format(i, 1))
    i += 1
    print('Loading done.')

    imgs = imgs[..., np.newaxis]

    np.save('/Users/alexivannikov/recognizeMelanoma/recognizeApp/media/numpy/imgs.npy', imgs)
    np.save('/Users/alexivannikov/recognizeMelanoma/recognizeApp/media/numpy/imgs_color.npy', imgs_color)
    print('Saving to .npy files done.')

    image_name = path.split('/')[-1]
    image_name = image_name.split('.')[0]

    pred_dir = '/Users/alexivannikov/recognizeMelanoma/recognizeMelanoma/media/color'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image in imgs_color:
        #image = (image[:, :, 3]).astype(np.uint8)
        img_color_path = os.path.join(pred_dir, str(image_name) + '.png')
        imsave(img_color_path, image)
    img_test = imgs
    return img_test, img_color_path


def predict_one(path):
    model = get_unet()
    model.load_weights('/Users/alexivannikov/recognizeMelanoma/recognizeApp/networks/weights.h5')
    img_test, img_color_path = load_test_data_for_one(path)
    imgs_train, imgs_mask_train = load_train_data()
    image_name = path.split('/')[-1]
    image_name = image_name.split('.')[0]
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    img_test = img_test.astype('float32')
    img_test -= mean
    img_test /= std
    img_mask_test = model.predict(img_test, verbose=1)

    np.save('/Users/alexivannikov/recognizeMelanoma/recognizeApp/media/numpy/imgs_mask_test.npy', img_mask_test)
    directory_result = os.path.normpath('/Users/alexivannikov/recognizeMelanoma/recognizeApp/static')
    pred_dir = '/Users/alexivannikov/recognizeMelanoma/recognizeMelanoma/media/mask'

    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image in img_mask_test:
        image = (image[:, :, 0]).astype(np.float32)
        mask_path = os.path.join(pred_dir, str(image_name) + '_mask.png')
        imsave(mask_path, image)
    image_path = create_samples(img_color_path, mask_path)
    res_path = '/media/color/'
    mask_path_path = '/media/mask/'
    img_color_path = img_color_path.split('/')[-1]
    mask_path = mask_path.split('/')[-1]
    res_color_path = os.path.join(res_path, img_color_path)
    res_mask_path = os.path.join(mask_path_path, mask_path)
    result_image_path = os.path.join(directory_result, image_path)
    if result_image_path == None:
        return None
    else:
        prediction = recognize(result_image_path)
        prediction_str = "{:.3f}".format(prediction)
        prediction = float(prediction_str)
        K.clear_session()
        return prediction, res_color_path, res_mask_path
