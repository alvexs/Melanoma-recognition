from __future__ import print_function

import re
import os
import numpy as np

from skimage.io import imsave, imread
from skimage.transform import resize

data_path = 'raw/'
data_path2 = 'raw/data'


image_rows = 256
image_cols = 256


def load_train_data():
    imgs_train = np.load('/Users/alexivannikov/recognizeMelanoma/recognizeApp/networks/imgs_train.npy')
    imgs_mask_train = np.load('/Users/alexivannikov/recognizeMelanoma/recognizeApp/networks/imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id
