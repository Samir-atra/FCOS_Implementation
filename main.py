# imports
# !pip3 install -U pip
# !pip3 install -U six numpy wheel packaging
# !pip3 install -U keras_preprocessing --no-deps
# !git clone https://github.com/tensorflow/tensorflow.git
# !cd tensorflow
# !pip install opencv-python
# !pip install tf-models-official
# import tqdm
import tensorflow as tf
# import tensorflow_datasets as tfds
# import tensorflow_models as tfm
import numpy as np
import json
import os
import cv2
import re
from operator import itemgetter
from Data.load_data import load_images




# dataset paths

# for colab
# train_imgs_path = "/content/train2014/"
# annotations = "/content/annotations/"


#for library
# train_imgs_path = "C:/Users/1845718/Documents/FCOS_Implementation/train2014/train2014/"
# val_imgs_path = "C:/Users/1845718/Documents/FCOS_Implementation/val2014/val2014/"
# test_imgs_path = "C:/Users/1845718/Documents/FCOS_Implementation/test2014/test2014/"
# annotations = "C:/Users/1845718/Documents/FCOS_Implementation/annotations_trainval2014/annotations/"

# for laptop
train_imgs_path = "/home/samer/Desktop/Beedoo/FCOS/FCOS_Implementation/COCO2014/train2014/"
val_imgs_path = "/home/samer/Desktop/Beedoo/FCOS/FCOS_Implementation/COCO2014/val2014/"
test_imgs_path = "/home/samer/Desktop/Beedoo/FCOS/FCOS_Implementation/COCO2014/test2014/"
annotations = "/home/samer/Desktop/Beedoo/FCOS/FCOS_Implementation/COCO2014/annotations/"

load_image(train_imgs_path)