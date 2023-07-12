#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth=True
sess = tf.Session(config=sess_config)

import sys
import os

#COCO_DATA = 'data/coco'
COCO_DATA = 'data/severstal-steel-defect-detection'
MASK_RCNN_MODEL_PATH = 'lib/Mask_RCNN/'

if MASK_RCNN_MODEL_PATH not in sys.path:
    sys.path.append(MASK_RCNN_MODEL_PATH)
    
from samples.coco import coco
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

from mrcnn.utils import Dataset
from mrcnn.config import Config



    
from lib import utils as siamese_utils
from lib import model as siamese_model
from lib import config as siamese_config
   
import time
import datetime
import random
import numpy as np
import skimage.io
import imgaug
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


# ### Dataset
# for reproducibility
import random as rn
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)
rn.seed(RANDOM_SEED)
# paths
from pathlib import Path
img_train_folder = Path('/data/lmp/code/siamese-mask-rcnn/data/severstal-steel-defect-detection/train_images')
img_test_folder = Path('/data/lmp/code/siamese-mask-rcnn/data/severstal-steel-defect-detection/test_images')
# reading in the training set
import pandas as pd
data = pd.read_csv('/data/lmp/code/siamese-mask-rcnn/data/severstal-steel-defect-detection/train.csv')
data['ClassId'] = data['ClassId'].astype(np.uint8)

data.info()
data.head()
# keep only the images with labels
squashed = data.dropna(subset=['EncodedPixels'], axis='rows', inplace=True)

# squash multiple rows per image into a list
squashed = (
    data[['ImageId', 'EncodedPixels', 'ClassId']]
        .groupby('ImageId', as_index=False)
        .agg(list)
)

# count the amount of class labels per image
squashed['DistinctDefectTypes'] = squashed['ClassId'].apply(lambda x: len(x))


def rle_to_mask(lre, shape=(1600, 256)):
    '''
    params:  rle   - run-length encoding string (pairs of start & length of encoding)
             shape - (width,height) of numpy array to return

    returns: numpy array with dimensions of shape parameter
    '''
    # the incoming string is space-delimited
    runs = np.asarray([int(run) for run in lre.split(' ')])

    # we do the same operation with the even and uneven elements, but this time with addition
    runs[1::2] += runs[0::2]
    # pixel numbers start at 1, indexes start at 0
    runs -= 1

    # extract the starting and ending indeces at even and uneven intervals, respectively
    run_starts, run_ends = runs[0::2], runs[1::2]

    # build the mask
    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)
    for start, end in zip(run_starts, run_ends):
        mask[start:end] = 1

    # transform the numpy array from flat to the original image shape
    return mask.reshape(shape)

def build_mask(encodings, labels):
    """ takes a pair of lists of encodings and labels,
        and turns them into a 3d numpy array of shape (256, 1600, 4)
    """

    # initialise an empty numpy array
    mask = np.zeros((256, 1600, 4), dtype=np.uint8)

    # building the masks
    for rle, label in zip(encodings, labels):
        # classes are [1, 2, 3, 4], corresponding indeces are [0, 1, 2, 3]
        index = label - 1

        # fit the mask into the correct layer
        # note we need to transpose the matrix to account for
        # numpy and openCV handling width and height in reverse order
        mask[:, :, index] = rle_to_mask(rle).T

    return mask

class SeverstalDataset(Dataset):

    def __init__(self, dataframe):

        # https://rhettinger.wordpress.com/2011/05/26/super-considered-super/
        super().__init__(self)

        # needs to be in the format of our squashed df,
        # i.e. image id and list of rle plus their respective label on a single row
        self.dataframe = dataframe
        self.active_classes = []

        def set_active_classes(self, active_classes):
            """active_classes could be an array of integers (class ids), or
               a filename (string) containing these class ids (one number per line)"""
            if type(active_classes) == str:
                with open(active_classes, 'r') as f:
                    content = f.readlines()
                active_classes = [int(x.strip()) for x in content]
            self.active_classes = list(active_classes)

    def load_dataset(self, subset='train'):
        """ takes:
                - pandas df containing
                    1) file names of our images
                       (which we will append to the directory to find our images)
                    2) a list of rle for each image
                       (which will be fed to our build_mask()
                       function we also used in the eda section)
            does:
                adds images to the dataset with the utils.Dataset's add_image() metho
        """

        # input hygiene
        assert subset in ['train', 'test'], f'"{subset}" is not a valid value.'
        img_folder = img_train_folder if subset == 'train' else img_test_folder

        # add our four classes
        for i in range(1, 5):
            self.add_class(source='', class_id=i, class_name=f'defect_{i}')

        # add the image to our utils.Dataset class
        for index, row in self.dataframe.iterrows():
            file_name = row.ImageId
            file_path = f'{img_folder}/{file_name}'

            assert os.path.isfile(file_path), 'File doesn\'t exist.'
            self.add_image(source='',
                           image_id=file_name,
                           path=file_path)

    def load_mask(self, image_id):
        """As found in:
            https://github.com/matterport/Mask_RCNN/blob/master/samples/coco/coco.py

        Load instance masks for the given image

        This function converts the different mask format to one format
        in the form of a bitmap [height, width, instances]

        Returns:
            - masks    : A bool array of shape [height, width, instance count] with
                         one mask per instance
            - class_ids: a 1D array of class IDs of the instance masks
        """

        # find the image in the dataframe
        row = self.dataframe.iloc[image_id]

        # extract function arguments
        rle = row['EncodedPixels']
        labels = row['ClassId']

        # create our numpy array mask
        mask = build_mask(encodings=rle, labels=labels)

        # we're actually doing semantic segmentation, so our second return value is a bit awkward
        # we have one layer per class, rather than per instance... so it will always just be
        # 1, 2, 3, 4. See the section on Data Shapes for the Labels.
        return mask.astype(np.bool), np.array([1, 2, 3, 4], dtype=np.int32)

from sklearn.model_selection import train_test_split

# stratified split to maintain the same class balance in both sets
train, validate = train_test_split(squashed, test_size=0.2, random_state=RANDOM_SEED)

train_classes = np.array(range(1,5))

# instantiating training set
dataset_train = SeverstalDataset(dataframe=train)
dataset_train.load_dataset()
dataset_train.prepare()
dataset_train.ACTIVE_CLASSES = train_classes

# instantiating validation set
dataset_validate = SeverstalDataset(dataframe=validate)
dataset_validate.load_dataset()
dataset_validate.prepare()


class SmallTrainConfig(siamese_config.Config):
    # Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 6 # A 16GB GPU is required for a batch_size of 12
    NUM_CLASSES = 1 + 4
    NAME = 'small_coco'
    EXPERIMENT = 'example'
    CHECKPOINT_DIR = 'checkpoints/'
    # Adapt loss weights
    LOSS_WEIGHTS = {'rpn_class_loss': 2.0, 
                    'rpn_bbox_loss': 0.1, 
                    'mrcnn_class_loss': 2.0, 
                    'mrcnn_bbox_loss': 0.5, 
                    'mrcnn_mask_loss': 1.0}
    
class LargeTrainConfig(siamese_config.Config):
    # Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 4
    IMAGES_PER_GPU = 3 # 4 16GB GPUs are required for a batch_size of 12
    NUM_CLASSES = 1 + 1
    NAME = 'large_coco'
    EXPERIMENT = 'example'
    CHECKPOINT_DIR = 'checkpoints/'
    # Reduced image sizes
    TARGET_MAX_DIM = 192
    TARGET_MIN_DIM = 150
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    # Reduce model size
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024
    FPN_FEATUREMAPS = 256
    # Reduce number of rois at all stages
    RPN_ANCHOR_STRIDE = 1
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000
    TRAIN_ROIS_PER_IMAGE = 200
    DETECTION_MAX_INSTANCES = 100
    MAX_GT_INSTANCES = 100
    # Adapt NMS Threshold
    DETECTION_NMS_THRESHOLD = 0.5
    # Adapt loss weights
    LOSS_WEIGHTS = {'rpn_class_loss': 2.0, 
                    'rpn_bbox_loss': 0.1, 
                    'mrcnn_class_loss': 2.0, 
                    'mrcnn_bbox_loss': 0.5, 
                    'mrcnn_mask_loss': 1.0}


# #### Decide between small and large model

# In[5]:


# The small model trains on a single GPU and runs much faster.
# The large model is the same we used in our experiments but needs multiple GPUs and more time for training.
model_size = 'small' # or 'large'


# In[6]:


if model_size == 'small':
    config = SmallTrainConfig()
elif model_size == 'large':
    config = LargeTrainConfig()
    
config.display()


# In[7]:


# Create model object in inference mode.
model = siamese_model.SiameseMaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)


# ### Training

# In[8]:


train_schedule = OrderedDict()
train_schedule[1] = {"learning_rate": config.LEARNING_RATE, "layers": "heads"}
train_schedule[120] = {"learning_rate": config.LEARNING_RATE, "layers": "all"}
train_schedule[160] = {"learning_rate": config.LEARNING_RATE/10, "layers": "all"}


# In[9]:


# Load weights trained on Imagenet
try: 
    model.load_latest_checkpoint(training_schedule=train_schedule)
except:
    model.load_imagenet_weights(pretraining='imagenet-687')


# In[10]:


for epochs, parameters in train_schedule.items():
    print("")
    print("training layers {} until epoch {} with learning_rate {}".format(parameters["layers"], 
                                                                          epochs, 
                                                                          parameters["learning_rate"]))
    model.train(dataset_train, dataset_validate,
                learning_rate=parameters["learning_rate"], 
                epochs=epochs, 
                layers=parameters["layers"])






