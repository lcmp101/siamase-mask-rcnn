import sys
import os

# print(os.environ['LD_LIBRARY_PATH'])
os.environ['LD_LIBRARY_PATH'] = '/data/lmp/anaconda3/envs/siamese-mask-rcnn/lib/'
print(os.environ['LD_LIBRARY_PATH'])

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
sess_config = tf.ConfigProto()

# COCO_DATA = 'data/coco/'
COCO_DATA = 'data/severstal-steel-defect-detection'
MASK_RCNN_MODEL_PATH = 'lib/Mask_RCNN/'

if MASK_RCNN_MODEL_PATH not in sys.path:
    sys.path.append(MASK_RCNN_MODEL_PATH)

from samples.coco import coco
from mrcnn import utils  # In[1]:

from mrcnn import model as modellib
from mrcnn import visualize

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

# In[4]:


# train_classes = coco_nopascal_classes
# train_classes = np.array(range(1,81))
train_classes = np.array(range(1,3))
#train_classes = np.array(range(1, 5))

# In[5]:


# Load COCO/val dataset
test_dataset = siamese_utils.IndexedCocoDataset()
coco_test_object = test_dataset.load_coco(COCO_DATA, subset="val", year="32", return_coco=True)
test_dataset.prepare()
test_dataset.build_indices()


class SmallTrainConfig(siamese_config.Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    NAME = 'coco'
    EXPERIMENT = 'evaluation'

config = SmallTrainConfig()

train_schedule = OrderedDict()
train_schedule[1] = {"learning_rate": config.LEARNING_RATE, "layers": "heads"}
train_schedule[25] = {"learning_rate": config.LEARNING_RATE, "layers": "all"}
train_schedule[50] = {"learning_rate": config.LEARNING_RATE / 10, "layers": "all"}

# Load pretrained checkpoint
config = SmallTrainConfig()
#checkpoint = '/data/lmp/code/siamese-mask-rcnn/logs/siamese_mrcnn_small_coco_severstal4class/siamese_mrcnn_0035.h5'
checkpoint = '/data/lmp/code/siamese-mask-rcnn/logs/siamese_mrcnn_small_coco_severstal2classv2/siamese_mrcnn_0042.h5'

# Evaluate on all classes
#test_dataset.ACTIVE_CLASSES = np.array(range(1, 5))
test_dataset.ACTIVE_CLASSES = np.array(range(1, 3))

# Load and evaluate models

# Create model object in inference mode.
model = siamese_model.SiameseMaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_checkpoint(checkpoint, training_schedule=train_schedule)
# Evaluate only active classes
active_class_idx = np.array(test_dataset.ACTIVE_CLASSES) - 1

# Evaluate on the validation set
print('evaluating five times')

for run in range(5):
    print('\t*** Evaluation run {} ***'.format(run + 1))

    siamese_utils.evaluate_dataset(model, test_dataset, coco_test_object, eval_type=["bbox", "segm"],
                                   dataset_type='coco', limit=0, image_ids=None,  # limit=0 -> entire data set
                                   class_index=active_class_idx, verbose=1)

    print('\n' * 5, end='')