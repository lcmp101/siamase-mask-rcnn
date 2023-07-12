#!/usr/bin/env python
# coding: utf-8

import sys
import os
#print(os.environ['LD_LIBRARY_PATH'])
os.environ['LD_LIBRARY_PATH']='/data/lmp/anaconda3/envs/siamese-mask-rcnn/lib/'
print(os.environ['LD_LIBRARY_PATH'])

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
sess_config = tf.ConfigProto()

#COCO_DATA = 'data/coco/'
COCO_DATA = 'data/severstal-steel-defect-detection'
MASK_RCNN_MODEL_PATH = 'lib/Mask_RCNN/'

if MASK_RCNN_MODEL_PATH not in sys.path:
    sys.path.append(MASK_RCNN_MODEL_PATH)
    
from samples.coco import coco
from mrcnn import utils

from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn.model import log
from mrcnn.visualize import display_images, display_differences

from lib import utils as siamese_utils
from lib import model as siamese_model
from lib import config as siamese_config

from lib import metrics as metrics
   
import time
import datetime
import random
import numpy as np
import skimage.io
import imgaug
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import OrderedDict
from natsort import natsorted as nt

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


# ### Dataset

# In[4]:


# train_classes = coco_nopascal_classes
#train_classes = np.array(range(1,81))
train_classes = np.array(range(1,3))
#train_classes = np.array(range(1,5))

# In[5]:


# Load COCO/val dataset
coco_val = siamese_utils.IndexedCocoDataset()
#coco_object = coco_val.load_coco(COCO_DATA, "val", year="3vAll", return_coco=True)
#coco_object = coco_val.load_coco(COCO_DATA, "val", year="26v2", return_coco=True)
#coco_object = coco_val.load_coco(COCO_DATA, "val", year="test26v2", return_coco=True)
#coco_object = coco_val.load_coco(COCO_DATA, "val", year="test2aug", return_coco=True)
coco_object = coco_val.load_coco(COCO_DATA, "val", year="test3vAll", return_coco=True)
coco_val.prepare()
coco_val.build_indices()
coco_val.ACTIVE_CLASSES = train_classes



# ### Model

# In[6]:


class SmallEvalConfig(siamese_config.Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    NAME = 'coco'
    EXPERIMENT = 'evaluation'
    CHECKPOINT_DIR = 'checkpoints/'
    NUM_TARGETS = 1
    
class LargeEvalConfig(siamese_config.Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    NAME = 'coco'
    EXPERIMENT = 'evaluation'
    CHECKPOINT_DIR = 'checkpoints/'
    NUM_TARGETS = 1
    
    # Large image sizes
    TARGET_MAX_DIM = 192
    TARGET_MIN_DIM = 150
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    # Large model size
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024
    FPN_FEATUREMAPS = 256
    # Large number of rois at all stages
    RPN_ANCHOR_STRIDE = 1
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000
    TRAIN_ROIS_PER_IMAGE = 200
    DETECTION_MAX_INSTANCES = 100
    MAX_GT_INSTANCES = 100


# #### Select small or large model config

# In[7]:


# The small model trains on a single GPU and runs much faster.
# The large model is the same we used in our experiments but needs multiple GPUs and more time for training.
model_size = 'small' # or 'large'


# In[8]:


if model_size == 'small':
    config = SmallEvalConfig()
elif model_size == 'large':
    config = LargeEvalConfig()
    
config.display()


# In[9]:


# Provide training schedule of the model
# When evaluationg intermediate steps the tranining schedule must be provided
train_schedule = OrderedDict()
if model_size == 'small':
    '''
    train_schedule[1] = {"learning_rate": config.LEARNING_RATE, "layers": "heads"}
    train_schedule[120] = {"learning_rate": config.LEARNING_RATE, "layers": "4+"}
    train_schedule[160] = {"learning_rate": config.LEARNING_RATE/10, "layers": "all"}
    '''
    train_schedule[1] = {"learning_rate": config.LEARNING_RATE, "layers": "heads"}
    train_schedule[25] = {"learning_rate": config.LEARNING_RATE, "layers": "all"}
    train_schedule[50] = {"learning_rate": config.LEARNING_RATE / 10, "layers": "all"}
elif model_size == 'large':
    train_schedule[1] = {"learning_rate": config.LEARNING_RATE, "layers": "heads"}
    train_schedule[240] = {"learning_rate": config.LEARNING_RATE, "layers": "all"}
    train_schedule[320] = {"learning_rate": config.LEARNING_RATE/10, "layers": "all"}


# In[10]:


# Select checkpoint
if model_size == 'small':
    #checkpoint = '/data/lmp/code/siamese-mask-rcnn/logs/siamese_mrcnn_small_coco_severstal3vall/siamese_mrcnn_0026.h5'
    #checkpoint = '/data/lmp/code/siamese-mask-rcnn/logs/siamese_mrcnn_small_coco_severstal4classv2/siamese_mrcnn_0050.h5'
    #checkpoint = '/data/lmp/code/siamese-mask-rcnn/logs/siamese_mrcnn_small_coco_severstalaug507/siamese_mrcnn_0042.h5'
    checkpoint = '/data/lmp/code/siamese-mask-rcnn/logs/siamese_mrcnn_small_coco_severstal3vall/siamese_mrcnn_0026.h5'
elif model_size == 'large':
    checkpoint = 'checkpoints/large_siamese_mrcnn_coco_full_0320.h5'


# ### Evaluation

# In[11]:


# Load and evaluate model
# Create model object in inference mode.
model = siamese_model.SiameseMaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_checkpoint(checkpoint, training_schedule=train_schedule)
# Evaluate only active classes
active_class_idx = np.array(coco_val.ACTIVE_CLASSES) - 1

# Evaluate on the validation set
print('starting evaluation ...')


# ### Visualization

# In[13]:

config.NUM_TARGETS = 1
# Create model object in inference mode.
model = siamese_model.SiameseMaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_checkpoint(checkpoint, training_schedule=train_schedule)
print("model loaded")


# In[17]:
def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    fig.tight_layout()
    return ax

# Select category
category = 1
#print(coco_val.category_image_index)


ids = nt([{image['id']:i} for i, image in enumerate(coco_val.image_info) if i in coco_val.category_image_index[category]])
#print(coco_val.category_image_index[category])
#print(ids)

# para bff6cdfad - ids[13] cat 2
#image_id = list(ids[13].values())[0]

dice1 = []
dice2 = []
results = []
pre = []
acc = []
ac2 = []
rec = []
iou = []
length = len(ids)

np.random.seed(1)

image_id = list(ids[0].values())[0]
image, _, gt_class_ids, _, gt_mask = modellib.load_image_gt(coco_val, model.config, image_id, use_mini_mask=False)
# Load target
target, _, _, _, _, random_image_id, box_ind = siamese_utils.get_same_target(image_id, category, coco_val, config, return_all=True)

'''
for i in range(length):
    image_id = list(ids[i].values())[0]
    print(image_id)
    # Load GT data
    image, _, gt_class_ids, _, gt_mask = modellib.load_image_gt(coco_val, model.config, image_id, use_mini_mask=False)

    # Load target
    #target, _, _, _, _, random_image_id, box_ind = siamese_utils.get_same_target(image_id, category, coco_val, config, return_all=True)
    # Run detection
    results = model.detect([[target]], [image], verbose=1)
    r = results[0]

    b = np.where(gt_class_ids == category)[0]

    dice1.append(metrics.dice_coef(gt_mask, r['masks'], b))
    dice2.append(metrics.dice_coef2(gt_mask, r['masks'], box_ind))
    pre.append(metrics.precision_score(gt_mask, r['masks'], b))
    rec.append(metrics.recall_score(gt_mask, r['masks'], b))
    acc.append(metrics.accuracy(gt_mask, r['masks'], b))
    ac2.append(metrics.acc2(gt_mask, r['masks'], b))
    iou.append(metrics.iou(gt_mask, r['masks'], b))
    '''
'''
print(dice1)
dice_total = np.sum(dice1)/length
print(dice_total)
dice_t2 = np.sum(dice2)/length
print(dice_t2)
'''
im_t = []
d_t1 = []
d_t2 = []
d_pre = []
d_rec = []
d_acc = []
d_ac2 = []
d_iou = []

length = len(ids)
num = 10
for run in range(num):
    print('\t*** Evaluation run {} ***'.format(run + 1))
    target, _, _, _, _, random_image_id, box_ind = siamese_utils.get_one_target(category, coco_val, config, return_all=True)
    print("RD id", random_image_id)
    im_t.append(random_image_id)
    dice1 = []
    dice2 = []
    results = []
    pre = []
    acc = []
    ac2 = []
    rec = []
    iou = []
    for i in range(length):
        image_id = list(ids[i].values())[0]
        print(image_id)
        # Load GT data
        image, _, gt_class_ids, _, gt_mask = modellib.load_image_gt(coco_val, model.config, image_id, use_mini_mask=False)

        # Load target
        #target, _, _, _, _, random_image_id, box_ind = siamese_utils.get_same_target(image_id, category, coco_val, config, return_all=True)
        # Run detection
        results = model.detect([[target]], [image], verbose=1)
        r = results[0]
        b = np.where(gt_class_ids == category)[0]

        dice1.append(metrics.dice_coef(gt_mask, r['masks'], b))
        dice2.append(metrics.dice_coef2(gt_mask, r['masks'], box_ind))

        pre.append(metrics.precision_score(gt_mask, r['masks'], b))
        rec.append(metrics.recall_score(gt_mask, r['masks'], b))
        acc.append(metrics.accuracy(gt_mask, r['masks'], b))
        ac2.append(metrics.acc2(gt_mask, r['masks'], b))
        iou.append(metrics.iou(gt_mask, r['masks'], b))

    dice_total = np.sum(dice1) / length
    print(dice_total)
    d_t1.append(dice_total)

    dice_t2 = np.sum(dice2) / length
    print(dice_t2)
    d_t2.append(dice_t2)

    pre_total = np.sum(pre) / length
    rec_total = np.sum(rec) / length
    acc_total = np.sum(acc) / length
    ac2_total = np.sum(ac2) / length
    iou_total = np.sum(iou) / length
    d_pre.append(pre_total)
    d_rec.append(rec_total)
    d_acc.append(acc_total)
    d_ac2.append(ac2_total)
    d_iou.append(iou_total)

    print('\n' * 5, end='')

print(ids)
print("Dices 1", dice1)
dice_total = np.sum(dice1)/length
print("Dices 1 total",dice_total)
print("Dices 2", dice2)
dice_t2 = np.sum(dice2)/length
print("Dices 2", dice_t2)

pre_total = np.sum(pre)/length
rec_total = np.sum(rec)/length
acc_total = np.sum(acc)/length
ac2_total = np.sum(ac2)/length
iou_total = np.sum(iou)/length
print("Prec", pre_total)
print("Rec", rec_total)
print("Acc", acc_total)
print("Acc", ac2_total)
print("IoU", iou_total)

total = np.stack([im_t, d_t1, d_t2, d_pre, d_rec, d_acc, d_ac2, d_iou],axis=1).T
#print("2", total)
import pandas as pd
df = pd.DataFrame({'target': total[0], 'dice': total[1], 'D2': total[2], 'pre': total[3], 'rec': total[4], 'acc': total[5],
                   'acc2': total[6], 'iou': total[7]})
print(df.head())
df.to_csv('/data/lmp/code/siamese-mask-rcnn/metrics/cat13val.csv',index=False)
'''
print("X random images")
print("random img", im_t)
print("dice", d_t1)
print("D2", d_t2)
print("pre", d_pre)
print("rec", d_rec)
print("acc", d_acc)
print("ac2", d_ac2)
print("iou", d_iou)
'''


category = 2
#print(coco_val.category_image_index)


ids = nt([{image['id']:i} for i, image in enumerate(coco_val.image_info) if i in coco_val.category_image_index[category]])
#print(coco_val.category_image_index[category])
#print(ids)

# para bff6cdfad - ids[13] cat 2
#image_id = list(ids[13].values())[0]

dice1 = []
dice2 = []
results = []
pre = []
acc = []
ac2 = []
rec = []
iou = []
length = len(ids)

np.random.seed(1)

image_id = list(ids[0].values())[0]
image, _, gt_class_ids, _, gt_mask = modellib.load_image_gt(coco_val, model.config, image_id, use_mini_mask=False)
# Load target
target, _, _, _, _, random_image_id, box_ind = siamese_utils.get_same_target(image_id, category, coco_val, config, return_all=True)


im_t = []
d_t1 = []
d_t2 = []
d_pre = []
d_rec = []
d_acc = []
d_ac2 = []
d_iou = []

length = len(ids)
num = 10
for run in range(num):
    print('\t*** Evaluation run {} ***'.format(run + 1))
    target, _, _, _, _, random_image_id, box_ind = siamese_utils.get_one_target(category, coco_val, config, return_all=True)
    print("RD id", random_image_id)
    im_t.append(random_image_id)
    dice1 = []
    dice2 = []
    results = []
    pre = []
    acc = []
    ac2 = []
    rec = []
    iou = []
    for i in range(length):
        image_id = list(ids[i].values())[0]
        print(image_id)
        # Load GT data
        image, _, gt_class_ids, _, gt_mask = modellib.load_image_gt(coco_val, model.config, image_id, use_mini_mask=False)

        # Load target
        #target, _, _, _, _, random_image_id, box_ind = siamese_utils.get_same_target(image_id, category, coco_val, config, return_all=True)
        # Run detection
        results = model.detect([[target]], [image], verbose=1)
        r = results[0]
        b = np.where(gt_class_ids == category)[0]

        dice1.append(metrics.dice_coef(gt_mask, r['masks'], b))
        dice2.append(metrics.dice_coef2(gt_mask, r['masks'], box_ind))

        pre.append(metrics.precision_score(gt_mask, r['masks'], b))
        rec.append(metrics.recall_score(gt_mask, r['masks'], b))
        acc.append(metrics.accuracy(gt_mask, r['masks'], b))
        ac2.append(metrics.acc2(gt_mask, r['masks'], b))
        iou.append(metrics.iou(gt_mask, r['masks'], b))

    dice_total = np.sum(dice1) / length
    print(dice_total)
    d_t1.append(dice_total)

    dice_t2 = np.sum(dice2) / length
    print(dice_t2)
    d_t2.append(dice_t2)

    pre_total = np.sum(pre) / length
    rec_total = np.sum(rec) / length
    acc_total = np.sum(acc) / length
    ac2_total = np.sum(ac2) / length
    iou_total = np.sum(iou) / length
    d_pre.append(pre_total)
    d_rec.append(rec_total)
    d_acc.append(acc_total)
    d_ac2.append(ac2_total)
    d_iou.append(iou_total)

    print('\n' * 5, end='')

#print(ids)
#print("Dices 1", dice1)
dice_total = np.sum(dice1)/length
#print("Dices 1 total",dice_total)
#print("Dices 2", dice2)
dice_t2 = np.sum(dice2)/length
#print("Dices 2", dice_t2)

pre_total = np.sum(pre)/length
rec_total = np.sum(rec)/length
acc_total = np.sum(acc)/length
ac2_total = np.sum(ac2)/length
iou_total = np.sum(iou)/length


total = np.stack([im_t, d_t1, d_t2, d_pre, d_rec, d_acc, d_ac2, d_iou],axis=1).T
#print("2", total)
import pandas as pd
df = pd.DataFrame({'target': total[0], 'dice': total[1], 'D2': total[2], 'pre': total[3], 'rec': total[4], 'acc': total[5],
                   'acc2': total[6], 'iou': total[7]})
print(df.head())
df.to_csv('/data/lmp/code/siamese-mask-rcnn/metrics/cat23vall.csv',index=False)

'''
category = 3
#print(coco_val.category_image_index)


ids = nt([{image['id']:i} for i, image in enumerate(coco_val.image_info) if i in coco_val.category_image_index[category]])
#print(coco_val.category_image_index[category])
#print(ids)

# para bff6cdfad - ids[13] cat 2
#image_id = list(ids[13].values())[0]

dice1 = []
dice2 = []
results = []
pre = []
acc = []
ac2 = []
rec = []
iou = []
length = len(ids)

np.random.seed(1)

image_id = list(ids[0].values())[0]
image, _, gt_class_ids, _, gt_mask = modellib.load_image_gt(coco_val, model.config, image_id, use_mini_mask=False)
# Load target
target, _, _, _, _, random_image_id, box_ind = siamese_utils.get_same_target(image_id, category, coco_val, config, return_all=True)


im_t = []
d_t1 = []
d_t2 = []
d_pre = []
d_rec = []
d_acc = []
d_ac2 = []
d_iou = []

length = len(ids)
num = 10
for run in range(num):
    print('\t*** Evaluation run {} ***'.format(run + 1))
    target, _, _, _, _, random_image_id, box_ind = siamese_utils.get_one_target(category, coco_val, config, return_all=True)
    print("RD id", random_image_id)
    im_t.append(random_image_id)
    dice1 = []
    dice2 = []
    results = []
    pre = []
    acc = []
    ac2 = []
    rec = []
    iou = []
    for i in range(length):
        image_id = list(ids[i].values())[0]
        print(image_id)
        # Load GT data
        image, _, gt_class_ids, _, gt_mask = modellib.load_image_gt(coco_val, model.config, image_id, use_mini_mask=False)

        # Load target
        #target, _, _, _, _, random_image_id, box_ind = siamese_utils.get_same_target(image_id, category, coco_val, config, return_all=True)
        # Run detection
        results = model.detect([[target]], [image], verbose=1)
        r = results[0]
        b = np.where(gt_class_ids == category)[0]

        dice1.append(metrics.dice_coef(gt_mask, r['masks'], b))
        dice2.append(metrics.dice_coef2(gt_mask, r['masks'], box_ind))

        pre.append(metrics.precision_score(gt_mask, r['masks'], b))
        rec.append(metrics.recall_score(gt_mask, r['masks'], b))
        acc.append(metrics.accuracy(gt_mask, r['masks'], b))
        ac2.append(metrics.acc2(gt_mask, r['masks'], b))
        iou.append(metrics.iou(gt_mask, r['masks'], b))

    dice_total = np.sum(dice1) / length
    print(dice_total)
    d_t1.append(dice_total)

    dice_t2 = np.sum(dice2) / length
    print(dice_t2)
    d_t2.append(dice_t2)

    pre_total = np.sum(pre) / length
    rec_total = np.sum(rec) / length
    acc_total = np.sum(acc) / length
    ac2_total = np.sum(ac2) / length
    iou_total = np.sum(iou) / length
    d_pre.append(pre_total)
    d_rec.append(rec_total)
    d_acc.append(acc_total)
    d_ac2.append(ac2_total)
    d_iou.append(iou_total)

    print('\n' * 5, end='')

#print(ids)
#print("Dices 1", dice1)
dice_total = np.sum(dice1)/length
#print("Dices 1 total",dice_total)
#print("Dices 2", dice2)
dice_t2 = np.sum(dice2)/length
#print("Dices 2", dice_t2)

pre_total = np.sum(pre)/length
rec_total = np.sum(rec)/length
acc_total = np.sum(acc)/length
ac2_total = np.sum(ac2)/length
iou_total = np.sum(iou)/length


total = np.stack([im_t, d_t1, d_t2, d_pre, d_rec, d_acc, d_ac2, d_iou],axis=1).T
#print("2", total)
import pandas as pd
df = pd.DataFrame({'target': total[0], 'dice': total[1], 'D2': total[2], 'pre': total[3], 'rec': total[4], 'acc': total[5],
                   'acc2': total[6], 'iou': total[7]})
print(df.head())
df.to_csv('/data/lmp/code/siamese-mask-rcnn/metrics/cat3Aug2val2.csv',index=False)


category = 4
#print(coco_val.category_image_index)


ids = nt([{image['id']:i} for i, image in enumerate(coco_val.image_info) if i in coco_val.category_image_index[category]])
#print(coco_val.category_image_index[category])
#print(ids)

# para bff6cdfad - ids[13] cat 2
#image_id = list(ids[13].values())[0]

dice1 = []
dice2 = []
results = []
pre = []
acc = []
ac2 = []
rec = []
iou = []
length = len(ids)

np.random.seed(1)

image_id = list(ids[0].values())[0]
image, _, gt_class_ids, _, gt_mask = modellib.load_image_gt(coco_val, model.config, image_id, use_mini_mask=False)
# Load target
target, _, _, _, _, random_image_id, box_ind = siamese_utils.get_same_target(image_id, category, coco_val, config, return_all=True)


im_t = []
d_t1 = []
d_t2 = []
d_pre = []
d_rec = []
d_acc = []
d_ac2 = []
d_iou = []

length = len(ids)
num = 10
for run in range(num):
    print('\t*** Evaluation run {} ***'.format(run + 1))
    target, _, _, _, _, random_image_id, box_ind = siamese_utils.get_one_target(category, coco_val, config, return_all=True)
    print("RD id", random_image_id)
    im_t.append(random_image_id)
    dice1 = []
    dice2 = []
    results = []
    pre = []
    acc = []
    ac2 = []
    rec = []
    iou = []
    for i in range(length):
        image_id = list(ids[i].values())[0]
        print(image_id)
        # Load GT data
        image, _, gt_class_ids, _, gt_mask = modellib.load_image_gt(coco_val, model.config, image_id, use_mini_mask=False)

        # Load target
        #target, _, _, _, _, random_image_id, box_ind = siamese_utils.get_same_target(image_id, category, coco_val, config, return_all=True)
        # Run detection
        results = model.detect([[target]], [image], verbose=1)
        r = results[0]
        b = np.where(gt_class_ids == category)[0]

        dice1.append(metrics.dice_coef(gt_mask, r['masks'], b))
        dice2.append(metrics.dice_coef2(gt_mask, r['masks'], box_ind))

        pre.append(metrics.precision_score(gt_mask, r['masks'], b))
        rec.append(metrics.recall_score(gt_mask, r['masks'], b))
        acc.append(metrics.accuracy(gt_mask, r['masks'], b))
        ac2.append(metrics.acc2(gt_mask, r['masks'], b))
        iou.append(metrics.iou(gt_mask, r['masks'], b))

    dice_total = np.sum(dice1) / length
    print(dice_total)
    d_t1.append(dice_total)

    dice_t2 = np.sum(dice2) / length
    print(dice_t2)
    d_t2.append(dice_t2)

    pre_total = np.sum(pre) / length
    rec_total = np.sum(rec) / length
    acc_total = np.sum(acc) / length
    ac2_total = np.sum(ac2) / length
    iou_total = np.sum(iou) / length
    d_pre.append(pre_total)
    d_rec.append(rec_total)
    d_acc.append(acc_total)
    d_ac2.append(ac2_total)
    d_iou.append(iou_total)

    print('\n' * 5, end='')

#print(ids)
#print("Dices 1", dice1)
dice_total = np.sum(dice1)/length
#print("Dices 1 total",dice_total)
#print("Dices 2", dice2)
dice_t2 = np.sum(dice2)/length
#print("Dices 2", dice_t2)

pre_total = np.sum(pre)/length
rec_total = np.sum(rec)/length
acc_total = np.sum(acc)/length
ac2_total = np.sum(ac2)/length
iou_total = np.sum(iou)/length


total = np.stack([im_t, d_t1, d_t2, d_pre, d_rec, d_acc, d_ac2, d_iou],axis=1).T
#print("2", total)
import pandas as pd
df = pd.DataFrame({'target': total[0], 'dice': total[1], 'D2': total[2], 'pre': total[3], 'rec': total[4], 'acc': total[5],
                   'acc2': total[6], 'iou': total[7]})
print(df.head())
df.to_csv('/data/lmp/code/siamese-mask-rcnn/metrics/cat4Aug2val2.csv',index=False)



'''