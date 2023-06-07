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
#train_classes = np.array(range(1,3))
train_classes = np.array(range(1,5))

# In[5]:


# Load COCO/val dataset
coco_val = siamese_utils.IndexedCocoDataset()
coco_object = coco_val.load_coco(COCO_DATA, "val", year="26", return_coco=True)
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
    checkpoint = '/data/lmp/code/siamese-mask-rcnn/logs/siamese_mrcnn_small_coco_severstal4class/siamese_mrcnn_0035.h5'
    #checkpoint = '/data/lmp/code/siamese-mask-rcnn/logs/siamese_mrcnn_small_coco_severstal4classaug/siamese_mrcnn_0028.h5'
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

image_id = list(ids[5].values())[0]
#image_id = list(np.random.choice(ids).values())[0]

#image_id = np.random.choice(coco_val.category_image_index[category])
#image_id = 4283
print(image_id)
image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(coco_val, config, image_id, use_mini_mask=False)
info = coco_val.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, coco_val.image_reference(image_id)))
print("Original image shape: ", modellib.parse_image_meta(image_meta[np.newaxis,...])["original_image_shape"][0])

log("original_image", image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

# Display GT
visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id,
                             coco_val.class_names, ax=get_ax(1),
                             show_bbox=False, show_mask=False,
                             title="Ground Truth")


target, window, scale, padding, crop, random_image_id = siamese_utils.get_same_target(image_id, category, coco_val, config, return_all=True)
print(print("target", random_image_id))

# Run object detection
#results = model.dett(images=[image], verbose=1)
results = model.detect([[target]], [image], verbose=1)
r = results[0]
print(r)
#print(r['class_ids'])

visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            coco_val.class_names, r['scores'], ax=get_ax(1), title="Ground truth and detections")

# Display predictions only
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                             coco_val.class_names, r['scores'], ax=get_ax(1),
                             show_bbox=False, show_mask=False,
                             title="Predictions")

visualize.display_differences(image, gt_bbox, gt_class_id, gt_mask,
                              r['rois'], r['class_ids'], r['scores'], r['masks'],coco_val.class_names,
                              ax=get_ax(), show_box=False, show_mask=False, iou_threshold=0.5, score_threshold=0.5)


#siamese_utils.display_results(target, image, r['rois'], r['masks'], r['class_ids'], r['scores'],show_mask=False, show_bbox=True)

#target, window, scale, padding, crop, random_image_id = siamese_utils.get_one_target(category, coco_val, config, return_all=True)
#results = model.detect([[target]], [image], verbose=1)

'''# Display results
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)
'''

# Compute AP over range 0.5 to 0.95 and print it
AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                          r['rois'], r['class_ids'], r['scores'], r['masks'])
visualize.plot_precision_recall(AP, precisions, recalls)



# Grid of ground truth objects and their predictions
visualize.plot_overlaps(gt_class_id, r['class_ids'], r['scores'],
                        overlaps, coco_val.class_names)





# Get anchors and convert to pixel coordinates
anchors = model.get_anchors(image.shape)
anchors = utils.denorm_boxes(anchors, image.shape[:2])
log("anchors", anchors)

# Generate RPN trainig targets
# target_rpn_match is 1 for positive anchors, -1 for negative anchors
# and 0 for neutral anchors.
target_rpn_match, target_rpn_bbox = modellib.build_rpn_targets(
    image.shape, anchors, gt_class_id, gt_bbox, model.config)
log("target_rpn_match", target_rpn_match)
log("target_rpn_bbox", target_rpn_bbox)

positive_anchor_ix = np.where(target_rpn_match[:] == 1)[0]
negative_anchor_ix = np.where(target_rpn_match[:] == -1)[0]
neutral_anchor_ix = np.where(target_rpn_match[:] == 0)[0]
positive_anchors = anchors[positive_anchor_ix]
negative_anchors = anchors[negative_anchor_ix]
neutral_anchors = anchors[neutral_anchor_ix]
log("positive_anchors", positive_anchors)
log("negative_anchors", negative_anchors)
log("neutral anchors", neutral_anchors)

# Apply refinement deltas to positive anchors
refined_anchors = utils.apply_box_deltas(
    positive_anchors,
    target_rpn_bbox[:positive_anchors.shape[0]] * model.config.RPN_BBOX_STD_DEV)
log("refined_anchors", refined_anchors, )

visualize.draw_boxes(image, ax=get_ax(), boxes=positive_anchors, refined_boxes=refined_anchors)
#print(r)
'''
nms = utils.non_max_suppression(r['rois'], r['scores'], 0.3)
print(r['rois'])
print("nms", nms)
r['rois'] = r['rois'][nms]
print(r['rois'])
'''

siamese_utils.display_results(target, image, r['rois'], r['masks'], r['class_ids'], r['scores'],show_mask=False, show_bbox=True)