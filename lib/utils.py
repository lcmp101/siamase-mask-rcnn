# Simaese Mask R-CNN Utils

import tensorflow as tf
import sys
import os
import time
import random
import numpy as np
import skimage.io
import skimage.transform as skt
import imgaug
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 6.0)

MASK_RCNN_MODEL_PATH = 'Mask_RCNN/'

if MASK_RCNN_MODEL_PATH not in sys.path:
    sys.path.append(MASK_RCNN_MODEL_PATH)
    
from samples.coco import coco
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
import json
import pandas as pd

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
    
import warnings
warnings.filterwarnings("ignore")
    
### Data Generator ###
    
def get_one_target(category, dataset, config, augmentation=None, target_size_limit=0, max_attempts=10, return_all=False, return_original_size=False):

    n_attempts = 0
    while True:
        # Get index with corresponding images for each category
        category_image_index = dataset.category_image_index
        # Draw a random image
        random_image_id = np.random.choice(category_image_index[category])
        #random_image_id = 333
        # Load image
        target_image, target_image_meta, target_class_ids, target_boxes, target_masks = \
            modellib.load_image_gt(dataset, config, random_image_id, augmentation=augmentation,
                          use_mini_mask=config.USE_MINI_MASK)
        # print(random_image_id, category, target_class_ids)

        if not np.any(target_class_ids == category):
            continue

        # try:
        #     box_ind = np.random.choice(np.where(target_class_ids == category)[0])   
        # except ValueError:
        #     return None
        box_ind = np.random.choice(np.where(target_class_ids == category)[0])   
        tb = target_boxes[box_ind,:]
        target = target_image[tb[0]:tb[2],tb[1]:tb[3],:]
        original_size = target.shape
        target, window, scale, padding, crop = utils.resize_image(
            target,
            min_dim=config.TARGET_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE, #Same scaling as the image
            max_dim=config.TARGET_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE) #Same output format as the image

        n_attempts = n_attempts + 1
        if (min(original_size[:2]) >= target_size_limit) or (n_attempts >= max_attempts):
            break
    
    if return_all:
        return target, window, scale, padding, crop, random_image_id, box_ind
    elif return_original_size:
        return target, original_size
    else:
        return target


def get_same_target(image_id, category, dataset, config, augmentation=None, target_size_limit=0, max_attempts=10, return_all=False,
                   return_original_size=False):
    n_attempts = 0
    while True:
        # Get index with corresponding images for each category
        #category_image_index = dataset.category_image_index
        # Draw a random image
        random_image_id = image_id
        # random_image_id = 333
        # Load image
        target_image, target_image_meta, target_class_ids, target_boxes, target_masks = \
            modellib.load_image_gt(dataset, config, random_image_id, augmentation=augmentation,
                                   use_mini_mask=config.USE_MINI_MASK)
        # print(random_image_id, category, target_class_ids)

        if not np.any(target_class_ids == category):
            continue

        # try:
        #     box_ind = np.random.choice(np.where(target_class_ids == category)[0])
        # except ValueError:
        #     return None
        box_ind = np.random.choice(np.where(target_class_ids == category)[0])

        tb = target_boxes[box_ind, :]
        target = target_image[tb[0]:tb[2], tb[1]:tb[3], :]
        original_size = target.shape
        target, window, scale, padding, crop = utils.resize_image(
            target,
            min_dim=config.TARGET_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,  # Same scaling as the image
            max_dim=config.TARGET_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE)  # Same output format as the image

        n_attempts = n_attempts + 1
        if (min(original_size[:2]) >= target_size_limit) or (n_attempts >= max_attempts):
            break

    if return_all:
        return target, window, scale, padding, crop, random_image_id, box_ind
    elif return_original_size:
        return target, original_size
    else:
        return target


def siamese_data_generator(dataset, config, shuffle=True, augmentation=imgaug.augmenters.Fliplr(0.5), random_rois=0,
                   batch_size=1, detection_targets=False, diverse=0):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.
    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augment: If True, applies image augmentation to images (currently only
             horizontal flips are supported)
    random_rois: If > 0 then generate proposals to be used to train the
                 network classifier and mask heads. Useful if training
                 the Mask RCNN part without the RPN.
    batch_size: How many images to return in each call
    detection_targets: If True, generate detection targets (class IDs, bbox
        deltas, and masks). Typically for debugging or visualizations because
        in trainig detection targets are generated by DetectionTargetLayer.
    diverse: Float in [0,1] indicatiing probability to draw a target
        from any random class instead of one from the image classes
    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The containtes
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, size of image meta]
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                are those of the image unless use_mini_mask is True, in which
                case they are defined in MINI_MASK_SHAPE.
    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    backbone_shapes = modellib.compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    # Keras requires a generator to run indefinately.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]
            image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                modellib.load_image_gt(dataset, config, image_id, augmentation=augmentation,
                              use_mini_mask=config.USE_MINI_MASK)

            # Replace class ids with foreground/background info if binary
            # class option is chosen
            # if binary_classes == True:
            #    gt_class_ids = np.minimum(gt_class_ids, 1)

            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue
                
#             print(gt_class_ids)

            # Use only positive class_ids
            categories = np.unique(gt_class_ids)
            _idx = categories > 0
            categories = categories[_idx]
            # Use only active classes
            active_categories = []
            for c in categories:
                if any(c == dataset.ACTIVE_CLASSES):
                    active_categories.append(c)
            
            # Skiop image if it contains no instance of any active class    
            if not np.any(np.array(active_categories) > 0):
                continue
            # Randomly select category
            category = np.random.choice(active_categories)
                
            # Generate siamese target crop
            if not config.NUM_TARGETS:
                config.NUM_TARGETS = 1
            targets = []
            for i in range(config.NUM_TARGETS):
                targets.append(get_one_target(category, dataset, config, augmentation=augmentation))
#             target = np.stack(target, axis=0)
                    
#             print(target_class_id)
            target_class_id = category
            target_class_ids = np.array([target_class_id])
            
            idx = gt_class_ids == target_class_id
            siamese_class_ids = idx.astype('int8')
#             print(idx)
#             print(gt_boxes.shape, gt_masks.shape)
            siamese_class_ids = siamese_class_ids[idx]
            gt_class_ids = gt_class_ids[idx]
            gt_boxes = gt_boxes[idx,:]
            gt_masks = gt_masks[:,:,idx]
            image_meta = image_meta[:14]
#             print(gt_boxes.shape, gt_masks.shape)

            # RPN Targets
            rpn_match, rpn_bbox = modellib.build_rpn_targets(image.shape, anchors,
                                                    gt_class_ids, gt_boxes, config)

            # Mask R-CNN Targets
            if random_rois:
                rpn_rois = modellib.generate_random_rois(
                    image.shape, random_rois, gt_class_ids, gt_boxes)
                if detection_targets:
                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask =\
                        modellib.build_detection_targets(
                            rpn_rois, gt_class_ids, gt_boxes, gt_masks, config)

            # Init batch arrays
            if b == 0:
                batch_image_meta = np.zeros(
                    (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros(
                    [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                batch_targets = np.zeros(
                    (batch_size, config.NUM_TARGETS) + targets[0].shape, dtype=np.float32)
#                 batch_target_class_ids = np.zeros(
#                     (batch_size, config.MAX_TARGET_INSTANCES), dtype=np.int32)
                if config.USE_MINI_MASK:
                    batch_gt_masks = np.zeros((batch_size, config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1],
                                               config.MAX_GT_INSTANCES))
                else:
                    batch_gt_masks = np.zeros(
                        (batch_size, image.shape[0], image.shape[1], config.MAX_GT_INSTANCES))
                if random_rois:
                    batch_rpn_rois = np.zeros(
                        (batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                    if detection_targets:
                        batch_rois = np.zeros(
                            (batch_size,) + rois.shape, dtype=rois.dtype)
                        batch_mrcnn_class_ids = np.zeros(
                            (batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                        batch_mrcnn_bbox = np.zeros(
                            (batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                        batch_mrcnn_mask = np.zeros(
                            (batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                siamese_class_ids = siamese_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]
                

            # Add to batch
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = modellib.mold_image(image.astype(np.float32), config)
            batch_targets[b] = np.stack([modellib.mold_image(target.astype(np.float32), config) for target in targets], axis=0)
            batch_gt_class_ids[b, :siamese_class_ids.shape[0]] = siamese_class_ids
#             batch_target_class_ids[b, :target_class_ids.shape[0]] = target_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            if random_rois:
                batch_rpn_rois[b] = rpn_rois
                if detection_targets:
                    batch_rois[b] = rois
                    batch_mrcnn_class_ids[b] = mrcnn_class_ids
                    batch_mrcnn_bbox[b] = mrcnn_bbox
                    batch_mrcnn_mask[b] = mrcnn_mask
            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images, batch_image_meta, batch_targets, batch_rpn_match, batch_rpn_bbox,
                          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                outputs = []

                if random_rois:
                    inputs.extend([batch_rpn_rois])
                    if detection_targets:
                        inputs.extend([batch_rois])
                        # Keras requires that output and targets have the same number of dimensions
                        batch_mrcnn_class_ids = np.expand_dims(
                            batch_mrcnn_class_ids, -1)
                        outputs.extend(
                            [batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            modellib.logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise
                
                
### Dataset Utils ###

class IndexedCocoDataset(coco.CocoDataset):
    
    def __init__(self):
        super(IndexedCocoDataset, self).__init__()
        self.active_classes = []

    def set_active_classes(self, active_classes):
        """active_classes could be an array of integers (class ids), or
           a filename (string) containing these class ids (one number per line)"""
        if type(active_classes) == str:
            with open(active_classes, 'r') as f:
                content = f.readlines()
            active_classes = [int(x.strip()) for x in content]
        self.active_classes = list(active_classes)
        
    def get_class_ids(self, active_classes, dataset_dir, subset, year):
        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        class_ids = sorted(list(filter(lambda c: c in coco.getCatIds(), self.active_classes)))
        return class_ids

        self.class_ids_with_holes = class_ids
    
    def build_indices(self):

        self.image_category_index = IndexedCocoDataset._build_image_category_index(self)
        self.category_image_index = IndexedCocoDataset._build_category_image_index(self.image_category_index)

    def _build_image_category_index(dataset):

        image_category_index = []
        for im in range(len(dataset.image_info)):
            # List all classes in an image
            coco_class_ids = list(\
                                  np.unique(\
                                            [dataset.image_info[im]['annotations'][i]['category_id']\
                                             for i in range(len(dataset.image_info[im]['annotations']))]\
                                           )\
                                 )
            # Map 91 class IDs 81 to Mask-RCNN model type IDs
            class_ids = [dataset.map_source_class_id("coco.{}".format(coco_class_ids[k]))\
                         for k in range(len(coco_class_ids))]
            # Put list together
            image_category_index.append(class_ids)

        return image_category_index

    def _build_category_image_index(image_category_index):

        category_image_index = []
        # Loop through all 81 Mask-RCNN classes/categories
        for category in range(max(image_category_index)[0]+1):
            # Find all images corresponding to the selected class/category 
            images_per_category = np.where(\
                [any(image_category_index[i][j] == category\
                 for j in range(len(image_category_index[i])))\
                 for i in range(len(image_category_index))])[0]
            # Put list together
            category_image_index.append(images_per_category)

        return category_image_index



### Evaluation ###



class customCOCOeval(COCOeval):
    
    def summarize(self, class_index=None, verbose=1):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if not class_index is None:
                    s = s[:,:,class_index,aind,mind]
                else:
                    s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if not class_index is None:
                    s = s[:,class_index,aind,mind]
                else:
                    s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            if verbose > 0:
                print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self, cass_index=None):
        self.summarize(class_index)

def evaluate_coco(model, dataset, coco_object, eval_type="bbox", 
                  limit=0, image_ids=None, class_index=None, verbose=1, return_results=False):
    """Wrapper to keep original function name usable"""
        
    results = evaluate_dataset(model, dataset, coco_object, eval_type=eval_type, dataset_type='coco',
                     limit=limit, image_ids=image_ids, class_index=class_index, verbose=verbose, return_results=return_results)
    
    if return_results:
        return results
    
        
def evaluate_dataset(model, dataset, dataset_object, eval_type="bbox", dataset_type='coco', 
                     limit=0, image_ids=None, class_index=None, verbose=1, random_detections=False, return_results=False):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    assert dataset_type in ['coco']
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    dataset_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        if i%100 == 0 and verbose > 1:
            print("Processing image {}/{} ...".format(i, len(image_ids)))
        
        # Load GT data
        _, _, gt_class_ids, _, _ = modellib.load_image_gt(dataset, model.config, 
                                                          image_id, augmentation=False, 
                                                          use_mini_mask=model.config.USE_MINI_MASK)

        # BOILERPLATE: Code duplicated in siamese_data_loader

        # Skip images that have no instances. This can happen in cases
        # where we train on a subset of classes and the image doesn't
        # have any of the classes we care about.
        if not np.any(gt_class_ids > 0):
            continue

        # Use only positive class_ids
        categories = np.unique(gt_class_ids)
        _idx = categories > 0
        categories = categories[_idx]
        # Use only active classes
        active_categories = []
        for c in categories:
            if any(c == dataset.ACTIVE_CLASSES):
                active_categories.append(c)

        # Skiop image if it contains no instance of any active class    
        if not np.any(np.array(active_categories) > 0):
            continue

        # END BOILERPLATE

        # Evaluate for every category individually
        for category in active_categories:
            
            # Load image
            image = dataset.load_image(image_id)

            # Draw random target
            target = []
            for k in range(model.config.NUM_TARGETS):
                try:
                    target.append(get_one_target(category, dataset, model.config))
                except:
                    print('error fetching target of category', category)
                    continue
            target = np.stack(target, axis=0)
            # Run detection
            t = time.time()
            try:
                r = model.detect([target], [image], verbose=0, random_detections=random_detections)[0]
            except:
                print('error running detection for category', category)
                continue
            t_prediction += (time.time() - t)
        
            
            # Format detections
            r["class_ids"] = np.array([category for i in range(r["class_ids"].shape[0])])

            # Convert results to COCO format
            # Cast masks to uint8 because COCO tools errors out on bool
            if dataset_type == 'coco':
                image_results = coco.build_coco_results(dataset, dataset_image_ids[i:i + 1],
                                                   r["rois"], r["class_ids"],
                                                   r["scores"],
                                                   r["masks"].astype(np.uint8))
            results.extend(image_results)
    
    # Load results. This modifies results with additional attributes.
    dataset_results = dataset_object.loadRes(results)
    
    # allow evaluating bbox & segm:
    if not isinstance(eval_type, (list,)):
        eval_type = [eval_type]
        
    for current_eval_type in eval_type:
        # Evaluate
        cocoEval = customCOCOeval(dataset_object, dataset_results, current_eval_type)
        cocoEval.params.imgIds = dataset_image_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize(class_index=class_index, verbose=verbose)
        if verbose > 0:
            print("Prediction time: {}. Average {}/image".format(
                t_prediction, t_prediction / len(image_ids)))
            print("Total time: ", time.time() - t_start)
        
    if return_results:
        return cocoEval


def evaluate_same(model, dataset, dataset_object, eval_type="bbox", dataset_type='coco',
                     limit=0, image_ids=None, class_index=None, verbose=1, random_detections=False,
                     return_results=False):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    assert dataset_type in ['coco']
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    dataset_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        if i % 100 == 0 and verbose > 1:
            print("Processing image {}/{} ...".format(i, len(image_ids)))

        # Load GT data
        _, _, gt_class_ids, _, _ = modellib.load_image_gt(dataset, model.config,
                                                          image_id, augmentation=False,
                                                          use_mini_mask=model.config.USE_MINI_MASK)

        # BOILERPLATE: Code duplicated in siamese_data_loader

        # Skip images that have no instances. This can happen in cases
        # where we train on a subset of classes and the image doesn't
        # have any of the classes we care about.
        if not np.any(gt_class_ids > 0):
            continue

        # Use only positive class_ids
        categories = np.unique(gt_class_ids)
        _idx = categories > 0
        categories = categories[_idx]
        # Use only active classes
        active_categories = []
        for c in categories:
            if any(c == dataset.ACTIVE_CLASSES):
                active_categories.append(c)

        # Skiop image if it contains no instance of any active class
        if not np.any(np.array(active_categories) > 0):
            continue

        # END BOILERPLATE

        # Evaluate for every category individually
        for category in active_categories:

            # Load image
            image = dataset.load_image(image_id)

            # Draw random target
            target = []
            for k in range(model.config.NUM_TARGETS):
                try:
                    target.append(get_same_target(image_id, category, dataset, model.config))
                except:
                    print('error fetching target of category', category)
                    continue
            target = np.stack(target, axis=0)
            # Run detection
            t = time.time()
            try:
                r = model.detect([target], [image], verbose=0, random_detections=random_detections)[0]
            except:
                print('error running detection for category', category)
                continue
            t_prediction += (time.time() - t)

            # Format detections
            r["class_ids"] = np.array([category for i in range(r["class_ids"].shape[0])])

            # Convert results to COCO format
            # Cast masks to uint8 because COCO tools errors out on bool
            if dataset_type == 'coco':
                image_results = coco.build_coco_results(dataset, dataset_image_ids[i:i + 1],
                                                        r["rois"], r["class_ids"],
                                                        r["scores"],
                                                        r["masks"].astype(np.uint8))
            results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    dataset_results = dataset_object.loadRes(results)

    # allow evaluating bbox & segm:
    if not isinstance(eval_type, (list,)):
        eval_type = [eval_type]

    for current_eval_type in eval_type:
        # Evaluate
        cocoEval = customCOCOeval(dataset_object, dataset_results, current_eval_type)
        cocoEval.params.imgIds = dataset_image_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize(class_index=class_index, verbose=verbose)
        if verbose > 0:
            print("Prediction time: {}. Average {}/image".format(
                t_prediction, t_prediction / len(image_ids)))
            print("Total time: ", time.time() - t_start)

    if return_results:
        return cocoEval

    
### Visualization ###

def display_results(target, image, boxes, masks, class_ids,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        from matplotlib.gridspec import GridSpec
        # Use GridSpec to show target smaller than image
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 3)
        ax = plt.subplot(gs[:, 1:])
        target_ax = plt.subplot(gs[1, 0])
        auto_show = True

    # Generate random colors
    colors = colors or visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)
    
    target_height, target_width = target.shape[:2]
    target_ax.set_ylim(target_height + 10, -10)
    target_ax.set_xlim(-10, target_width + 10)
    target_ax.axis('off')
    # target_ax.set_title('target')

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = visualize.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{:.3f}".format(score) if score else 'no score'
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = visualize.apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = visualize.find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = visualize.Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    target_ax.imshow(target.astype(np.uint8))
    if auto_show:
        plt.show()
        
    return


def display_grid(target_list, image_list, boxes_list, masks_list, class_ids_list,
                      scores_list=None, category_names_list=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None,
                      show_scores=True,
                      target_shift=10, fontsize=14,
                      linewidth=2, save=False):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """

    if type(target_list) == list:
        M = int(np.sqrt(len(target_list)))
        if len(target_list) - M**2 > 1e-3:
            M = M + 1
    else:
        M = 1
        target_list = [target_list]
        image_list = [image_list]
        boxes_list = [boxes_list]
        masks_list = [masks_list]
        class_ids_list = [class_ids_list]
        if scores_list is not None:
            scores_list = [scores_list]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        from matplotlib.gridspec import GridSpec
        # Use GridSpec to show target smaller than image
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(M, M, hspace=0.1, wspace=0.02, left=0, right=1, bottom=0, top=1)
        # auto_show = True REMOVE

    index = 0
    for m1 in range(M):
        for m2 in range(M):
            ax = plt.subplot(gs[m1, m2])

            if index >= len(target_list):
                continue

            target = target_list[index]
            image = image_list[index]
            boxes = boxes_list[index]
            masks = masks_list[index]
            class_ids = class_ids_list[index]
            scores = scores_list[index]

            # Number of instances
            N = boxes.shape[0]
            if not N:
                print("\n*** No instances to display *** \n")
            else:
                assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

            # Generate random colors
            colors = visualize.random_colors(N)

            # Show area outside image boundaries.
            height, width = image.shape[:2]
            ax.set_ylim(height, 0)
            ax.set_xlim(0, width)
            ax.axis('off')
            ax.set_title(title)
            
            masked_image = image.astype(np.uint32).copy()
            for i in range(N):
                color = colors[i]

                # Bounding box
                if not np.any(boxes[i]):
                    # Skip this instance. Has no bbox. Likely lost in image cropping.
                    continue
                y1, x1, y2, x2 = boxes[i]
                if show_bbox:
                    p = visualize.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=linewidth,
                                        alpha=0.7, linestyle="dashed",
                                        edgecolor=color, facecolor='none')
                    ax.add_patch(p)

                # Label
                if not captions:
                    class_id = class_ids[i]
                    score = scores[i] if scores is not None else None
                    x = random.randint(x1, (x1 + x2) // 2)
                    caption = "{:.3f}".format(score) if score else 'no score'
                else:
                    caption = captions[i]
                if show_scores:
                    ax.text(x1, y1 + 8, caption,
                            color='w', size=int(10/14*fontsize), backgroundcolor="none")

                # Mask
                mask = masks[:, :, i]
                if show_mask:
                    masked_image = visualize.apply_mask(masked_image, mask, color)

                # Mask Polygon
                # Pad to ensure proper polygons for masks that touch image edges.
                padded_mask = np.zeros(
                    (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
                padded_mask[1:-1, 1:-1] = mask
                contours = visualize.find_contours(padded_mask, 0.5)
                for verts in contours:
                    # Subtract the padding and flip (y, x) to (x, y)
                    verts = np.fliplr(verts) - 1
                    p = visualize.Polygon(verts, facecolor="none", edgecolor=color)
                    ax.add_patch(p)
            ax.imshow(masked_image.astype(np.uint8))

            target_height, target_width = target.shape[:2]
            target_height = target_height // 2
            target_width = target_width // 2
            target_area = target_height * target_width
            target_scaling = np.sqrt((192//2*96//2) / target_area)
            target_height = int(target_height * target_scaling)
            target_width = int(target_width * target_scaling)
            ax.imshow(target, extent=[target_shift, target_shift + target_width * 2, height - target_shift, height - target_shift - target_height * 2], zorder=9)
            rect = visualize.patches.Rectangle((target_shift, height - target_shift), target_width * 2, -target_height * 2, linewidth=5, edgecolor='white', facecolor='none', zorder=10)
            ax.add_patch(rect)
            if category_names_list is not None:
                plt.title(category_names_list[index], fontsize=fontsize)
            index = index + 1

    if auto_show:
        plt.show()
        
    if save:
        fig.savefig('grid.pdf', bbox_inches='tight')
        
    return

