import json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras import backend as K
from mrcnn import utils
from skimage.measure import find_contours


def apply_mask2(image, mask):
    """Apply the given mask to the image.
    """
    alpha = 0.5
    for c in range(3):
        #image[:, :, c] = np.where(mask == 1, image[:, :, c], image[:, :, c])
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * 1,
                                  image[:, :, c])
    return image


def dice_coefficient(image,gt_boxes, gt_masks, pred_boxes, pred_masks):
    N = gt_boxes.shape[0]
    #masked_image_gt = image.astype(np.uint32).copy()
    #masked_image_gt = np.zeros(image.shape[1::-1])
    masked_image_gt = np.zeros(image.shape)
    '''
    plt.clf()
    plt.figure(2)
    plt.title("mask")
    plt.imshow(masked_image_gt, vmin=0, vmax=1)
    plt.show()'''

    for i in range(N):
        mask = gt_masks[:, :, i]
        '''plt.clf()
        plt.figure(3)
        plt.title("mask 2")
        plt.imshow(masked_image_gt, cmap='gray', vmin=0, vmax=1)
        plt.show()'''
        masked_image_gt = apply_mask2(masked_image_gt, mask)
        #plt.figure(4)
        #plt.imshow(masked_image_gt, cmap='gray', vmin=0, vmax=1)

    plt.clf()
    plt.figure(2)
    plt.title("mask gt")
    plt.imshow(masked_image_gt, cmap='gray', vmin=0, vmax=1)
    plt.show()


    M = pred_boxes.shape[0]
    mask_pred = np.zeros(image.shape)
    for i in range(M):
        mask = pred_masks[:, :, i]
        mask_pred = apply_mask2(mask_pred, mask)

    plt.clf()
    plt.figure(3)
    plt.title("mask pred")
    plt.imshow(mask_pred, cmap='gray', vmin=0, vmax=1)
    plt.show()

    # convert the pixel/mask matrix to a one-dimensional series
    predicted = mask_pred.flatten()
    truth = masked_image_gt.flatten()

    # our masks will consist of ones and zeros
    # summing the result of their product gives us the cross section
    overlap = np.sum(predicted * truth)
    total_surface_area = np.sum(predicted + truth)
    result = 2 * overlap / total_surface_area

    # passing our calculated values to the formula
    return result



def dice_coef3(gt_masks, pred_masks, class_ind):

    M = pred_masks.shape[-1]
    pred = np.zeros(pred_masks.shape[0:2])
    for i in range(M):
        pred = np.logical_or(pred, pred_masks[:,:,i])

    plt.clf()
    plt.figure(3)
    plt.title("mask pred")
    plt.imshow(pred, cmap='gray', vmin=0, vmax=1)
    plt.show()

    gt = np.zeros(pred_masks.shape[0:2])
    for i in class_ind:
        gt = np.logical_or(gt, gt_masks[:,:,i])
    plt.clf()
    plt.figure(4)
    plt.title("gt")
    plt.imshow(gt, cmap='gray', vmin=0, vmax=1)
    plt.show()

    intersection = np.sum(pred*gt)
    score = 2. * intersection.sum() / (gt.sum() + pred.sum())

    return score

def dice_coef(gt_masks, pred_masks, class_ind):

    M = pred_masks.shape[-1]
    pred = np.zeros(pred_masks.shape[0:2])
    for i in range(M):
        pred = np.logical_or(pred, pred_masks[:,:,i])

    gt = np.zeros(pred_masks.shape[0:2])
    for i in class_ind:
        gt = np.logical_or(gt, gt_masks[:, :, i])

    intersection = np.sum(pred*gt)
    score = 2. * intersection.sum() / (gt.sum() + pred.sum())

    return score

def dice_coef2(gt_masks, pred_masks, class_ind):

    M = pred_masks.shape[-1]
    pred = np.zeros(pred_masks.shape[0:2])
    for i in range(M):
        pred = np.logical_or(pred, pred_masks[:,:,i])

    N = gt_masks.shape[-1]
    gt = np.zeros(gt_masks.shape[0:2])
    for i in range(N):
        gt = np.logical_or(gt, gt_masks[:, :, i])

    intersection = np.sum(pred*gt)
    score = 2. * intersection.sum() / (gt.sum() + pred.sum())

    return score


def precision_score(gt_masks, pred_masks, class_ind):

    M = pred_masks.shape[-1]
    pred = np.zeros(pred_masks.shape[0:2])
    for i in range(M):
        pred = np.logical_or(pred, pred_masks[:, :, i])

    gt = np.zeros(pred_masks.shape[0:2])
    for i in class_ind:
        gt = np.logical_or(gt, gt_masks[:, :, i])

    intersect = np.sum(pred*gt)
    total_pixel_pred = np.sum(pred)
    precision = np.mean(intersect/total_pixel_pred)
    return round(precision, 3)



def recall_score(gt_masks, pred_masks, class_ind):
    M = pred_masks.shape[-1]
    pred = np.zeros(pred_masks.shape[0:2])
    for i in range(M):
        pred = np.logical_or(pred, pred_masks[:, :, i])

    gt = np.zeros(pred_masks.shape[0:2])
    for i in class_ind:
        gt = np.logical_or(gt, gt_masks[:, :, i])

    intersect = np.sum(pred*gt)
    total_pixel_truth = np.sum(gt)
    recall = np.mean(intersect/total_pixel_truth)
    return round(recall, 3)


def accuracy(gt_masks, pred_masks, class_ind):
    M = pred_masks.shape[-1]
    pred = np.zeros(pred_masks.shape[0:2])
    for i in range(M):
        pred = np.logical_or(pred, pred_masks[:, :, i])

    gt = np.zeros(pred_masks.shape[0:2])
    for i in class_ind:
        gt = np.logical_or(gt, gt_masks[:, :, i])

    intersect = np.sum(pred*gt)
    union = np.sum(pred) + np.sum(gt) - intersect
    xor = np.sum(gt==pred)
    acc = np.mean(xor/(union + xor - intersect))
    return round(acc, 3)



def iou(gt_masks, pred_masks, class_ind):
    M = pred_masks.shape[-1]
    pred = np.zeros(pred_masks.shape[0:2])
    for i in range(M):
        pred = np.logical_or(pred, pred_masks[:, :, i])

    gt = np.zeros(pred_masks.shape[0:2])
    for i in class_ind:
        gt = np.logical_or(gt, gt_masks[:, :, i])

    intersect = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersect
    iou = np.mean(intersect/union)
    return round(iou, 3)