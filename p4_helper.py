import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.ops import box_iou
import sys, os
import json
import random
import cv2
from PIL import Image
import trimesh
import pyrender
import tqdm

_HOUGHVOTING_NUM_INLIER = 100
_HOUGHVOTING_DIRECTION_INLIER = 0.9
_LABEL2MASK_THRESHOL = 500

def hello_helper():
    print("Hello from p4_helper.py!")


def loss_cross_entropy(scores, labels):
    """
    scores: a tensor [batch_size, num_classes, height, width]
    labels: a tensor [batch_size, num_classes, height, width]
    """

    cross_entropy = -torch.sum(labels * torch.log(scores + 1e-10), dim=1)
    loss = torch.div(torch.sum(cross_entropy), torch.sum(labels)+1e-10)

    return loss

def loss_Rotation(pred_R, gt_R, label, model):
    """
    pred_R: a tensor [N, 3, 3]
    gt_R: a tensor [N, 3, 3]
    label: a tensor [N, ]
    model: a tensor [N_cls, 1024, 3]
    """
    device = pred_R.device
    models_pcd = model[label - 1].to(device)
    gt_points = models_pcd @ gt_R
    pred_points = models_pcd @ pred_R
    loss = ((pred_points - gt_points) ** 2).sum(dim=2).sqrt().mean()
    return loss


def IOUselection(pred_bbxes, gt_bbxes, threshold):
    """
        pred_bbx is N_pred_bbx * 6 (batch_ids, x1, y1, x2, y2, cls)
        gt_bbx is gt_bbx * 6 (batch_ids, x1, y1, x2, y2, cls)
        threshold : threshold of IOU for selection of predicted bbx
    """
    device = pred_bbxes.device
    output_bbxes = torch.empty((0, 6)).to(device = device, dtype =torch.float)
    for pred_bbx in pred_bbxes:
        for gt_bbx in gt_bbxes:
            if pred_bbx[0] == gt_bbx[0] and pred_bbx[5] == gt_bbx[5]:
                iou = box_iou(pred_bbx[1:5].unsqueeze(dim=0), gt_bbx[1:5].unsqueeze(dim=0)).item()
                if iou > threshold:
                    output_bbxes = torch.cat((output_bbxes, pred_bbx.unsqueeze(dim=0)), dim=0)
    return output_bbxes


def HoughVoting(label, centermap, num_classes=10):
    """
    label [bs, 3, H, W]
    centermap [bs, 3*maxinstance, H, W]
    """
    batches, H, W = label.shape
    x = np.linspace(0, W - 1, W)
    y = np.linspace(0, H - 1, H)
    xv, yv = np.meshgrid(x, y)
    xy = torch.from_numpy(np.array((xv, yv))).to(device = label.device, dtype=torch.float32)
    x_index = torch.from_numpy(x).to(device = label.device, dtype=torch.int32)
    centers = torch.zeros(batches, num_classes, 2)
    depths = torch.zeros(batches, num_classes)
    for bs in range(batches):
        for cls in range(1, num_classes + 1):
            if (label[bs] == cls).sum() >= _LABEL2MASK_THRESHOL:
                pixel_location = xy[:2, label[bs] == cls]
                pixel_direction = centermap[bs, (cls-1)*3:cls*3][:2, label[bs] == cls]
                y_index = x_index.unsqueeze(dim=0) - pixel_location[0].unsqueeze(dim=1)
                y_index = torch.round(pixel_location[1].unsqueeze(dim=1) + (pixel_direction[1]/pixel_direction[0]).unsqueeze(dim=1) * y_index).to(torch.int32)
                mask = (y_index >= 0) * (y_index < H)
                count = y_index * W + x_index.unsqueeze(dim=0)
                center, inlier_num = torch.bincount(count[mask]).argmax(), torch.bincount(count[mask]).max()
                center_x, center_y = center % W, torch.div(center, W, rounding_mode='trunc')
                if inlier_num > _HOUGHVOTING_NUM_INLIER:
                    centers[bs, cls - 1, 0], centers[bs, cls - 1, 1] = center_x, center_y
                    xyplane_dis = xy - torch.tensor([center_x, center_y])[:, None, None].to(device = label.device)
                    xyplane_direction = xyplane_dis/(xyplane_dis**2).sum(dim=0).sqrt()[None, :, :]
                    predict_direction = centermap[bs, (cls-1)*3:cls*3][:2]
                    inlier_mask = ((xyplane_direction * predict_direction).sum(dim=0).abs() >= _HOUGHVOTING_DIRECTION_INLIER) * label[bs] == cls
                    depths[bs, cls - 1] = centermap[bs, (cls-1)*3:cls*3][2, inlier_mask].mean()
    return centers, depths



