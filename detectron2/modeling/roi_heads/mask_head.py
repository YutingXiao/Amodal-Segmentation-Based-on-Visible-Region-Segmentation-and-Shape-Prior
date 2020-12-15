# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import Image
import copy
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from .memo_functions import vq, vq_st
from typing import List
from detectron2.structures import Instances
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
ROI_MASK_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")
ROI_MASK_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def mask_rcnn_loss(pred_mask_logits, instances):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)
        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # num_regions*28*28
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)
    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0
    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)
    mask_loss = F.binary_cross_entropy_with_logits(
        pred_mask_logits, gt_masks.to(dtype=torch.float32), reduction="mean"
    )
    return mask_loss


def amodal_mask_rcnn_loss(pred_mask_logits, instances, weights=None, mode="amodal", version="n"):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        weights : weights of each instace
        version: n(normal), a(attention), g(ground truth)

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """

    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)
        if mode == "amodal":
            gt_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
        elif mode == "visible":
            gt_per_image = instances_per_image.gt_visible_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
        # num_regions*28*28
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len for amodal or visible
        gt_masks.append(gt_per_image)
    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0
    gt_masks = cat(gt_masks, dim=0)


    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5

    # Log the training accuracy for amodal mask(using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/{}_{}_accuracy".format(mode, version), mask_accuracy)
    storage.put_scalar("mask_rcnn/{}_{}_false_positive".format(mode, version), false_positive)
    storage.put_scalar("mask_rcnn/{}_{}_false_negative".format(mode, version), false_negative)

    if isinstance(weights, float):
        mask_loss = weights * F.binary_cross_entropy_with_logits(
            pred_mask_logits, gt_masks.to(dtype=torch.float32), reduction="mean"
        )
    else:
        mask_loss = F.binary_cross_entropy_with_logits(
            pred_mask_logits, gt_masks.to(dtype=torch.float32), weight=weights, reduction="mean"
        )

    return mask_loss


def mask_fm_loss(features, betas):
    assert len(features) != 0
    loss = 0
    betas = betas * 2 if len(features[0]) == 2 * len(betas) else betas
    if len(features) == 2:
        for f1, f2, beta in zip(features[0], features[1], betas):
            n = f1.size(0)
            loss += beta * torch.mean(1 - F.cosine_similarity(f1.view(n, -1), f2.view(n, -1).detach()))
            # loss += F.mse_loss(f1, f2.detach()) * beta / len(betas)
    if len(features) == 3:
        for f1, f2, f3, beta in zip(features[0], features[1], features[2], betas):
            n = f1.size(0)
            loss += (3 - F.cosine_similarity(f1.view(n, -1), f2.view(n, -1)) * beta +
                     F.cosine_similarity(f2.view(n, -1), f3.view(n, -1)) * beta +
                     F.cosine_similarity(f1.view(n, -1), f3.view(n, -1)) / 3)

    return loss


def mask_rcnn_inference(pred_mask_logits, pred_instances):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1

    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob  # (1, Hmask, Wmask)


def amodal_mask_rcnn_inference(multi_pred_mask_logits, pred_instances):
    # amodal processing
    pred_mask_logits_lst = []
    for i in multi_pred_mask_logits:
        pred_mask_logits_lst += [x for x in i]

    for i in range(len(pred_mask_logits_lst)):
        pred_mask_logits = pred_mask_logits_lst[i]
        cls_agnostic_mask = pred_mask_logits.size(1) == 1

        if cls_agnostic_mask:
            mask_probs_pred = pred_mask_logits.sigmoid()
        else:
            # Select masks corresponding to the predicted classes
            num_masks = pred_mask_logits.shape[0]
            class_pred = cat([i.pred_classes for i in pred_instances])
            indices = torch.arange(num_masks, device=class_pred.device)
            mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
        # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

        num_boxes_per_image = [len(i) for i in pred_instances]
        mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

        for prob, instances in zip(mask_probs_pred, pred_instances):
            if i == 0:
                instances.pred_amodal_masks = prob  # (1, Hmask, Wmask)
            elif i == 1:
                instances.pred_visible_masks = prob  # (1, Hmask, Wmask)
            elif i == 2:
                instances.pred_amodal2_masks = prob  # (1, Hmask, Wmask)
            elif i == 3:
                instances.pred_visible2_masks = prob  # (1, Hmask, Wmask)

    if len(multi_pred_mask_logits) >= 2:
        for instances in pred_instances:
            # instances.pred_amodal_ensemble_masks = (instances.pred_amodal_masks + instances.pred_amodal2_masks) / 2
            # instances.pred_visible_ensemble_masks = (instances.pred_visible_masks + instances.pred_visible2_masks) / 2

            a = instances.pred_amodal_masks
            b = instances.pred_amodal2_masks
            instances.pred_amodal_ensemble_masks = torch.relu(b - a) + a

            a = instances.pred_visible_masks
            b = instances.pred_visible2_masks
            instances.pred_visible_ensemble_masks = torch.relu(b - a) + a


class BaseMaskRCNNHead(nn.Module):
    """
    Implement the basic Mask R-CNN losses and inference logic.
    """

    def __init__(self, cfg, input_shape):
        super().__init__()
        self.vis_period = cfg.VIS_PERIOD

    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.
        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        x = self.layers(x)
        if self.training:
            return {"loss_mask": mask_rcnn_loss(x, instances, self.vis_period)}
        else:
            mask_rcnn_inference(x, instances)
            return instances

    def layers(self, x):
        """
        Neural network layers that makes predictions from input features.
        """
        raise NotImplementedError


@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(MaskRCNNConvUpsampleHead, self).__init__()

        # fmt: off
        self.cfg = cfg
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        # fmt: on

        self.conv_norm_relus = []

        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)

        self.deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        x = F.relu(self.deconv(x))
        return self.predictor(x)


@ROI_MASK_HEAD_REGISTRY.register()
class Amodal_Visible_Head(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(Amodal_Visible_Head, self).__init__()

        # fmt: off
        self.cfg = cfg
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        # num_vis_conv      = cfg.MODEL.ROI_MASK_HEAD.NUM_VIS_CONV
        self.cycle        = cfg.MODEL.ROI_MASK_HEAD.AMODAL_CYCLE
        self.fm = cfg.MODEL.ROI_MASK_HEAD.AMODAL_FEATURE_MATCHING
        self.fm_beta = cfg.MODEL.ROI_MASK_HEAD.AMODAL_FM_BETA
        self.MEMORY_REFINE = cfg.MODEL.ROI_MASK_HEAD.RECON_NET.MEMORY_REFINE
        self.version = cfg.MODEL.ROI_MASK_HEAD.VERSION
        input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        # fmt: on

        self.amodal_conv_norm_relus = []

        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("amodal_mask_fcn{}".format(k + 1), conv)
            self.amodal_conv_norm_relus.append(conv)

        self.amodal_deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.amodal_predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
        for layer in self.amodal_conv_norm_relus + [self.amodal_deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.amodal_predictor.weight, std=0.001)
        if self.amodal_predictor.bias is not None:
            nn.init.constant_(self.amodal_predictor.bias, 0)
        # self.amodal_pool = nn.MaxPool2d(kernel_size=2)
        self.amodal_pool = nn.AvgPool2d(kernel_size=2)
        self.visible_pool = nn.AvgPool2d(kernel_size=2)
        self.visible_conv_norm_relus = []

        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("visible_mask_fcn{}".format(k + 1), conv)
            self.visible_conv_norm_relus.append(conv)

        self.visible_deconv = ConvTranspose2d(
                conv_dims if num_conv > 0 else input_channels,
                conv_dims,
                kernel_size=2,
                stride=2,
                padding=0,
            )

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.visible_predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
        for layer in self.visible_conv_norm_relus + [self.visible_deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.visible_predictor.weight, std=0.001)
        if self.visible_predictor.bias is not None:
            nn.init.constant_(self.visible_predictor.bias, 0)

    def forward(self, x, instances):
        input_features = x
        output_mask_logits = []
        if self.version == 0:
            amodal_mask_logits, _ = self.single_head_forward(x, head="amodal")
            amodal_attention = self.amodal_pool(classes_choose(amodal_mask_logits, instances)).unsqueeze(1)
            visible_mask_logits, _ = self.single_head_forward(input_features * amodal_attention, head="visible")
            output_mask_logits.append([amodal_mask_logits, visible_mask_logits])
        elif self.version == 1:
            visible_mask_logits, _ = self.single_head_forward(x, head="visible")
            visible_attention = self.amodal_pool(classes_choose(visible_mask_logits, instances)).unsqueeze(1)
            amodal_mask_logits, _ = self.single_head_forward(input_features * visible_attention, head="amodal")
            output_mask_logits.append([amodal_mask_logits, visible_mask_logits])
        elif self.version == 2:
            # a2v2a2v
            amodal_mask_logits, _ = self.single_head_forward(x, head="amodal")
            amodal_attention = self.amodal_pool(classes_choose(amodal_mask_logits, instances)).unsqueeze(1)
            visible_mask_logits, _ = self.single_head_forward(input_features * amodal_attention, head="visible")
            output_mask_logits.append([amodal_mask_logits, visible_mask_logits])

            visible_attention = self.visible_pool(classes_choose(visible_mask_logits, instances)).unsqueeze(1)
            amodal_mask_logits, _ = self.single_head_forward(input_features * visible_attention, head="amodal")
            amodal_attention = self.amodal_pool(classes_choose(amodal_mask_logits, instances)).unsqueeze(1)
            visible_mask_logits, _ = self.single_head_forward(input_features * amodal_attention, head="visible")
            output_mask_logits.append([amodal_mask_logits, visible_mask_logits])

            mask_side_len = x.size(2)
            amodal_attention, visible_attention = get_gt_masks(instances, mask_side_len, output_mask_logits)
            amodal_mask_logits, _ = self.single_head_forward(input_features * visible_attention, head="amodal")
            visible_mask_logits, _ = self.single_head_forward(input_features * amodal_attention, head="visible")
            output_mask_logits.append([amodal_mask_logits, visible_mask_logits])
        elif self.version == 3:
            visible_mask_logits, _ = self.single_head_forward(x, head="visible")
            visible_attention = self.visible_pool(classes_choose(visible_mask_logits, instances)).unsqueeze(1)
            amodal_mask_logits, _ = self.single_head_forward(input_features * visible_attention, head="amodal")
            output_mask_logits.append([amodal_mask_logits, visible_mask_logits])

            amodal_attention = self.amodal_pool(classes_choose(amodal_mask_logits, instances)).unsqueeze(1)
            visible_mask_logits, _ = self.single_head_forward(input_features * amodal_attention, head="visible")
            visible_attention = self.visible_pool(classes_choose(visible_mask_logits, instances)).unsqueeze(1)
            amodal_mask_logits, _ = self.single_head_forward(input_features * visible_attention, head="amodal")
            output_mask_logits.append([amodal_mask_logits, visible_mask_logits])

            mask_side_len = x.size(2)
            amodal_attention, visible_attention = get_gt_masks(instances, mask_side_len, output_mask_logits)
            amodal_mask_logits, _ = self.single_head_forward(input_features * visible_attention, head="amodal")
            visible_mask_logits, _ = self.single_head_forward(input_features * amodal_attention, head="visible")
            output_mask_logits.append([amodal_mask_logits, visible_mask_logits])

        elif self.version == 4:
            amodal_mask_logits, _ = self.single_head_forward(x, head="amodal")
            visible_mask_logits, _ = self.single_head_forward(x, head="visible")
            output_mask_logits.append([amodal_mask_logits, visible_mask_logits])

            amodal_attention = self.amodal_pool(classes_choose(amodal_mask_logits, instances)).unsqueeze(1)
            visible_attention = self.visible_pool(classes_choose(visible_mask_logits, instances)).unsqueeze(1)

            amodal_mask_logits, _ = self.single_head_forward(input_features * amodal_attention, head="amodal")
            visible_mask_logits, _ = self.single_head_forward(input_features * visible_attention, head="visible")
            output_mask_logits.append([amodal_mask_logits, visible_mask_logits])

            mask_side_len = x.size(2)
            amodal_attention, visible_attention = get_gt_masks(instances, mask_side_len, output_mask_logits)
            amodal_mask_logits, _ = self.single_head_forward(input_features * visible_attention, head="amodal")
            visible_mask_logits, _ = self.single_head_forward(input_features * amodal_attention, head="visible")
            output_mask_logits.append([amodal_mask_logits, visible_mask_logits])

        return output_mask_logits, []

    def single_head_forward(self, x, head="amodal"):
        features = []
        i = 0
        if head == "amodal":
            for layer in self.amodal_conv_norm_relus:
                x = layer(x)
                if i in self.fm:
                    features.append(x)
                i += 1
            x = F.relu(self.amodal_deconv(x), inplace=True)
            if i in self.fm:
                features.append(x)
            mask_logits = self.amodal_predictor(x)

        elif head == "visible":
            for layer in self.visible_conv_norm_relus:
                x = layer(x)
                if i in self.fm:
                    features.append(x)
                i += 1
            x = F.relu(self.visible_deconv(x), inplace=True)
            if i in self.fm:
                features.append(x)
            mask_logits = self.visible_predictor(x)
        else:
            raise ValueError("Do not have this head")

        return mask_logits, features

    def _forward(self, x, instances):
        masks_logits1, feature_maps = self.forward_through(x, instances)

        if self.cycle:
            pred_logits1 = (classes_choose(masks_logits1[0], instances), classes_choose(masks_logits1[1], instances))
            x = x * (self.visible_pool(pred_logits1[1]) > 0).unsqueeze(1)
            masks_logits2, feature_maps2 = self.forward_through(x, instances)
            return (masks_logits1[0], masks_logits1[1], masks_logits2[0], masks_logits2[1]),\
                   (feature_maps, feature_maps2)
        else:
            return masks_logits1, []

    def _forward_through(self, x, instances):
        input_features = x
        feature_maps = []
        for layer in self.amodal_conv_norm_relus:
            x = layer(x)
            feature_maps.append(x)
        x = F.relu(self.amodal_deconv(x), inplace=True)
        # x = F.relu(x)
        amodal_mask_logits = self.amodal_predictor(x)

        amodal_attention = self.amodal_pool(classes_choose(amodal_mask_logits, instances))

        x = (amodal_attention > 0.0).unsqueeze(1) * input_features

        for layer in self.visible_conv_norm_relus:
            x = layer(x)
        x = F.relu(self.visible_deconv(x), inplace=True)
        visible_mask_logits = self.visible_predictor(x)
        return (amodal_mask_logits, visible_mask_logits), [feature_maps[i] for i in self.fm]


@ROI_MASK_HEAD_REGISTRY.register()
class Parallel_Amodal_Visible_Head(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(Parallel_Amodal_Visible_Head, self).__init__()

        # fmt: off
        self.cfg = cfg
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        # num_vis_conv      = cfg.MODEL.ROI_MASK_HEAD.NUM_VIS_CONV
        self.fm           = cfg.MODEL.ROI_MASK_HEAD.AMODAL_FEATURE_MATCHING
        self.fm_beta      = cfg.MODEL.ROI_MASK_HEAD.AMODAL_FM_BETA
        input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        self.SPRef        = cfg.MODEL.ROI_MASK_HEAD.RECON_NET.MEMORY_REFINE
        self.SPk          = cfg.MODEL.ROI_MASK_HEAD.RECON_NET.MEMORY_REFINE_K
        self.version      = cfg.MODEL.ROI_MASK_HEAD.VERSION
        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        self.attention_mode = cfg.MODEL.ROI_MASK_HEAD.ATTENTION_MODE
        # fmt: on

        self.amodal_conv_norm_relus = []
        self.visible_conv_norm_relus = []
        for k in range(num_conv):
            a_conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("amodal_mask_fcn{}".format(k + 1), a_conv)
            self.amodal_conv_norm_relus.append(a_conv)

            v_conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("visible_mask_fcn{}".format(k + 1), v_conv)
            self.visible_conv_norm_relus.append(v_conv)

        self.amodal_deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.visible_deconv = ConvTranspose2d(
                conv_dims if num_conv > 0 else input_channels,
                conv_dims,
                kernel_size=2,
                stride=2,
                padding=0,
            )

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.amodal_predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
        self.visible_predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)

        nn.init.normal_(self.amodal_predictor.weight, std=0.001)
        if self.amodal_predictor.bias is not None:
            nn.init.constant_(self.amodal_predictor.bias, 0)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.visible_predictor.weight, std=0.001)
        if self.visible_predictor.bias is not None:
            nn.init.constant_(self.visible_predictor.bias, 0)

        for layer in self.amodal_conv_norm_relus + [self.amodal_deconv] + self.visible_conv_norm_relus + [self.visible_deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        # self.amodal_pool = nn.MaxPool2d(kernel_size=2)
        self.amodal_pool = nn.AvgPool2d(kernel_size=2)
        self.visible_pool = nn.AvgPool2d(kernel_size=2)

        if self.SPRef:
            self.fuse_layer = Conv2d(
                input_channels + self.cfg.MODEL.ROI_MASK_HEAD.RECON_NET.MEMORY_REFINE_K,
                input_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )

    def forward(self, x, instances=None):

        output_mask_logits = []
        output_feature = []
        masks_logits, feature_maps = self.forward_through(x, x)
        output_mask_logits.append(masks_logits)
        output_feature.append(feature_maps)

        if masks_logits[0].size(0) == 0:
            return output_mask_logits, output_feature
        # comparsion

        if self.version == 1:
            amodal_attention = self.amodal_pool(classes_choose(masks_logits[0], instances)).unsqueeze(1).sigmoid()

            visible_masks_logits_, visible_feature_maps_ = self.single_head_forward(x * amodal_attention, "visible")

            if self.SPRef:
                pred_amodal_masks = classes_choose(masks_logits[0], instances).unsqueeze(1)
                nn_latent_vectors = self.recon_net.encode(pred_amodal_masks).view(pred_amodal_masks.size(0), -1)

                if instances[0].has("gt_classes"):
                    shape_prior = self.recon_net.nearest_decode(nn_latent_vectors,
                                                                cat([i.gt_classes for i in instances], dim=0),
                                                                k=self.SPk).detach()
                else:
                    shape_prior = self.recon_net.nearest_decode(nn_latent_vectors,
                                                                cat([i.pred_classes for i in instances], dim=0),
                                                                k=self.SPk).detach()
                shape_prior = F.avg_pool2d(shape_prior, 2)
                amodal_masks_logits_, amodal_feature_maps_ = self.single_head_forward(
                    self.fuse_layer(cat([x, shape_prior], dim=1)), "amodal")
            else:
                amodal_masks_logits_, amodal_feature_maps_ = self.single_head_forward(x, "amodal")

            output_mask_logits.append([amodal_masks_logits_, visible_masks_logits_])

            if instances[0].has("gt_masks"):
                mask_side_len = x.size(2)
                amodal_attention, _ = get_gt_masks(instances, mask_side_len, masks_logits)

                visible_masks_logits_, visible_feature_maps_ = self.single_head_forward(x * amodal_attention, "visible")

                output_mask_logits.append([visible_masks_logits_])

                output_feature.append(visible_feature_maps_)

        elif self.version == 2:
            # amodal_attention = self.amodal_pool(classes_choose(masks_logits[0], instances)).unsqueeze(1).sigmoid()

            visible_attention = self.visible_pool(classes_choose(masks_logits[1], instances)).unsqueeze(1).sigmoid()

            visible_masks_logits_, visible_feature_maps_ = self.single_head_forward(x * visible_attention, "visible")
            visible_attention = self.visible_pool(classes_choose(visible_masks_logits_, instances)).unsqueeze(1).sigmoid()

            amodal_masks_logits_, amodal_feature_maps_ = self.single_head_forward(x * visible_attention, "amodal")

            output_mask_logits.append([amodal_masks_logits_, visible_masks_logits_])

            if instances[0].has("gt_masks"):
                mask_side_len = x.size(2)
                _, visible_attention = get_gt_masks(instances, mask_side_len, masks_logits)

                amodal_masks_logits_, amodal_feature_maps_ = self.single_head_forward(x * visible_attention, "amodal")
                visible_masks_logits_, visible_feature_maps_ = self.single_head_forward(x * visible_attention, "visible")

                output_mask_logits.append([amodal_masks_logits_, visible_masks_logits_])

                output_feature.append(amodal_feature_maps_ + visible_feature_maps_)

        elif self.version == 3:
            amodal_attention = (self.amodal_pool(classes_choose(masks_logits[0], instances)).unsqueeze(1) > 0).float() \
                if self.attention_mode == "mask" else self.amodal_pool(classes_choose(masks_logits[0], instances)).unsqueeze(1).sigmoid()

            visible_masks_logits_, visible_feature_maps_ = self.single_head_forward(x * amodal_attention, "visible")
            visible_attention = (self.visible_pool(classes_choose(visible_masks_logits_, instances)).unsqueeze(1) > 0).float() \
                if self.attention_mode == "mask" else self.visible_pool(classes_choose(visible_masks_logits_, instances)).unsqueeze(1).sigmoid()

            if self.SPRef:
                pred_amodal_masks = classes_choose(masks_logits[0], instances).unsqueeze(1)
                nn_latent_vectors = self.recon_net.encode(pred_amodal_masks).view(pred_amodal_masks.size(0), -1)
                if instances[0].has("gt_classes"):
                    shape_prior = self.recon_net.nearest_decode(nn_latent_vectors,
                                                                cat([i.gt_classes for i in instances], dim=0),
                                                                k=self.SPk).detach()
                else:
                    shape_prior = self.recon_net.nearest_decode(nn_latent_vectors,
                                                                cat([i.pred_classes for i in instances], dim=0),
                                                                k=self.SPk).detach()
                shape_prior = F.avg_pool2d(shape_prior, 2)
                amodal_masks_logits_, amodal_feature_maps_ = self.single_head_forward(
                    self.fuse_layer(cat([x * visible_attention, shape_prior], dim=1)), "amodal")
            else:
                amodal_masks_logits_, amodal_feature_maps_ = self.single_head_forward(x * visible_attention, "amodal")

            output_mask_logits.append([amodal_masks_logits_, visible_masks_logits_])

            if instances[0].has("gt_masks"):
                mask_side_len = x.size(2)
                amodal_attention, visible_attention = get_gt_masks(instances, mask_side_len, masks_logits)

                if self.SPRef:
                    amodal_masks_logits_, amodal_feature_maps_ = self.single_head_forward(
                        self.fuse_layer(cat([x * visible_attention, shape_prior], dim=1)), "amodal")
                else:
                    amodal_masks_logits_, amodal_feature_maps_ = self.single_head_forward(x * visible_attention, "amodal")
                visible_masks_logits_, visible_feature_maps_ = self.single_head_forward(x * amodal_attention, "visible")

                output_mask_logits.append([amodal_masks_logits_, visible_masks_logits_])

                output_feature.append(amodal_feature_maps_ + visible_feature_maps_)

        return output_mask_logits, output_feature

    def forward_through(self, x1, x2):
        features = []
        i = 0
        for layer in self.amodal_conv_norm_relus:
            x1 = layer(x1)
            if i in self.fm:
                features.append(x1)
            i += 1
        x1 = F.relu(self.amodal_deconv(x1), inplace=True)
        if i in self.fm:
            features.append(x1)
        amodal_mask_logits = self.amodal_predictor(x1)

        i = 0
        for layer in self.visible_conv_norm_relus:
            x2 = layer(x2)
            if i in self.fm:
                features.append(x2)
            i += 1
        x2 = F.relu(self.visible_deconv(x2), inplace=True)
        if i in self.fm:
            features.append(x2)
        visible_mask_logits = self.visible_predictor(x2)

        return [amodal_mask_logits, visible_mask_logits], features

    def single_head_forward(self, x, head="amodal"):
        features = []
        i = 0
        if head == "amodal":
            for layer in self.amodal_conv_norm_relus:
                x = layer(x)
                if i in self.fm:
                    features.append(x)
                i += 1
            x = F.relu(self.amodal_deconv(x), inplace=True)
            if i in self.fm:
                features.append(x)
            mask_logits = self.amodal_predictor(x)

        elif head == "visible":
            for layer in self.visible_conv_norm_relus:
                x = layer(x)
                if i in self.fm:
                    features.append(x)
                i += 1
            x = F.relu(self.visible_deconv(x), inplace=True)
            if i in self.fm:
                features.append(x)
            mask_logits = self.visible_predictor(x)
        else:
            raise ValueError("Do not have this head")

        return mask_logits, features

    def shape_prior_ref_forward(self, x, refined_visible_logits, shape_prior, instances=None):
        shape_prior = F.avg_pool2d(shape_prior, 2)
        visible_attention = self.amodal_pool(classes_choose(refined_visible_logits, instances)).unsqueeze(1).sigmoid()
        x = self.fuse_layer(cat([x * visible_attention, shape_prior], dim=1))
        amodal_masks_logits, _ = self.single_head_forward(x, "amodal")

        return amodal_masks_logits


def classes_choose(logits, instances_cls):
    cls_agnostic_mask = logits.size(1) == 1
    total_num_masks = logits.size(0)

    if isinstance(instances_cls, list):
        assert logits.size(0) == sum(len(x) for x in instances_cls)

        classes_label = []
        for instances_per_image in instances_cls:
            if len(instances_per_image) == 0:
                continue
            if not cls_agnostic_mask:
                if instances_per_image.has("gt_classes"):
                    classes_label_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
                elif instances_per_image.has("pred_classes"):
                    classes_label_per_image = instances_per_image.pred_classes.to(dtype=torch.int64)
                else:
                    raise ValueError("classes label missing")
                classes_label.append(classes_label_per_image)
        try:
            classes_label = cat(classes_label, dim=0)
        except:
            pass
    else:
        assert logits.size(0) == instances_cls.size(0)
        classes_label = instances_cls

    if cls_agnostic_mask:
        pred_logits = logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        pred_logits = logits[indices, classes_label]

    return pred_logits


def get_gt_masks(instances, mask_side_len, pred_mask_logits):
    amodal_gt_masks = []
    visible_gt_masks = []

    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue

        amodal_gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits[0].device)
        # num_regions*28*28
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        amodal_gt_masks.append(amodal_gt_masks_per_image)

        visible_gt_masks_per_image = instances_per_image.gt_visible_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits[0].device)
        # num_regions*28*28
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        visible_gt_masks.append(visible_gt_masks_per_image)
    if len(amodal_gt_masks) == 0:
        return pred_mask_logits[0].sum() * 0
    amodal_gt_masks = cat(amodal_gt_masks, dim=0).unsqueeze(1)
    visible_gt_masks = cat(visible_gt_masks, dim=0).unsqueeze(1)
    #
    # vis.images(amodal_gt_masks, win_name="amodal_gt_masks", nrow=16)
    # vis.images(visible_gt_masks, win_name="visible_gt_masks", nrow=16)
    return amodal_gt_masks, visible_gt_masks


class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar


class ResBlock(nn.Module):
    def __init__(self, dim, norm="BN", **kwargs):
        super().__init__()
        num_groups = kwargs.pop("num_group", None)
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            get_norm(norm, dim) if not num_groups else get_norm(norm, dim, num_groups=num_groups),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            get_norm(norm, dim) if not num_groups else get_norm(norm, dim, num_groups=num_groups),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return x + self.block(x)


def build_mask_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)

