# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.utils.registry import Registry
from detectron2.layers import cat
from detectron2.utils.events import get_event_storage

from ..postprocessing import detector_postprocess
RECLS_NET_REGISTRY = Registry("RECLS_NET")
RECLS_NET_REGISTRY.__doc__ = """
Registry for recls heads, which make recls predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def mask_recls_filter_loss(recls, pred_mask_logits, mask_features, instances, box_ths=0.8, mask_ths=0.95, gt_weight=0.1,
                           pre_logits=[]):
    # cls_agnostic_mask = pred_mask_logits.size(1) == 1
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"
    index = [0]

    pred_classes_lst = []
    gt_classes_lst = []
    pred_attention_lst = []
    gt_attention_lst = []
    pre_logits_lst = []

    pred_attention_features = mask_features * F.avg_pool2d(pred_mask_logits.detach(), 2)
    for instances_per_image in instances:
        arange = torch.arange(len(instances_per_image))
        iou_box = pairwise_iou(instances_per_image.proposal_boxes, instances_per_image.gt_boxes)[arange, arange]

        gt_visible_per_image = instances_per_image.gt_visible_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).unsqueeze(1).to(device=pred_mask_logits.device)

        pred_masks_per_image = pred_mask_logits[index[-1]: index[-1] + len(instances_per_image)] > 0
        # pred_masks_per_image = pred_masks_per_image[arange, instances_per_image.gt_classes]

        # iou_mask = torch.sum((gt_visible_per_image * pred_masks_per_image) > 0, dim=(1, 2, 3)).float() / \
        #            torch.sum((gt_visible_per_image + pred_masks_per_image) > 0, dim=(1, 2, 3)).float()
        recall_mask = torch.sum((gt_visible_per_image * pred_masks_per_image) > 0, dim=(1, 2, 3)).float() / \
                     torch.sum(gt_visible_per_image, dim=(1, 2, 3)).float()
        # filter_inds = ((iou_box > box_ths) * (iou_mask > mask_ths)).nonzero()
        filter_inds = ((iou_box > box_ths) * (recall_mask > mask_ths)).nonzero()

        pred_classes = instances_per_image.gt_classes

        if len(pre_logits):
            pre_logits_lst.append(pre_logits[index[-1]: index[-1] + len(instances_per_image)][filter_inds[:, 0]])

        pred_attention_lst.append(pred_attention_features[index[-1]: index[-1] + len(instances_per_image)][filter_inds[:, 0]])
        pred_classes_lst.append(pred_classes[filter_inds[:, 0]])

        visible_features = mask_features[index[-1]: index[-1] + len(instances_per_image)] * F.avg_pool2d(gt_visible_per_image.float(), 2)
        gt_attention_lst.append(visible_features)
        gt_classes_lst.append(pred_classes)

        index.append(len(instances_per_image))

    pred_attention_features = cat(pred_attention_lst, dim=0)
    gt_attention_features = cat(gt_attention_lst, dim=0)
    pred_classes = cat(pred_classes_lst, dim=0)
    gt_classes = cat(gt_classes_lst, dim=0)

    # pre_logits = cat(pre_logits_lst, dim=0) if len(pre_logits) else []
    num = pred_attention_features.size(0) + gt_attention_features.size(0)
    loss = 0
    if pred_attention_features.size(0):
        repred_class_logits = recls(pred_attention_features)
        repred_classes = torch.argmax(repred_class_logits, dim=1)
        acc = torch.mean((repred_classes == pred_classes).float())
        storage = get_event_storage()
        storage.put_scalar("recls/cls_accuracy(pred)", acc)

        loss = F.cross_entropy(repred_class_logits, pred_classes, reduction="sum") / num

    if gt_attention_features.size(0):
        repred_class_logits = recls(gt_attention_features)
        repred_classes = torch.argmax(repred_class_logits, dim=1)
        acc = torch.mean((repred_classes == gt_classes).float())
        storage = get_event_storage()
        storage.put_scalar("recls/cls_accuracy(gt)", acc)

        loss += (F.cross_entropy(repred_class_logits, gt_classes, reduction="sum") / num) * gt_weight
    # if len(pre_logits):
    #     pred_mask_classes = cat(pred_classes_lst, dim=0)
    #     indices = torch.arange(pred_mask_classes.size(0))
    #     repred_class_prob = F.softmax(repred_class_logits, dim=-1)[indices, pred_mask_classes]
    #     pred_class_prob = F.softmax(pre_logits, dim=-1)[indices, pred_mask_classes]
    #     loss += torch.mean(F.relu(pred_class_prob - repred_class_prob + 0.05, inplace=True))
    return loss


def mask_recls_adaptive_loss(recls, pred_mask_logits, mask_features, instances, gt_weight=0.1):
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"
    index = 0
    classes_lst = []
    feature_attention_lst = []
    weight_lst = []
    # mode = recls.attention_mode
    if recls.attention_mode == "attention":
        attention_features = mask_features * F.avg_pool2d(pred_mask_logits.detach(), 2)
    elif recls.attention_mode == "mask":
        attention_features = mask_features * F.avg_pool2d((pred_mask_logits.detach() > 0).float(), 2)
    # else:
    #     raise ValueError("Wrong mode")
    for instances_per_image in instances:
        # arange = torch.arange(len(instances_per_image))
        # iou_box = pairwise_iou(instances_per_image.proposal_boxes, instances_per_image.gt_boxes)[arange, arange]

        gt_visible_per_image = instances_per_image.gt_visible_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).unsqueeze(1).to(device=pred_mask_logits.device)

        pred_masks_per_image = pred_mask_logits[index: index + len(instances_per_image)] > 0
        # pred_masks_per_image = pred_masks_per_image[arange, instances_per_image.gt_classes]

        # iou_mask = torch.sum((gt_visible_per_image * pred_masks_per_image) > 0, dim=(1, 2, 3)).float() / \
        #            torch.sum((gt_visible_per_image + pred_masks_per_image) > 0, dim=(1, 2, 3)).float()
        iou_mask = torch.sum((gt_visible_per_image * pred_masks_per_image) > 0, dim=(1, 2, 3)).float() / \
                   torch.sum((gt_visible_per_image + pred_masks_per_image) > 0, dim=(1, 2, 3)).float()
        weight_lst.append(iou_mask)
        feature_attention_lst.append(attention_features[index: index + len(instances_per_image)])
        classes_lst.append(instances_per_image.gt_classes)

        if gt_weight != 0:
            weight_lst.append(torch.ones(gt_visible_per_image.size(0)).cuda() * gt_weight)
            feature_attention_lst.append(mask_features[index: index + len(instances_per_image)] * F.avg_pool2d(gt_visible_per_image.float(), 2))
            classes_lst.append(instances_per_image.gt_classes)

        index += len(instances_per_image)

    attention_features = cat(feature_attention_lst, dim=0)
    classes = cat(classes_lst, dim=0)
    weights = cat(weight_lst, dim=0)
    indices = (~torch.isfinite(weights)).nonzero()
    weights[indices[:, 0]] = 0

    repred_class_logits = recls(attention_features)
    repred_classes = torch.argmax(repred_class_logits, dim=1)
    acc = torch.mean((repred_classes == classes).float())
    storage = get_event_storage()
    storage.put_scalar("recls/cls_accuracy", acc)
    loss = torch.mean(F.cross_entropy(repred_class_logits, classes, reduction="none") * weights)

    return loss


def mask_recls_margin_loss(repred_class_logits, proposals, pred_class_logits, margin=0.1):
    assert pred_class_logits.size(0) == repred_class_logits.size(0) == sum(len(i) for i in proposals)

    gt_classes = cat([i.gt_classes for i in proposals], dim=0)
    indices = torch.arange(gt_classes.size(0))

    repred_class_prob = F.softmax(repred_class_logits, dim=-1)[indices, gt_classes]
    pred_class_prob = F.softmax(pred_class_logits, dim=-1)[indices, gt_classes]

    return torch.mean(F.relu(pred_class_prob - repred_class_prob + margin, inplace=True))


@RECLS_NET_REGISTRY.register()
class ReclsConvFCHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        num_conv   = cfg.MODEL.ROI_MASK_HEAD.RECLS_NET.NUM_CONV
        conv_dim   = cfg.MODEL.ROI_MASK_HEAD.RECLS_NET.CONV_DIM
        num_fc     = cfg.MODEL.ROI_MASK_HEAD.RECLS_NET.NUM_FC
        fc_dim     = cfg.MODEL.ROI_MASK_HEAD.RECLS_NET.FC_DIM
        norm       = cfg.MODEL.ROI_MASK_HEAD.RECLS_NET.NORM
        self.rescoring = cfg.MODEL.ROI_MASK_HEAD.RECLS_NET.RESCORING
        self.attention_mode = cfg.MODEL.ROI_MASK_HEAD.ATTENTION_MODE
        # fmt: on
        assert num_conv + num_fc > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        cls = nn.Linear(fc_dim, num_classes)
        self.add_module("recls", cls)
        self._output_size = num_classes

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)
        nn.init.normal_(self.recls.weight, std=0.01)
        nn.init.constant_(self.recls.bias, 0)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x), inplace=True)
        x = self.recls(x)
        return x

    @property
    def output_size(self):
        return self._output_size


def build_recls_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.RECLS_NET.NAME
    return RECLS_NET_REGISTRY.get(name)(cfg, input_shape)
