# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import torch
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
import copy

from ..postprocessing import detector_postprocess

logger = logging.getLogger(__name__)
"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def extract_edge(masks_logits):
    kernel_size = 7
    masks = (masks_logits > 0.5).float()
    f = torch.ones((1, 1, kernel_size, kernel_size)).float()
    inside = F.conv2d(masks, f.to(masks.device), padding=3) >= 49
    masks = masks - inside.float()

    return masks


def get_similarity(pred_masks, recon_net, filter_inds, metric="l1", post_process="normal"):
    if recon_net.name == "VQVAE":
        recon_logits = recon_net(pred_masks, filter_inds[:, 1])
    elif recon_net.name == "CVAE":
        recon_logits = recon_net(pred_masks, filter_inds[:, 1])
    else:
        latent_vectors = recon_net.encode(pred_masks).view(filter_inds.size(0), -1)

        feature_side_len = np.sqrt(latent_vectors.size(1) / recon_net.conv_dims)
        assert feature_side_len % 1 == 0
        recon_logits = recon_net.nearest_decode(latent_vectors, filter_inds[:, 1])

        # recon_net.pred_mask_logits = recon_logits

    if post_process == "normal":
        recon_masks = (recon_logits > 0.5).float()
    elif post_process == "edge":
        recon_masks = extract_edge(recon_logits)
    else:
        raise ValueError("post process not found")

    # similiarity = torch.mean(torch.abs(pred_masks - (recon_logits > 0.5).float()).view(recon_logits.size(0), -1), dim=1)
    if metric == "iou":
        similiarity = torch.sum((pred_masks * recon_masks) > 0, dim=(1, 2, 3)).float() /\
                      torch.sum((pred_masks + recon_masks) > 0, dim=(1, 2, 3)).float()
    elif metric == "l1":
        similiarity = torch.mean(torch.abs(pred_masks - recon_masks).view(recon_logits.size(0), -1), dim=1)
    else:
        raise ValueError("Metrics Wrong!")
    return similiarity, recon_logits


def fast_rcnn_inference(boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image,
                        mask_pooler=None, mask_head=None, recon_net=None, features=None, recon_alpha=None, recls=None):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputs.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputs.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    if mask_pooler is not None and mask_head is not None:

        result_per_image = [
            fast_rcnn_inference_single_image_recon_recls(
                boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh,
                topk_per_image, features, mask_pooler, mask_head, recon_net, recon_alpha, recls
            )
            for scores_per_image, boxes_per_image, image_shape
            in zip(scores, boxes, image_shapes)
        ]

    else:
        result_per_image = [
            fast_rcnn_inference_single_image(
                boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
            )
            for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
        ]
    return tuple(list(x) for x in zip(*result_per_image))


def fast_rcnn_inference_single_image(
    boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]

    scores = scores[filter_mask]
    # Apply per-class NMS
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


def fast_rcnn_inference_single_image_recon_recls(
    boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image, features, mask_pooler, mask_head,
        recon_net=None, alpha=2, recls=None
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]

    scores = scores[filter_mask]

    # apply recon net
    mask_features = mask_pooler(features, [Boxes(boxes)])
    if mask_head.cfg.MODEL.ROI_HEADS.NAME == "StandardROIHeads":
        pred_mask_logits = mask_head(mask_features)
    else:
        results = Instances(image_shape)
        results.pred_classes = filter_inds[:, 1]

        pred_mask_logits, _, = mask_head(mask_features, [results])

    n = 1
    if recls and pred_mask_logits[0][1].size(0) != 0:
        if recls.rescoring:
            pred_visible_mask_logits = pred_mask_logits[1][1] if len(pred_mask_logits) > 1 else pred_mask_logits[0][1]
            pred_visible_mask_logits = get_pred_masks_logits_by_cls(pred_visible_mask_logits, filter_inds[:, 1])
            if recls.attention_mode == "mask":
                recls_logits = recls(mask_features * F.avg_pool2d((pred_visible_mask_logits > 0).float(), 2))
            else:
                recls_logits = recls(mask_features * F.avg_pool2d(pred_visible_mask_logits, 2))

            recls_prob = torch.softmax(recls_logits, dim=1)

            indices = torch.arange(recls_prob.size(0), device=recls_prob.device)
            # filter_inds[:, 1] = torch.argmax(recls_logits, dim=1)
            # scores = scores * (recls_logits[0][indices, filter_inds[:, 1]] * 0.3 + 0.7)
            scores = scores * (recls_prob[indices, filter_inds[:, 1]] * 0.4 + 0.6)
            n += 1

    if recon_net and pred_mask_logits[0][0].size(0):
        if recon_net.rescoring:
            mode = "normal"

            select = 1 if len(pred_mask_logits) == 2 else 0
            indices = torch.arange(pred_mask_logits[select][0].size(0), device=pred_mask_logits[select][0].device)
            pred_masks = (pred_mask_logits[select][0][indices, filter_inds[:, 1]] > 0).unsqueeze(1).float()
            similiarity, recon_logits = get_similarity(pred_masks, recon_net, filter_inds, post_process=mode)

            # similiarity_filter_l = ((scores > 0.6) * (similiarity > 0.8)).nonzero()
            # similiarity_filter_s = ((scores > 0.6) * (similiarity < 0.5)).nonzero()
            # if 64 > len(similiarity_filter_l) > 0:
            #     vis.images(cat([pred_masks[similiarity_filter_l[:, 0]], recon_logits[similiarity_filter_l[:, 0]]], dim=0),
            #                win_name="large similiarity:{}".format(len(similiarity_filter_l)),
            #                nrow=len(similiarity_filter_l[:, 0]))
            # if 64 > len(similiarity_filter_s) > 0:
            #     vis.images(cat([pred_masks[similiarity_filter_s[:, 0]], recon_logits[similiarity_filter_s[:, 0]]], dim=0),
            #                win_name="small similiarity:{}".format(len(similiarity_filter_s)),
            #                nrow=len(similiarity_filter_s[:, 0]))

            # Apply per-class NMS
            # print("sorted simi:{}".format(sorted(np.array(similiarity.cpu()))))
            # print("Scores changed")
            scores = scores * torch.relu(torch.log(torch.FloatTensor([alpha]).to(similiarity.device) - similiarity) /
                                         torch.log(torch.FloatTensor([alpha]).to(similiarity.device)))
            n += 1

    scores = scores ** (1 / n)
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    results = Instances(image_shape)
    results.pred_boxes = Boxes(boxes)
    results.scores = scores
    results.pred_classes = filter_inds[:, 1]

    return results, filter_inds[:, 0]


class FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """

    def __init__(
        self, box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta

        box_type = type(proposals[0].proposal_boxes)
        # cat(..., dim=0) concatenates over all images in the batch
        self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
        assert not self.proposals.tensor.requires_grad, "Proposals should not require gradients!"
        self.image_shapes = [x.image_size for x in proposals]

        # The following fields should exist only when training.
        if proposals[0].has("gt_boxes"):
            self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
            assert proposals[0].has("gt_classes")
            self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
        if num_fg > 0:
            storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
            storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        self._log_accuracy()
        return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")

    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """

        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        # import ipdb
        # ipdb.set_trace()
        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        return {
            "loss_cls": self.softmax_cross_entropy_loss(),
            "loss_box_reg": self.smooth_l1_loss(),
        }

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        K = self.pred_proposal_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas.view(num_pred * K, B),
            self.proposals.tensor.unsqueeze(1).expand(num_pred, K, B).reshape(-1, B),
        )
        return boxes.view(num_pred, K * B).split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image, mask_pooler=None, mask_head=None,
                  features=None, proposals=None, targets=None, recon_net=None, recon_alpha=None, feature_dict=None,
                  recls=None):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
            mask_pooler (func): used in mask as scores
            mask_head (list): used in mask as scores
            features(list): used in mask as scores
            proposals:used in mask as scores
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes
        return fast_rcnn_inference(
            boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image, mask_pooler=mask_pooler,
            mask_head=mask_head, features=features, recon_net=recon_net, recon_alpha=recon_alpha, recls=recls,
        )


class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNNOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas


def get_pred_masks_logits_by_cls(pred_mask_logits, instances):
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)

    if isinstance(instances, list):
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

        gt_masks = gt_masks.float()

        return pred_mask_logits.unsqueeze(1), gt_masks.unsqueeze(1)
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = instances

        pred_mask_logits = pred_mask_logits[indices, gt_classes]

        return pred_mask_logits.unsqueeze(1)
