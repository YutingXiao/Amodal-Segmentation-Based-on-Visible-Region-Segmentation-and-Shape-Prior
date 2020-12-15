import torch
import torch.nn as nn
from torch.nn import functional as F
import fvcore.nn.weight_init as weight_init

from detectron2.utils.events import get_event_storage
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.utils.registry import Registry
from .mask_head import ResBlock

ROI_DIS_HEAD_REGISTRY = Registry("ROI_DIS_HEAD")
ROI_DIS_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def amodal_dis_mask_rcnn_loss(mask_dis_logits, lamd_adv=1):
    """
    Compute the discriminator loss defined in GAN.

    Args:
        mask_dis_logits (Tensor): A tensor of shape (B, N), the output of discriminator.
        lamd_adv: parameters
    Returns:
        mask_dis_loss (Tensor): A scalar tensor containing the loss.
    """
    dis_fake_logits = mask_dis_logits[0]
    dis_real_logits = mask_dis_logits[1]
    gen_fake_logits = mask_dis_logits[2]

    adversarial_loss = AdversarialLoss().cuda()
    # discriminator loss
    dis_real_loss = adversarial_loss(dis_real_logits, True)
    dis_fake_loss = adversarial_loss(dis_fake_logits, False)
    loss_dis = lamd_adv * (dis_real_loss + dis_fake_loss) / 2

    # generator adversarial loss
    loss_gen = adversarial_loss(gen_fake_logits, True, False) * lamd_adv

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn_dis/loss_dis", loss_dis)
    storage.put_scalar("mask_rcnn_dis/loss_gen", loss_gen)
    return loss_gen, loss_dis


@ROI_DIS_HEAD_REGISTRY.register()
class GeneralDis(nn.Module):
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
        super(GeneralDis, self).__init__()

        # fmt: off
        assert input_shape.height == input_shape.width
        num_resblock = cfg.MODEL.ROI_DIS_HEAD.NUM_RESBLOCK
        conv_dims = cfg.MODEL.ROI_DIS_HEAD.CONV_DIM
        self.norm = cfg.MODEL.ROI_DIS_HEAD.NORM
        # fmt: on

        self.net = []

        for k in range(num_resblock):
            conv = Conv2d(
                1 if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims, num_groups=2),
                activation=F.relu,
            )
            self.add_module("dis_conv{}".format(k + 1), conv)
            self.net.append(conv)
            res_block = ResBlock(conv_dims, norm=self.norm, num_group=2)
            self.add_module("dis_res_block{}".format(k + 1), res_block)
            self.net.append(res_block)

        for layer in self.net:
            if isinstance(layer, Conv2d):
                weight_init.c2_msra_fill(layer)

        self.aver_pooling = nn.AvgPool2d(kernel_size=4)
        self.cls = nn.Linear(conv_dims, 1)

    def forward(self, x):
        for layer in self.net:
            x = layer(x)

        x = self.aver_pooling(x).squeeze(2).squeeze(2)
        dis_logits = torch.sigmoid(self.cls(x))

        return dis_logits


class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


def build_dis_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_DIS_HEAD.NAME
    return ROI_DIS_HEAD_REGISTRY.get(name)(cfg, input_shape)
