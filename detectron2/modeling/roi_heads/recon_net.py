import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.cluster import KMeans
import torchvision.transforms as transforms
import fvcore.nn.weight_init as weight_init

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.structures import Boxes, Instances, pairwise_iou
from .memo_functions import vq, vq_st

from ..postprocessing import detector_postprocess

ROI_RECON_HEAD_REGISTRY = Registry("ROI_RECON_HEAD")


def mask_recon_loss(pred_mask_logits, instances, recon_net, targets, box_ths=0.8, mask_ths=0.9, iter=0):
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
        recon_net (nn.Modlule): reconstruction network
        mask_ths (float): IOU threshold.
        iter (int): iterations

    Returns:
        mask_recon_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    # total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"
    targets_classes_lst = []
    pred_classes_lst = []
    index = [0]

    input_gt = []
    input_pred = []
    for instances_per_image in instances:
        arange = torch.arange(len(instances_per_image))
        iou_box = pairwise_iou(instances_per_image.proposal_boxes, instances_per_image.gt_boxes)[arange, arange]

        gt_amodal_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)

        pred_masks_per_image = pred_mask_logits[index[-1]: index[-1] + len(instances_per_image)] > 0
        pred_masks_per_image = pred_masks_per_image[arange, instances_per_image.gt_classes]

        iou_mask = torch.sum((gt_amodal_per_image * pred_masks_per_image) > 0, dim=(1, 2)).float() / \
                   torch.sum((gt_amodal_per_image + pred_masks_per_image) > 0, dim=(1, 2)).float()
        filter_inds = (iou_box > box_ths) * (iou_mask > mask_ths).nonzero()

        gt_classes = instances_per_image.gt_classes
        if cls_agnostic_mask:
            input_pred.append((pred_mask_logits[index[-1]: index[-1] + len(instances_per_image)][filter_inds[:, 0]][:, 0] > 0).detach())
        else:
            indices = torch.arange(gt_classes.size(0))
            pred_mask_logits_per_instance = pred_mask_logits[index[-1]: index[-1] + len(instances_per_image)]
            pred_mask_logits_per_instance = pred_mask_logits_per_instance[indices, gt_classes]
            input_pred.append((pred_mask_logits_per_instance[filter_inds[:, 0]] > 0).detach())
        pred_classes_lst.append(gt_classes[filter_inds[:, 0]])

        index.append(len(instances_per_image))

    for target_per_image in targets:
        try:
            input_gt.append(target_per_image.gt_masks.crop_and_resize(
                target_per_image.gt_boxes.tensor, mask_side_len).to(device=pred_mask_logits.device))
        except:
            import ipdb
            ipdb.set_trace()

        gt_classes = target_per_image.gt_classes
        targets_classes_lst.append(gt_classes)

    inputs = cat(input_gt + input_pred, dim=0).unsqueeze(1).float().detach()

    recon_masks_logits, _ = recon_net(inputs)
    mask_recon_loss = F.mse_loss(recon_masks_logits, inputs.to(dtype=torch.float32))
    # if iter % 2000 == 1:
    #     try:
    #         vis.images(cat(input_gt).unsqueeze(1), win_name="gt_mask_input_{}_{}".format(recon_net.name, iter))
    #         vis.images((recon_masks_logits[:cat(input_gt).size(0)]).float(),
    #                    win_name="gt_mask_output_{}_{}".format(recon_net.name, iter))
    #     except:
    #         pass
    #     # vis.images(cat(input_gt).unsqueeze(1), win_name="gt_mask_input_to_recon_{}".format(iter))
    #     # vis.images((recon_masks_logits[:cat(input_gt).size(0)]).float(), win_name="gt_mask_output_from_recon_{}".format(iter))
    #     if cat(input_gt).size(0) != recon_masks_logits.size(0):
    #         try:
    #             vis.images(cat(input_pred).unsqueeze(1), win_name="pred_mask_input_{}_{}".format(recon_net.name, iter))
    #             vis.images((recon_masks_logits[cat(input_gt).size(0):]).float(),
    #                        win_name="pred_mask_output_{}_{}".format(recon_net.name, iter))
    #         except:
    #             pass

    return mask_recon_loss


def mask_recon_inference(pred_instances, targets, recon_net, iou_ths=0.9):
    assert len(pred_instances) == 1
    vector_dict = {}

    pred_instance = pred_instances[0]
    target = targets[0]
    masks = []
    classes = []
    mask_side_len = pred_instances[0].pred_amodal_masks.size(2)

    res = detector_postprocess(pred_instance, pred_instance.image_size[0], pred_instance.image_size[1])
    gt_masks_orig_size = pred_instance.gt_masks_inference.tensor
    pred_masks_orig_size = res.pred_amodal_masks.cuda()

    iou = torch.sum((gt_masks_orig_size * pred_masks_orig_size) > 0, dim=(1, 2)).float() /\
          torch.sum((gt_masks_orig_size + pred_masks_orig_size) > 0, dim=(1, 2)).float()
    # print(iou)
    filter_inds = (iou > iou_ths).nonzero()
    if pred_instance.has("pred_amodal_masks"):
        masks.append(pred_instance.pred_amodal2_masks[filter_inds[:, 0]])  # (1, Hmask, Wmask)
    else:
        masks.append(pred_instance.pred_amodal_masks[filter_inds[:, 0]])
    classes.append(pred_instance.gt_classes_inference[filter_inds[:, 0]])

    if filter_inds.size(0) != 0:
        masks.append(pred_instance.gt_masks_inference[filter_inds[:, 0]].crop_and_resize(
            pred_instance.pred_boxes.tensor[filter_inds[:, 0]], mask_side_len
        ).to(device=pred_masks_orig_size.device).unsqueeze(1).float())
        classes.append(pred_instance.gt_classes_inference[filter_inds[:, 0]])

    masks.append(target.gt_masks.crop_and_resize(
        target.gt_boxes.tensor, mask_side_len
    ).to(device=pred_masks_orig_size.device).unsqueeze(1).float())
    classes.append(target.gt_classes)

    masks = cat(masks, dim=0)
    classes = cat(classes, dim=0)

    # else:
    #     mask_side_len = pred_instances[0].pred_masks.size(2)
    #
    #     res = detector_postprocess(pred_instance, pred_instance.image_size[0], pred_instance.image_size[1])
    #     gt_masks_orig_size = pred_instance.gt_masks_inference.tensor
    #     pred_masks_orig_size = res.pred_masks.cuda()
    #
    #     iou = torch.sum((gt_masks_orig_size * pred_masks_orig_size) > 0, dim=(1, 2)).float() /\
    #           torch.sum((gt_masks_orig_size + pred_masks_orig_size) > 0, dim=(1, 2)).float()
    #     filter_inds = (iou > iou_ths).nonzero()
    #
    #     pred_amodal_masks_to_recon = pred_instance.pred_masks[filter_inds[:, 0]]  # (1, Hmask, Wmask)
    #     classes_to_pred_recon = pred_instance.gt_classes_inference[filter_inds[:, 0]]
    #
    #     gt_masks_to_recon = target.gt_masks.crop_and_resize(
    #         target.gt_boxes.tensor, mask_side_len
    #     ).to(device=pred_masks_orig_size.device)
    #     classes_to_gt_recon = target.gt_classes
    #
    #     masks = cat([pred_amodal_masks_to_recon, gt_masks_to_recon.unsqueeze(1).float()], dim=0)
    #     classes = cat([classes_to_pred_recon, classes_to_gt_recon], dim=0)

    if recon_net.memory_aug:
        masks_aug = masks
        classes_aug = classes
        for degree in [0, 90, 180, 270]:
            for i in range(2):
                if degree == 0 and i != 0:
                    continue
                angle = - degree * math.pi / 180
                theta = torch.tensor([
                    [math.cos(angle), math.sin(-angle), 0],
                    [math.sin(angle), math.cos(angle), 0]
                ], dtype=torch.float)
                theta = cat([theta.unsqueeze(0)] * masks.size(0), dim=0)
                grid = F.affine_grid(theta, masks.size(), align_corners=True).to(masks.device)
                output = F.grid_sample(masks, grid, align_corners=True)
                if i == 0:
                    output = output.flip(2)

                masks_aug = cat([masks_aug, output], dim=0)
                classes_aug = cat([classes_aug, classes], dim=0)
        masks = masks_aug
        classes = classes_aug

    recon_masks_logits, latent_vectors = recon_net((masks > 0.5).float())
    # recon_masks = (recon_masks_logits > 0.5).float()
    for i in range(len(classes.unique())):
        index = (classes == classes.unique()[i].item()).nonzero()
        vector_dict[classes.unique()[i].item()] = latent_vectors[index].view(len(index), -1)


        # if len(pred_instances) < 10:
        #     if pred_masks_to_recon.size(0) != 0:
        #         vis.images((pred_masks_to_recon > 0.5).float(), win_name="inference_pred_masks_in_recon_{}".format(len(pred_instances)))
        #         vis.images(recon_masks_logits[:pred_masks_to_recon.size(0)], win_name="inference_pred_masks_out_recon_{}".format(len(pred_instances)))
        #     vis.images((gt_masks_to_recon.unsqueeze(1) > 0.5).float(), win_name="inference_gt_masks_in_recon_{}".format(len(pred_instances)))
        #     vis.images(recon_masks_logits[pred_masks_to_recon.size(0):], win_name="inference_gt_masks_out_recon_{}".format(len(pred_instances)))

    recon_net.recording_vectors(vector_dict)


@ROI_RECON_HEAD_REGISTRY.register()
class General_Recon_Net(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec, K=100):
        self.name = "AE"
        self.num_conv = cfg.MODEL.ROI_MASK_HEAD.RECON_NET.NUM_CONV
        self.conv_dims = cfg.MODEL.ROI_MASK_HEAD.RECON_NET.CONV_DIM
        self.norm = cfg.MODEL.ROI_MASK_HEAD.RECON_NET.NORM
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.num_cluster = cfg.MODEL.ROI_MASK_HEAD.RECON_NET.KMEANS
        self.rescoring = cfg.MODEL.ROI_MASK_HEAD.RECON_NET.RESCORING
        self.memory_aug = cfg.MODEL.ROI_MASK_HEAD.RECON_NET.MEMO_AUG
        input_channels = 1
        self.vector_dict = {}
        super(General_Recon_Net, self).__init__()
        self.encoder = []
        self.decoder = []

        for k in range(self.num_conv):
            conv = Conv2d(
                input_channels if k == 0 else self.conv_dims,
                self.conv_dims,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, self.conv_dims),
                activation=F.relu,
            )
            self.add_module("mask_fcn_enc{}".format(k + 1), conv)

            self.encoder.append(conv)

            deconv = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                   Conv2d(self.conv_dims,
                                          self.conv_dims,
                                          kernel_size=3,
                                          stride=1,
                                          padding=0 if k == self.num_conv - 2 and self.num_conv > 2 else 1,
                                          bias=not self.norm,
                                          norm=get_norm(self.norm, self.conv_dims, num_groups=2)
                                          ),
                                   # get_norm(self.norm, self.conv_dims, num_groups=2),
                                   nn.ReLU())
            self.add_module("mask_fcn_dec{}".format(k + 1), deconv)
            self.decoder.append(deconv)
        self.outconv = nn.Sequential(nn.Conv2d(self.conv_dims,
                                               1,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1),
                                     nn.Sigmoid())
        # self.d = {1: 14, 2: 7, 3: 4, 4: 2}
        # self.fc = nn.Linear(self.conv_dims, self.conv_dims * self.d[self.num_conv] * self.d[self.num_conv], bias=False)

        # for layer in self.encoder + self.decoder:
        #     weight_init.c2_msra_fill(layer)
        # d = {3: 4, 4: 2}
        # self.codebook = nn.Embedding(K * num_classes, d[self.num_conv] ** 2)

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        latent_vector = x

        for layer in self.decoder:
            x = layer(x)
        x = self.outconv(x)
        return x, latent_vector

    def encode(self, x):
        for layer in self.encoder:
            x = layer(x)

        return x

    def decode(self, vectors):
        for layer in self.decoder:
            vectors = layer(vectors)
        x = self.outconv(vectors)

        return x

    def recording_vectors(self, vector_inference):
        for key, item in vector_inference.items():
            if self.vector_dict.get(key, None) is not None:
                self.vector_dict[key] = cat([self.vector_dict[key], item], dim=0)
            else:
                self.vector_dict[key] = item

    def nearest_decode(self, vectors, pred_classes, k=1):
        side_len = math.sqrt(vectors.size(1) / self.conv_dims)
        assert side_len % 1 == 0
        memo_latent_vectors = torch.zeros((vectors.size(0), k, vectors.size(1))).to(vectors.device)

        classes_lst = pred_classes.unique()
        for i in range(len(classes_lst)):
            with torch.no_grad():
                index = (pred_classes == pred_classes.unique()[i].item()).nonzero()
                vectors_per_classes = vectors[index[:, 0]]
                if pred_classes.unique()[i].item() in self.vector_dict:
                    codebook = self.vector_dict[pred_classes.unique()[i].item()]

                    codebook_sqr = torch.sum(codebook ** 2, dim=1)
                    inputs_sqr = torch.sum(vectors_per_classes ** 2, dim=1, keepdim=True)

                    # Compute the distances to the codebook
                    distances = torch.addmm(codebook_sqr + inputs_sqr,
                                            vectors_per_classes, codebook.t(), alpha=-2.0, beta=1.0)

                    # _, indices_flatten = torch.min(distances, dim=1)
                    # indices = indices_flatten.view(*inputs_size[:-1])
                    indices = torch.topk(- distances, k)[1]
                    nn_vectors = codebook[indices]

                    # nn_vectors = torch.index_select(codebook, dim=0, index=indices)
                    memo_latent_vectors[index[:, 0]] = nn_vectors
                else:
                    memo_latent_vectors[index[:, 0]] = vectors_per_classes.unsqueeze(1)

        # memo_latent_vectors = (memo_latent_vectors + vectors) / 2
        vectors = memo_latent_vectors.view(pred_classes.size(0) * k, self.conv_dims, int(side_len), int(side_len))

        for layer in self.decoder:
            vectors = layer(vectors)
        x = self.outconv(vectors)
        x = x.view(pred_classes.size(0), k, x.size(2), x.size(3))
        return x

    def cluster(self):
        for i in range(self.num_classes):
            print("Start cluster No.{} class".format(i + 1))
            if i not in self.vector_dict:
                continue
            if self.vector_dict[i].size(0) > self.num_cluster:
                codes = self.vector_dict[i]
                kmeans = KMeans(n_clusters=self.num_cluster)
                kmeans.fit(codes.cpu())
                self.vector_dict[i] = torch.FloatTensor(kmeans.cluster_centers_).cuda()


@ROI_RECON_HEAD_REGISTRY.register()
class C2VQ_Recon_Net(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """
    def __init__(self, cfg, input_shape: ShapeSpec, K=10000):
        super(C2VQ_Recon_Net, self).__init__()
        self.name = "VQVAE"
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.num_conv = cfg.MODEL.ROI_MASK_HEAD.RECON_NET.NUM_CONV
        self.conv_dims = cfg.MODEL.ROI_MASK_HEAD.RECON_NET.CONV_DIM
        self.norm = cfg.MODEL.ROI_MASK_HEAD.RECON_NET.NORM
        input_channels = 1

        self.encoder = []
        self.decoder = []
        for k in range(self.num_conv):
            conv = Conv2d(input_channels if k == 0 else self.conv_dims,
                          self.conv_dims, kernel_size=3,
                          stride=2,
                          padding=1,
                          bias=not self.norm,
                          norm=get_norm(self.norm, self.conv_dims),
                          activation=F.relu if k != self.num_conv - 1 else None,
                          )
            self.add_module("mask_fcn_enc{}".format(k + 1), conv)

            self.encoder.append(conv)

            deconv = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                   Conv2d(self.conv_dims,
                                          self.conv_dims,
                                          kernel_size=3,
                                          stride=1,
                                          padding=0 if k == self.num_conv - 2 else 1,
                                          bias=not self.norm,
                                          norm=get_norm(self.norm, self.conv_dims)),
                                   nn.ReLU(inplace=True))
            self.add_module("mask_fcn_dec{}".format(k + 1), deconv)
            self.decoder.append(deconv)
        self.outconv = nn.Sequential(nn.Conv2d(self.conv_dims,
                                               1,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1),
                                     nn.Sigmoid())
        self.d = {1: 14, 2: 7, 3: 4}
        self.codebook = C2VQEmbedding(K, self.conv_dims * self.d[self.num_conv] ** 2, num_classes=num_classes)

    def encode(self, x, c):
        for layer in self.conv_norm_relus:
            x = layer(x)
        z_e_x = x
        latents = self.codebook(z_e_x, c)
        return latents

    def decode(self, latents, c):
        z_q_x = self.codebook.embedding(latents, c).permute(0, 3, 1, 2)  # (B, D, H, W)
        x = z_q_x
        for layer in self.decoder:
            x = layer(x)
        x_tilde = self.outconv(x)
        return x_tilde

    def forward(self, x, c):
        for layer in self.encoder:
            x = layer(x)

        z_e_x = x.view(x.size(0), -1).unsqueeze(2).unsqueeze(2)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x, c)

        x = z_q_x_st.view(x.size(0), self.conv_dims, self.d[self.num_conv], self.d[self.num_conv])
        for layer in self.decoder:
            x = layer(x)
        x_tilde = self.outconv(x)
        return x_tilde, z_e_x, z_q_x


class C2VQEmbedding(nn.Module):
    def __init__(self, K, D, num_classes=60):
        super().__init__()
        self.K = K
        self.embedding = nn.Embedding(K * num_classes, D)
        # self.embedding.weight.data.

    def forward(self, z_e_x, c):
        class_dict = {}
        c_lst = [c.unique()[i].item() for i in range(c.unique().size(0))]
        for i in c_lst:
            class_dict[i] = (c == i).nonzero()[:, 0]

        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = torch.zeros_like(z_e_x_)
        for key, index in class_dict.items():
            latents[index] = vq(z_e_x_[index], self.embedding.weight[key * self.K: (key + 1) * self.K])
        return latents

    def straight_through(self, z_e_x, c):
        class_dict = {}
        c_lst = [c.unique()[i].item() for i in range(c.unique().size(0))]
        for i in c_lst:
            class_dict[i] = (c == i).nonzero()[:, 0]

        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        # z_q_x_ = torch.zeros_like(z_e_x_).cuda()
        # indices = torch.zeros(z_e_x_.size(0) * z_e_x_.size(1) * z_e_x_.size(2)).long().cuda()
        z_q_x_ = []
        indices = []
        for key, index in class_dict.items():
            z_q_x_s, indices_s = vq_st(z_e_x_[index], self.embedding.weight[key * self.K: (key + 1) * self.K].detach())
            z_q_x_.append(z_q_x_s)
            indices.append(indices_s + key * self.K)
        z_q_x_ = torch.cat(z_q_x_, dim=0)
        indices = torch.cat(indices, dim=0)
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight, dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar


class VQ(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, ema=False, ema_decay=0.99, ema_eps=1e-5):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.ema = ema
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps

        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        nn.init.kaiming_uniform_(self.embedding.weight, 1)

        if ema:
            self.embedding.weight.requires_grad_(False)
            # set up moving averages
            self.register_buffer('ema_cluster_size', torch.zeros(n_embeddings))
            self.register_buffer('ema_weight', self.embedding.weight.clone().detach())

    def embed(self, encoding_indices):
        return self.embedding(encoding_indices).permute(0,4,1,2,3).squeeze(2)  # in (B,1,H,W); out (B,E,H,W)

    def forward(self, z):
        # input (B,E,H,W); permute and reshape to (B*H*W,E) to compute distances in E-space
        flat_z = z.permute(0,2,3,1).reshape(-1, self.embedding_dim)   # (B*H*W,E)
        # compute distances to nearest embedding
        distances = flat_z.pow(2).sum(1, True) + self.embedding.weight.pow(2).sum(1) - 2 * flat_z.matmul(self.embedding.weight.t())
        # quantize z to nearest embedding
        encoding_indices = distances.argmin(1).reshape(z.shape[0], 1, *z.shape[2:])   # (B,1,H,W)
        z_q = self.embed(encoding_indices)

        # perform ema updates
        if self.ema and self.training:
            with torch.no_grad():
                # update cluster size
                encodings = F.one_hot(encoding_indices.flatten(), self.n_embeddings).float().to(z.device)
                self.ema_cluster_size -= (1 - self.ema_decay) * (self.ema_cluster_size - encodings.sum(0))
                # update weight
                dw = z.permute(1,0,2,3).flatten(1) @ encodings  # (E,B*H*W) dot (B*H*W,n_embeddings)
                self.ema_weight -= (1 - self.ema_decay) * (self.ema_weight - dw.t())
                # update embedding weight with normalized ema_weight
                n = self.ema_cluster_size.sum()
                updated_cluster_size = (self.ema_cluster_size + self.ema_eps) / (n + self.n_embeddings * self.ema_eps) * n
                self.embedding.weight.data = self.ema_weight / updated_cluster_size.unsqueeze(1)

        return encoding_indices, z_q   # out (B,1,H,W) codes and (B,E,H,W) embedded codes


@ROI_RECON_HEAD_REGISTRY.register()
class VQVAE2(nn.Module):
    def __init__(self, input_dims, n_embeddings, embedding_dim, n_channels, n_res_channels, n_res_layers,
                 ema=False, ema_decay=0.99, ema_eps=1e-5, **kwargs):   # keep kwargs so can load from config with arbitrary other args
        super().__init__()
        self.ema = ema

        self.enc1 = nn.Sequential(nn.Conv2d(input_dims[0], n_channels//2, kernel_size=4, stride=2, padding=1),
                                  nn.ReLU(True),
                                  nn.Conv2d(n_channels//2, n_channels, kernel_size=4, stride=2, padding=1),
                                  nn.ReLU(True),
                                  nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
                                  nn.ReLU(True),
                                  nn.Sequential(*[ResidualLayer(n_channels, n_res_channels) for _ in range(n_res_layers)]),
                                  nn.Conv2d(n_channels, embedding_dim, kernel_size=1))

        self.enc2 = nn.Sequential(nn.Conv2d(embedding_dim, n_channels//2, kernel_size=4, stride=2, padding=1),
                                  nn.ReLU(True),
                                  nn.Conv2d(n_channels//2, n_channels, kernel_size=3, padding=1),
                                  nn.ReLU(True),
                                  nn.Sequential(*[ResidualLayer(n_channels, n_res_channels) for _ in range(n_res_layers)]),
                                  nn.Conv2d(n_channels, embedding_dim, kernel_size=1))

        self.dec2 = nn.Sequential(nn.Conv2d(embedding_dim, n_channels, kernel_size=3, padding=1),
                                  nn.ReLU(True),
                                  nn.Sequential(*[ResidualLayer(n_channels, n_res_channels) for _ in range(n_res_layers)]),
                                  nn.ConvTranspose2d(n_channels, embedding_dim, kernel_size=4, stride=2, padding=1))

        self.dec1 = nn.Sequential(nn.Conv2d(2*embedding_dim, n_channels, kernel_size=3, padding=1),
                                  nn.ReLU(True),
                                  nn.Sequential(*[ResidualLayer(n_channels, n_res_channels) for _ in range(n_res_layers)]),
                                  nn.ConvTranspose2d(n_channels, n_channels//2, kernel_size=4, stride=2, padding=1),
                                  nn.ReLU(True),
                                  nn.ConvTranspose2d(n_channels//2, input_dims[0], kernel_size=4, stride=2, padding=1))

        self.proj_to_vq1 = nn.Conv2d(2*embedding_dim, embedding_dim, kernel_size=1)
        self.upsample_to_dec1 = nn.ConvTranspose2d(embedding_dim, embedding_dim, kernel_size=4, stride=2, padding=1)

        self.vq1 = VQ(n_embeddings, embedding_dim, ema, ema_decay, ema_eps)
        self.vq2 = VQ(n_embeddings, embedding_dim, ema, ema_decay, ema_eps)

    def encode(self, x):
        z1 = self.enc1(x)
        z2 = self.enc2(z1)
        return (z1, z2)  # each is (B,E,H,W)

    def embed(self, encoding_indices):
        encoding_indices1, encoding_indices2 = encoding_indices

        return (self.vq1.embed(encoding_indices1), self.vq2.embed(encoding_indices2))

    def quantize(self, z_e):
        # unpack inputs
        z1, z2 = z_e

        # quantize top level
        encoding_indices2, zq2 = self.vq2(z2)

        # quantize bottom level conditioned on top level decoder and bottom level encoder
        #   decode top level
        quantized2 = z2 + (zq2 - z2).detach()  # stop decoder optimization from accessing the embedding
        dec2_out = self.dec2(quantized2)
        #   condition on bottom encoder and top decoder
        vq1_input = torch.cat([z1, dec2_out], 1)
        vq1_input = self.proj_to_vq1(vq1_input)
        encoding_indices1, zq1 = self.vq1(vq1_input)
        return (encoding_indices1, encoding_indices2), (zq1, zq2)

    def decode(self, z_e, z_q):
        # unpack inputs
        zq1, zq2 = z_q
        if z_e is not None:
            z1, z2 = z_e
            # stop decoder optimization from accessing the embedding
            zq1 = z1 + (zq1 - z1).detach()
            zq2 = z2 + (zq2 - z2).detach()

        # upsample quantized2 to match spacial dim of quantized1
        zq2_upsampled = self.upsample_to_dec1(zq2)
        # decode
        combined_latents = torch.cat([zq1, zq2_upsampled], 1)
        return self.dec1(combined_latents)

    def forward(self, x, commitment_cost, writer=None):
        # Figure 2a in paper
        z_e = self.encode(x)
        encoding_indices, z_q = self.quantize(z_e)
        recon_x = self.decode(z_e, z_q)

        # compute loss over the hierarchy -- cf eq 2 in paper
        recon_loss    = F.mse_loss(recon_x, x)
        q_latent_loss = sum(F.mse_loss(z_i.detach(), zq_i) for z_i, zq_i in zip(z_e, z_q)) if not self.ema else torch.zeros(1, device=x.device)
        e_latent_loss = sum(F.mse_loss(z_i, zq_i.detach()) for z_i, zq_i in zip(z_e, z_q))
        loss = recon_loss + q_latent_loss + commitment_cost * e_latent_loss

        return loss


class ResidualLayer(nn.Sequential):
    def __init__(self, n_channels, n_res_channels):
        super().__init__(nn.Conv2d(n_channels, n_res_channels, kernel_size=3, padding=1),
                         nn.ReLU(True),
                         nn.Conv2d(n_res_channels, n_channels, kernel_size=1))

    def forward(self, x):
        return F.relu(x + super().forward(x), True)


@ROI_RECON_HEAD_REGISTRY.register()
class CVAE(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec, K=100):
        self.name = "CVAE"
        self.num_conv = cfg.MODEL.ROI_MASK_HEAD.RECON_NET.NUM_CONV
        self.conv_dims = cfg.MODEL.ROI_MASK_HEAD.RECON_NET.CONV_DIM
        self.norm = cfg.MODEL.ROI_MASK_HEAD.RECON_NET.NORM
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.rescoring = cfg.MODEL.ROI_MASK_HEAD.RECON_NET.RESCORING
        self.lambda_kl = cfg.MODEL.ROI_MASK_HEAD.RECON_NET.LAMBDA_KL
        self.latent_dim = 128
        input_channels = 1
        super(CVAE, self).__init__()
        self.encoder = []
        self.decoder = []
        for k in range(self.num_conv):
            conv = Conv2d(
                input_channels if k == 0 else self.conv_dims,
                self.conv_dims,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, self.conv_dims),
                activation=F.relu,
            )
            self.add_module("mask_fcn_enc{}".format(k + 1), conv)

            self.encoder.append(conv)

            deconv = nn.Sequential(
                ConvTranspose2d(self.conv_dims if self.num_conv > 0 else input_channels,
                                self.conv_dims,
                                kernel_size=2,
                                stride=2,
                                padding=1 if k == self.num_conv - 2 and self.num_conv > 2 else 0),
                                nn.ReLU(inplace=True))
            self.add_module("mask_fcn_dec{}".format(k + 1), deconv)
            self.decoder.append(deconv)
        self.outconv = nn.Sequential(nn.Conv2d(self.conv_dims,
                                               1,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1),
                                     nn.Sigmoid())
        d = {1: 14, 2: 7, 3: 4, 4: 2}
        self.mean = nn.Linear(self.conv_dims * d[self.num_conv] ** 2, self.latent_dim)
        self.log_var = nn.Linear(self.conv_dims * d[self.num_conv] ** 2, self.latent_dim)
        self.fc = nn.Linear(self.conv_dims * d[self.num_conv] ** 2 + self.num_classes, self.latent_dim)

        # self.fc = nn.Linear(self.conv_dims, self.conv_dims * self.d[self.num_conv] * self.d[self.num_conv], bias=False)

        # for layer in self.encoder + self.decoder:
        #     weight_init.c2_msra_fill(layer)
        # d = {3: 4, 4: 2}
        # self.codebook = nn.Embedding(K * num_classes, d[self.num_conv] ** 2)

    def forward(self, x, c):
        for layer in self.encoder:
            x = layer(x)

        mask_side_len = x.size(2)
        means = self.mean(x.view(x.size(0), -1))
        log_vars = self.log_var(x.view(x.size(0), -1))

        std = torch.exp(0.5 * log_vars)
        eps = torch.randn([x.size(0), self.latent_dim]).type_as(std)
        x = eps * std + means

        c_onehot = self._onehot(c, self.num_classes).cuda()
        x = torch.cat([x, c_onehot], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), self.conv_dims, mask_side_len, mask_side_len)
        for layer in self.decoder:
            x = layer(x)
        x = self.outconv(x)
        print(x)
        return x, means, log_vars

    def _onehot(self, c, dim):
        onehot = torch.ones((c.size(0), dim)) * -1
        arange = torch.arange(c.size(0))
        onehot[arange, c] = 1

        return onehot


def build_reconstruction_head(cfg, input_shape):
    """
    Build a reconstruction net for mask head defined by `cfg.MODEL.ROI_MASK_HEAD.RECON_NET.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.RECON_NET.NAME
    return ROI_RECON_HEAD_REGISTRY.get(name)(cfg, input_shape)
