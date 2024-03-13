from typing import List

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict

from datapreparation.kitti360pose.imports import Object3d, Pose, Cell

from models.superglue_matcher import get_pos_in_cell, get_pos_in_cell_intersect
try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False
try:
    import horovod.torch as hvd
except ImportError:
    hvd = None
import torch.nn as nn
from torch.nn import functional as F


class MatchingLoss(nn.Module):
    def __init__(self):
        """Matching loss for SuperGlue-based matching training"""
        super(MatchingLoss, self).__init__()
        self.eps = 1e-3

    # Matches as list[tensor ∈ [M1, 2], tensor ∈ [M2, 2], ...] for Mi: i ∈ [1, batch_size]
    def forward(self, P, all_matches):
        assert len(P.shape) == 3
        assert len(all_matches[0].shape) == 2 and all_matches[0].shape[-1] == 2
        assert len(P) == len(all_matches)
        batch_losses = []
        for i in range(len(all_matches)):
            matches = all_matches[i]
            cell_losses = -torch.log(P[i, matches[:, 0], matches[:, 1]])
            batch_losses.append(torch.mean(cell_losses))

        return torch.mean(torch.stack(batch_losses))


def calc_recall_precision(batch_gt_matches, batch_matches0, batch_matches1):
    assert len(batch_gt_matches) == len(batch_matches0) == len(batch_matches1)
    all_recalls = []
    all_precisions = []

    for idx in range(len(batch_gt_matches)):
        gt_matches, matches0, matches1 = (
            batch_gt_matches[idx],
            batch_matches0[idx],
            batch_matches1[idx],
        )
        gt_matches = gt_matches.tolist()

        recall = []
        for i, j in gt_matches:
            recall.append(matches0[i] == j or matches1[j] == i)

        precision = []
        for i, j in enumerate(matches0):
            if j >= 0:
                precision.append(
                    [i, j] in gt_matches
                )  # CARE: this only works as expected after tolist()

        recall = np.mean(recall) if len(recall) > 0 else 0.0
        precision = np.mean(precision) if len(precision) > 0 else 0.0
        all_recalls.append(recall)
        all_precisions.append(precision)

    return np.mean(all_recalls), np.mean(all_precisions)


def calc_pose_error_intersect(objects, matches0, poses: List[Pose], directions):
    assert len(objects) == len(matches0) == len(poses)
    # assert isinstance(poses[0], Pose)

    batch_size, pad_size = matches0.shape
    poses = np.array([pose.pose for pose in poses])[:, 0:2]  # Assuming this is the best cell!

    errors = []
    for i_sample in range(batch_size):
        pose_prediction = get_pos_in_cell_intersect(
            objects[i_sample], matches0[i_sample], directions[i_sample]
        )
        errors.append(np.linalg.norm(poses[i_sample] - pose_prediction))
    return np.mean(errors)


def calc_pose_error(
    objects, matches0, poses: List[Pose], offsets=None, use_mid_pred=False, return_samples=False
):
    """Calculates the mean error of a batch by averaging the positions of all matches objects plus corresp. offsets.
    All calculations are in x-y-plane.

    Args:
        objects (List[List[Object3D]]): Objects-list for each sample in the batch.
        matches0 (np.ndarray): SuperGlue matching output of the batch.
        poses (np.ndarray): Ground-truth poses [batch_size, 3]
        offsets (List[np.ndarray], optional): List of offset vectors for all hints. Zero offsets are used if not given.
        use_mid_pred (bool, optional): If set, predicts the center of the cell regardless of matches and offsets. Defaults to False.

    Returns:
        [float]: Mean error.
    """
    assert len(objects) == len(matches0) == len(poses)
    # assert isinstance(poses[0], Pose)

    batch_size, pad_size = matches0.shape
    poses = np.array([pose.pose for pose in poses])[:, 0:2]  # Assuming this is the best cell!

    if offsets is not None:
        assert len(objects) == len(offsets)
    else:
        offsets = np.zeros(
            (batch_size, pad_size, 2)
        )  # Set zero offsets to just predict the mean of matched-objects' centers

    errors = []
    for i_sample in range(batch_size):
        if use_mid_pred:
            pose_prediction = np.array((0.5, 0.5))
        else:
            pose_prediction = get_pos_in_cell(
                objects[i_sample], matches0[i_sample], offsets[i_sample]
            )
        errors.append(np.linalg.norm(poses[i_sample] - pose_prediction))

    if return_samples:
        return errors
    else:
        return np.mean(errors)


def calc_pose_error_no_superglue(
    gt_coords, pred_coords, use_mid_pred=False, return_samples=False
):
    """Calculates the mean error of a batch by averaging the positions of all matches objects plus corresp. offsets.
    All calculations are in x-y-plane.

    Args:
        objects (List[List[Object3D]]): Objects-list for each sample in the batch.
        matches0 (np.ndarray): SuperGlue matching output of the batch.
        poses (np.ndarray): Ground-truth poses [batch_size, 3]
        offsets (List[np.ndarray], optional): List of offset vectors for all hints. Zero offsets are used if not given.
        use_mid_pred (bool, optional): If set, predicts the center of the cell regardless of matches and offsets. Defaults to False.

    Returns:
        [float]: Mean error.
    """
    assert len(gt_coords) == len(pred_coords)
    # assert isinstance(poses[0], Pose)

    batch_size, pad_size = gt_coords.shape

    errors = []
    for i_sample in range(batch_size):
        if use_mid_pred:
            pose_prediction = np.array((0.5, 0.5))
        else:
            pose_prediction = pred_coords[i_sample]

        errors.append(np.linalg.norm(gt_coords[i_sample] - pose_prediction))

    if return_samples:
        return errors
    else:
        return np.mean(errors)


class PairwiseRankingLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        """Pairwise Ranking loss for retrieval training.
        Implementation taken from a public GitHub, original paper:
        "Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models"
        (Kiros, Salakhutdinov, Zemel. 2014)

        Args:
            margin (float, optional): _description_. Defaults to 1.0.
        """
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, im, s):  # Norming the input (as in paper) is actually not helpful
        im = im / torch.norm(im, dim=1, keepdim=True)
        s = s / torch.norm(s, dim=1, keepdim=True)

        margin = self.margin
        # compute image-sentence score matrix
        scores = torch.mm(im, s.transpose(1, 0))
        # print(scores)
        diagonal = scores.diag()

        # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
        cost_s = torch.max(
            torch.autograd.Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()),
            (margin - diagonal).expand_as(scores) + scores,
        )
        # compare every diagonal score to scores in its row (i.e, all contrastive sentences for each image)
        cost_im = torch.max(
            torch.autograd.Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()),
            (margin - diagonal).expand_as(scores).transpose(1, 0) + scores,
        )

        for i in range(scores.size()[0]):
            cost_s[i, i] = 0
            cost_im[i, i] = 0

        return (cost_s.sum() + cost_im.sum()) / len(im)  # Take mean for batch-size stability


class HardestRankingLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(HardestRankingLoss, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, images, captions):
        assert images.shape == captions.shape and len(images.shape) == 2
        images = images / torch.norm(images, dim=1, keepdim=True)
        captions = captions / torch.norm(captions, dim=1, keepdim=True)
        num_samples = len(images)

        similarity_scores = torch.mm(images, captions.transpose(1, 0))  # [I x C]

        cost_images = (
            self.margin + similarity_scores - similarity_scores.diag().view((num_samples, 1))
        )
        cost_images.fill_diagonal_(0)
        cost_images = self.relu(cost_images)
        cost_images, _ = torch.max(cost_images, dim=1)
        cost_images = torch.mean(cost_images)

        cost_captions = (
            self.margin
            + similarity_scores.transpose(1, 0)
            - similarity_scores.diag().view((num_samples, 1))
        )
        cost_captions.fill_diagonal_(0)
        cost_captions = self.relu(cost_captions)
        cost_captions, _ = torch.max(cost_captions, dim=1)
        cost_captions = torch.mean(cost_captions)

        cost = cost_images + cost_captions
        return cost


# def gather_features(
#         image_features,
#         text_features,
#         local_loss=False,
#         gather_with_grad=False,
#         rank=0,
#         world_size=1,
#         use_horovod=False
# ):
#     assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
#     if use_horovod:
#         assert hvd is not None, 'Please install horovod'
#         if gather_with_grad:
#             all_image_features = hvd.allgather(image_features)
#             all_text_features = hvd.allgather(text_features)
#         else:
#             with torch.no_grad():
#                 all_image_features = hvd.allgather(image_features)
#                 all_text_features = hvd.allgather(text_features)
#             if not local_loss:
#                 # ensure grads for local rank when all_* features don't have a gradient
#                 gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
#                 gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
#                 gathered_image_features[rank] = image_features
#                 gathered_text_features[rank] = text_features
#                 all_image_features = torch.cat(gathered_image_features, dim=0)
#                 all_text_features = torch.cat(gathered_text_features, dim=0)
#     else:
#         # We gather tensors from all gpus
#         if gather_with_grad:
#             all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
#             all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
#         else:
#             gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
#             gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
#             dist.all_gather(gathered_image_features, image_features)
#             dist.all_gather(gathered_text_features, text_features)
#             if not local_loss:
#                 # ensure grads for local rank when all_* features don't have a gradient
#                 gathered_image_features[rank] = image_features
#                 gathered_text_features[rank] = text_features
#             all_image_features = torch.cat(gathered_image_features, dim=0)
#             all_text_features = torch.cat(gathered_text_features, dim=0)
#
#     return all_image_features, all_text_features


# class ClipLoss(nn.Module):
#
#     def __init__(
#             self,
#             local_loss=False,
#             gather_with_grad=False,
#             cache_labels=False,
#             rank=0,
#             world_size=1,
#             use_horovod=False,
#     ):
#         super().__init__()
#         self.local_loss = local_loss
#         self.gather_with_grad = gather_with_grad
#         self.cache_labels = cache_labels
#         self.rank = rank
#         self.world_size = world_size
#         self.use_horovod = use_horovod
#
#         # cache state
#         self.prev_num_logits = 0
#         self.labels = {}
#
#     def get_ground_truth(self, device, num_logits) -> torch.Tensor:
#         # calculated ground-truth and cache if enabled
#         if self.prev_num_logits != num_logits or device not in self.labels:
#             labels = torch.arange(num_logits, device=device, dtype=torch.long)
#             if self.world_size > 1 and self.local_loss:
#                 labels = labels + num_logits * self.rank
#             if self.cache_labels:
#                 self.labels[device] = labels
#                 self.prev_num_logits = num_logits
#         else:
#             labels = self.labels[device]
#         return labels
#
#     def get_logits(self, image_features, text_features, logit_scale=1/0.07):
#         if self.world_size > 1:
#             all_image_features, all_text_features = gather_features(
#                 image_features, text_features,
#                 self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
#
#             if self.local_loss:
#                 logits_per_image = logit_scale * image_features @ all_text_features.T
#                 logits_per_text = logit_scale * text_features @ all_image_features.T
#             else:
#                 logits_per_image = logit_scale * all_image_features @ all_text_features.T
#                 logits_per_text = logits_per_image.T
#         else:
#             logits_per_image = logit_scale * image_features @ text_features.T
#             logits_per_text = logit_scale * text_features @ image_features.T
#
#         return logits_per_image, logits_per_text
#
#     def forward(self, image_features, text_features, logit_scale=1/0.07, output_dict=False):
#         device = image_features.device
#         logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
#
#         labels = self.get_ground_truth(device, logits_per_image.shape[0])
#
#         total_loss = (
#                              F.cross_entropy(logits_per_image, labels) +
#                              F.cross_entropy(logits_per_text, labels)
#                      ) / 2
#         # print(f"contrastive_loss: {total_loss}")
#         return {"contrastive_loss": total_loss} if output_dict else total_loss


# def test_clip_loss_integration(world_size=1, batch_size=4, feature_size=256):
#     # Initialize a fake distributed environment
#     if not dist.is_initialized():
#         dist.init_process_group(backend='gloo', init_method='file:///tmp/somefile', world_size=world_size, rank=0)
#
#     # Generate dummy image and text features along with a logit scale
#     image_features = torch.randn(batch_size, feature_size)
#     text_features = torch.randn(batch_size, feature_size)
#     logit_scale = torch.tensor(1.0)
#
#     # Initialize ClipLoss
#     clip_loss = ClipLoss(world_size=world_size)
#
#     # Compute loss
#     loss = clip_loss(image_features, text_features, logit_scale)
#
#     print(f"Computed loss: {loss.item()}")


class ClipLoss(nn.Module):
    def __init__(self, temperature_init=1/0.07):
        super(ClipLoss, self).__init__()
        self.temperature = nn.Parameter(torch.tensor([temperature_init]))

    def forward(self, image_features, text_features):
        # 计算特征的余弦相似度
        logits_per_image = torch.matmul(image_features, text_features.t()) * self.temperature.to(image_features.device)
        logits_per_text = logits_per_image.t()

        # 计算损失
        loss = (self.cross_entropy_loss(logits_per_image) + self.cross_entropy_loss(logits_per_text)) / 2
        return loss

    def cross_entropy_loss(self, logits):
        labels = torch.arange(len(logits), device=logits.device)
        return nn.functional.cross_entropy(logits, labels)

class SemanticLoss(nn.Module):
    def __init__(self, known_classes):
        super(SemanticLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.known_classes = known_classes
        # print(f"SemanticLoss: {self.known_classes}")
    def forward(self, predictions, objects):
        class_indices = []  # Batch tensor to send into PyG
        for i_batch, objects_sample in enumerate(objects):
            for obj in objects_sample:
                class_idx = self.known_classes.get(obj.label)
                class_indices.append(class_idx)
        class_indices = torch.tensor(class_indices, dtype=torch.long, device=predictions.device)  # [N]
        # print(f"predictions: {predictions.shape}, class_indices: {class_indices.shape}")
        # print(class_indices.max())
        return self.loss_fn(predictions, class_indices)


# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_mask: float = 1,
        cost_dice: float = 1,
        num_points: int = 0,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert (
            cost_class != 0 or cost_mask != 0 or cost_dice != 0
        ), "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets, mask_type):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]
        # print("output_pred_logits: ", outputs["pred_logits"])
        # print("bs: ", bs)
        indices = []
        # print(len(targets))
        # print("targets: ", targets)
        # Iterate through batch size
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(
                -1
            )  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"].clone()

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            filter_ignore = tgt_ids == 253
            tgt_ids[filter_ignore] = 0
            cost_class = -out_prob[:, tgt_ids]
            cost_class[
                :, filter_ignore
            ] = (
                -1.0
            )  # for ignore classes pretend perfect match ;) TODO better worst class match?

            out_mask = outputs["pred_masks"][
                b
            ].T  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b][mask_type].to(out_mask)

            if self.num_points != -1:
                point_idx = torch.randperm(
                    tgt_mask.shape[1], device=tgt_mask.device
                )[: int(self.num_points * tgt_mask.shape[1])]
                # point_idx = torch.randint(0, tgt_mask.shape[1], size=(self.num_points,), device=tgt_mask.device)
            else:
                # sample all points
                point_idx = torch.arange(
                    tgt_mask.shape[1], device=tgt_mask.device
                )

            # out_mask = out_mask[:, None]
            # tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            # point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            # tgt_mask = point_sample(
            #     tgt_mask,
            #     point_coords.repeat(tgt_mask.shape[0], 1, 1),
            #     align_corners=False,
            # ).squeeze(1)

            # out_mask = point_sample(
            #     out_mask,
            #     point_coords.repeat(out_mask.shape[0], 1, 1),
            #     align_corners=False,
            # ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(
                    out_mask[:, point_idx], tgt_mask[:, point_idx]
                )

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(
                    out_mask[:, point_idx], tgt_mask[:, point_idx]
                )

            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets, mask_type):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets, mask_type)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


if __name__ == "__main__":
    # objects = [
    #     [
    #         EasyDict(center=np.array([0, 0])),
    #         EasyDict(center=np.array([10, 10])),
    #         EasyDict(center=np.array([99, 99])),
    #     ],
    # ]
    # matches0 = np.array((0, 1, -1)).reshape((1, 3))
    # poses = np.array((0, 10)).reshape((1, 2))
    # offsets = np.array([(2, 10), (-10, 0)]).reshape((1, 2, 2))
    #
    # err = calc_pose_error(objects, matches0, poses, offsets=None)
    # print(err)
    # err = calc_pose_error(objects, matches0, poses, offsets=offsets)
    # print(err)
    # test_clip_loss_integration()
    pass
