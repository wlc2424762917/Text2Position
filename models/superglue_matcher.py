from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.data import Data, Batch

import time
import numpy as np
import os
import pickle
from easydict import EasyDict

from models.modules import get_mlp, LanguageEncoder
from models.superglue import SuperGlue
from models.object_encoder import ObjectEncoder

# from models.pointcloud.pointnet2 import PointNet2


from datapreparation.kitti360pose.imports import Object3d as Object3d_K360
from models.modules import get_mlp, LanguageEncoder, Clip_LanguageEncoder, MaxPoolMultiHeadSelfAttention, Clip_LanguageEncoder_TransformerFuser, MaxPoolRelationMultiHeadSelfAttention, T5_LanguageEncoder
from models.cross_attention import TransformerCrossEncoderLayer, TransformerCrossEncoder


def get_mlp_offset(dims: List[int], add_batchnorm=False) -> nn.Sequential:
    """Return an MLP without trailing ReLU or BatchNorm for Offset/Translation regression.

    Args:
        dims (List[int]): List of dimension sizes
        add_batchnorm (bool, optional): Whether to add a BatchNorm. Defaults to False.

    Returns:
        nn.Sequential: Result MLP
    """
    if len(dims) < 3:
        print("get_mlp(): less than 2 layers!")
    mlp = []
    for i in range(len(dims) - 1):
        mlp.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mlp.append(nn.ReLU())
            if add_batchnorm:
                mlp.append(nn.BatchNorm1d(dims[i + 1]))
    return nn.Sequential(*mlp)


class SuperGlueMatch(torch.nn.Module):
    def __init__(
        self, known_classes: List[str], known_colors: List[str], known_words: List[str], args
    ):
        """Fine hints-to-objects matching module.
        Consists of object-encoder, language-encoder and SuperGlue-based matching module.

        Args:
            known_classes (List[str]): List of known classes (only used for embedding ablation)
            known_colors (List[str]): List of known colors (only used for embedding ablation)
            known_words (List[str]): List of known words (only used for embedding ablation)
            args: Global training args
        """
        super(SuperGlueMatch, self).__init__()
        self.embed_dim = args.embed_dim
        self.num_layers = args.num_layers
        self.sinkhorn_iters = args.sinkhorn_iters
        self.use_features = args.use_features
        self.args = args

        self.object_encoder = ObjectEncoder(args.embed_dim, known_classes, known_colors, args)

        # object fuse
        if args.only_clip_semantic_feature:
            self.linear_obj = nn.Linear(512+self.embed_dim, self.embed_dim)
        if args.use_clip_semantic_feature:
            self.linear_obj = nn.Linear(512+2*self.embed_dim, self.embed_dim)
        embed_dim = self.embed_dim

        # language encoder
        if args.language_encoder == "CLIP_text":
            self.language_encoder = Clip_LanguageEncoder(clip_version="ViT-B/32")
            self.language_linear = nn.Linear(512, embed_dim)

        elif args.language_encoder == "T5":
            self.language_encoder = T5_LanguageEncoder(T5_model_path=args.T5_model_path, T5_model_freeze=self.args.text_freeze)
            self.language_linear = nn.Linear(512, embed_dim)

        elif args.language_encoder == "T5_and_CLIP_text":
            self.lanuage_encoder_t5 = T5_LanguageEncoder(T5_model_path=args.T5_model_path, T5_model_freeze=self.args.text_freeze)
            self.language_linear_t5 = nn.Linear(512, embed_dim)
            self.lanuage_encoder_clip = Clip_LanguageEncoder(clip_version="ViT-B/32")
            self.language_linear_clip = nn.Linear(512, embed_dim)

        else:
            self.language_encoder = LanguageEncoder(known_words, self.embed_dim, bi_dir=True)

        if args.use_cross_attention:
            cross_encoder_layer = TransformerCrossEncoderLayer(
                d_model=embed_dim,  # Embedding dimension
                nhead=8,  # Number of attention heads
                dim_feedforward=2048,  # Dimension of feedforward network
                dropout=0.1,  # Dropout rate
                activation="relu",  # Activation function
                normalize_before=True,  # Layer normalization
                sa_val_has_pos_emb=False,  # Self-attention value has positional embedding
                ca_val_has_pos_emb=False,  # Cross-attention value has positional embedding
                attention_type='dot_prod'  # Type of attention mechanism
            )

            # Create a TransformerCrossEncoder with a single layer
            self.transformer_cross_encoder = TransformerCrossEncoder(
                cross_encoder_layer=cross_encoder_layer,
                num_layers=2,
                norm=nn.LayerNorm(embed_dim),
                return_intermediate=False
            )
        # offset prediction
        # if args.only_clip_semantic_feature:
        #     self.mlp_offsets = get_mlp_offset([512, 512 // 2, 2])
        # else:
        #     self.mlp_offsets = get_mlp_offset([self.embed_dim, self.embed_dim // 2, 2])

        self.mlp_offsets = get_mlp_offset([self.embed_dim, self.embed_dim // 2, 2])

        if args.only_clip_semantic_feature:
            config = {
                "descriptor_dim": self.embed_dim,
                "GNN_layers": ["self", "cross"] * self.num_layers,
                # 'GNN_layers': ['self', ] * self.num_layers,
                "sinkhorn_iterations": self.sinkhorn_iters,
                "match_threshold": 0.2,
            }
        else:
            config = {
                "descriptor_dim": self.embed_dim,
                "GNN_layers": ["self", "cross"] * self.num_layers,
                # 'GNN_layers': ['self', ] * self.num_layers,
                "sinkhorn_iterations": self.sinkhorn_iters,
                "match_threshold": 0.2,
            }
        self.superglue = SuperGlue(config)

        print("DEVICE", self.get_device())

    def forward(self, objects, hints, object_points):
        batch_size = len(objects)
        num_objects = len(objects[0])
        # print(f"batch_size: {batch_size}, num_objects: {num_objects}")
        """
        Encode the hints
        """
        if self.args.language_encoder == "CLIP_text" or self.args.language_encoder == "T5":
            description_encodings_ori = self.language_encoder(hints)
            description_encodings = self.language_linear(description_encodings_ori)
            # if self.embed_dim == 512:
            #     description_encodings += description_encodings_ori
            hint_encodings = F.normalize(description_encodings)
        elif self.args.language_encoder == "T5_and_CLIP_text":
            description_encodings_t5 = self.lanuage_encoder_t5(hints)
            description_encodings_t5 = self.language_linear_t5(description_encodings_t5)
            description_encodings = self.lanuage_encoder_clip(hints)
            description_encodings = self.language_linear_clip(description_encodings)
            hint_encodings = F.normalize(description_encodings)
        else:
            hint_encodings = torch.stack(
                [self.language_encoder(hint_sample) for hint_sample in hints]
            )  # [B, num_hints, DIM]
            hint_encodings = F.normalize(hint_encodings, dim=-1)  # Norming those too
        # print("hint_encodings", hint_encodings.shape)
        """
        Object encoder
        """
        object_encodings, class_embedding, color_embedding, pos_embedding, num_points_embedding, relation_embedding = self.object_encoder(objects, object_points)
        if self.args.only_clip_semantic_feature or self.args.use_clip_semantic_feature:
            object_encodings = self.linear_obj(object_encodings)
            object_encodings = object_encodings.reshape((batch_size, num_objects, self.embed_dim))
            object_encodings = F.normalize(object_encodings, dim=-1)
        else:
            object_encodings = object_encodings.reshape((batch_size, num_objects, self.embed_dim))
            object_encodings = F.normalize(object_encodings, dim=-1)

        """
        Match object-encodings to hint-encodings
        """
        desc0 = object_encodings.transpose(1, 2)  # [B, DIM, num_obj]
        desc1 = hint_encodings.transpose(1, 2)  # [B, DIM, num_hints]
        # print("obj:", desc0.shape, "hint:", desc1.shape)

        matcher_output = self.superglue(desc0, desc1)

        # Initialize a list to store the extracted objects
        extracted_objects = []
        hints = matcher_output["matches1"]
        # print(hints.shape)
        # print(len(hints))
        # all_pad = 0
        for batch_idx in range(len(hints)):
            # Extract indices for this batch
            indices = hints[batch_idx]

            # Filter out -1 (or any invalid index)
            valid_indices = indices[indices >= 0]

            # Extract objects corresponding to valid indices
            objs = object_encodings[batch_idx, valid_indices]

            # 计算需要补齐的长度
            pad_size = 6 - objs.shape[0]
            # all_pad += pad_size
            # 如果需要，进行补齐
            if pad_size > 0:
                objs = F.pad(objs, (0, 0, 0, pad_size), 'constant', 0)

            # Append the extracted objects to the list
            extracted_objects.append(objs)
        extracted_objects = torch.stack(extracted_objects)
        # print("extracted_objects", extracted_objects.shape, "hints:", hint_encodings.shape)
        # print("all_pad", all_pad)

        # TODO: cross attention between object and hint
        if self.args.use_cross_attention:
            if self.args.language_encoder == "T5_and_CLIP_text":
                description_encodings_t5, extracted_objects = description_encodings_t5.transpose(0, 1), extracted_objects.transpose(0, 1)
                hint_encodings, object_encodings, _ = self.transformer_cross_encoder(description_encodings_t5, extracted_objects)
                hint_encodings, object_encodings = hint_encodings[0].transpose(0, 1), object_encodings[0].transpose(0, 1)
            else:
                # print(f"hint_encodings: {hint_encodings.shape}, extracted_objects: {extracted_objects.shape}")
                hint_encodings_c, extracted_objects_c = hint_encodings.transpose(0, 1), extracted_objects.transpose(0, 1)
                hint_encodings_c, object_encodings_c, _ = self.transformer_cross_encoder(hint_encodings_c, extracted_objects_c)
                hint_encodings, object_encodings = hint_encodings_c[0].transpose(0, 1) + hint_encodings, object_encodings_c[0].transpose(0, 1) + extracted_objects
        """
        Predict offsets from hints
        """
        offsets = self.mlp_offsets(hint_encodings)  # [B, num_hints, 2]

        outputs = EasyDict()
        outputs.P = matcher_output["P"]  # [B, num_obj, num_hints]
        outputs.matches0 = matcher_output["matches0"]  # [B, num_obj]
        outputs.matches1 = matcher_output["matches1"]  # [B, num_hints]

        outputs.offsets = offsets

        # print(f"output_p: {outputs.P[0]}")
        # print("matcher_output", outputs.matches0.shape, outputs.matches1.shape)
        # print("matcher_output", outputs.matches0[0], outputs.matches1[0])
        # quit()

        outputs.matching_scores0 = matcher_output["matching_scores0"]
        outputs.matching_scores1 = matcher_output["matching_scores1"]

        return outputs

    @property
    def device(self):
        return next(self.mlp_offsets.parameters()).device

    def get_device(self):
        return next(self.mlp_offsets.parameters()).device


def get_pos_in_cell(objects: List[Object3d_K360], matches0, offsets):
    """Extract a pose estimation relative to the cell (∈ [0,1]²) by
    adding up for each matched objects its location plus offset-vector of corresponding hint,
    then taking the average.

    Args:
        objects (List[Object3d_K360]): List of objects of the cell
        matches0 : matches0 from SuperGlue
        offsets : Offset predictions for each hint

    Returns:
        np.ndarray: Pose estimate
    """
    pose_preds = []  # For each match the object-location plus corresponding offset-vector
    for obj_idx, hint_idx in enumerate(matches0):
        if obj_idx == -1 or hint_idx == -1:
            continue
        # pose_preds.append(objects[obj_idx].closest_point[0:2] + offsets[hint_idx]) # Object location plus offset of corresponding hint
        pose_preds.append(
            objects[obj_idx].get_center()[0:2] + offsets[hint_idx]
        )  # Object location plus offset of corresponding hint
    return (
        np.mean(pose_preds, axis=0) if len(pose_preds) > 0 else np.array((0.5, 0.5))
    )  # Guess the middle if no matches


def intersect(P0, P1):
    n = (P1 - P0) / np.linalg.norm(P1 - P0, axis=1)[:, np.newaxis]  # normalized
    projs = np.eye(n.shape[1]) - n[:, :, np.newaxis] * n[:, np.newaxis]  # I - n*n.T
    R = projs.sum(axis=0)
    q = (projs @ P0[:, :, np.newaxis]).sum(axis=0)
    p = np.linalg.lstsq(R, q, rcond=None)[0]
    return p


def get_pos_in_cell_intersect(objects: List[Object3d_K360], matches0, directions):
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
    points0 = []
    points1 = []
    for obj_idx, hint_idx in enumerate(matches0):
        if obj_idx == -1 or hint_idx == -1:
            continue
        points0.append(objects[obj_idx].get_center()[0:2])
        points1.append(objects[obj_idx].get_center()[0:2] + directions[hint_idx])
    if len(points0) < 2:
        return np.array((0.5, 0.5))
    else:
        return intersect(np.array(points0), np.array(points1))


if __name__ == "__main__":
    args = EasyDict()
    args.embed_dim = 16
    args.num_layers = 2
    args.sinkhorn_iters = 10
    args.num_mentioned = 4
    args.pad_size = 8
    args.use_features = ["class", "color", "position"]
    args.pointnet_layers = 3
    args.pointnet_variation = 0

    # dataset_train = Semantic3dPoseReferanceMockDataset(args, length=1024)
    # dataloader_train = DataLoader(dataset_train, batch_size=2, collate_fn=Semantic3dPoseReferanceMockDataset.collate_fn)
    # data = dataset_train[0]
    # batch = next(iter(dataloader_train))

    model = SuperGlueMatch(
        ["class1", "class2"], ["word1", "word2"], args, "./checkpoints/pointnet_K360.pth"
    )

    # out = model(batch['objects'], batch['hint_descriptions'])

    print("Done")
