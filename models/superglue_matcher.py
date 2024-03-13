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
from models.modules import get_mlp, TransformerWithMaxPool, LanguageEncoder, Clip_LanguageEncoder, MaxPoolMultiHeadSelfAttention, Clip_LanguageEncoder_TransformerFuser, MaxPoolRelationMultiHeadSelfAttention, T5_LanguageEncoder
from models.cross_attention import TransformerCrossEncoderLayer, TransformerCrossEncoder


COLORS = (
    np.array(
        [
            [47.2579917, 49.75368454, 42.4153065],
            [136.32696657, 136.95241796, 126.02741229],
            [87.49822126, 91.69058836, 80.14558512],
            [213.91030679, 216.25033052, 207.24611073],
            [110.39218852, 112.91977458, 103.68638249],
            [27.47505158, 28.43996795, 25.16840296],
            [66.65951839, 70.22342483, 60.20395996],
            [171.00852191, 170.05737735, 155.00130334],
        ]
    )
    / 255.0
)

CLASS_TO_INDEX = {
    "building": 0,
    "pole": 1,
    "traffic light": 2,
    "traffic sign": 3,
    "garage": 4,
    "stop": 5,
    "smallpole": 6,
    "lamp": 7,
    "trash bin": 8,
    "vending machine": 9,
    "box": 10,
    "road": 11,
    "sidewalk": 12,
    "parking": 13,
    "wall": 14,
    "fence": 15,
    "guard rail": 16,
    "bridge": 17,
    "tunnel": 18,
    "vegetation": 19,
    "terrain": 20,
    "pad": 21,
}

COLOR_NAMES = ["dark-green", "gray", "gray-green", "bright-gray", "gray", "black", "green", "beige"]

import MinkowskiEngine as ME
from models.mask3d import Mask3D

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

        # cell encoder
        if args.no_objects:
            self.cell_encoder = Mask3D()

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
                num_layers=4,
                norm=nn.LayerNorm(embed_dim),
                return_intermediate=False
            )

        if args.no_superglue:
            self.final_attn_pool = TransformerWithMaxPool(embed_dim, 8, 2, 1024)
        # offset prediction
        # if args.only_clip_semantic_feature:
        #     self.mlp_offsets = get_mlp_offset([512, 512 // 2, 2])
        # else:
        #     self.mlp_offsets = get_mlp_offset([self.embed_dim, self.embed_dim // 2, 2])

        self.mlp_offsets = get_mlp_offset([self.embed_dim, self.embed_dim // 2, 2])

        if not args.no_superglue:
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

        if not self.args.no_superglue:
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
        else:
            extracted_objects = object_encodings

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

        if self.args.no_superglue:
            hint_encodings = self.final_attn_pool(hint_encodings.transpose(0, 1))
        """
        Predict offsets from hints
        """
        offsets = self.mlp_offsets(hint_encodings)  # [B, num_hints, 2]
        if not self.args.no_superglue:
            outputs = EasyDict()
            outputs.P = matcher_output["P"]  # [B, num_obj, num_hints]
            outputs.matches0 = matcher_output["matches0"]  # [B, num_obj]
            outputs.matches1 = matcher_output["matches1"]  # [B, num_hints]
            outputs.offsets = offsets
            outputs.matching_scores0 = matcher_output["matching_scores0"]
            outputs.matching_scores1 = matcher_output["matching_scores1"]
        else:
            outputs = EasyDict()
            outputs.offsets = offsets

        return outputs

    def encode_cell(self, hints, cell_coordinates, cells_features, target, inverse_map, batched_raw_coordinates=None, batched_raw_color=None, num_query = 24, freeze_cell_encoder=False, use_queries=False, offset=True,):

        # print(f"batch_size: {batch_size}, num_objects: {num_objects}")
        """
        Encode the hints
        """
        if self.args.language_encoder == "CLIP_text" or self.args.language_encoder == "T5":
            description_encodings_ori = self.language_encoder(hints)
            description_encodings = self.language_linear(description_encodings_ori)
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
        cell encoder
        """
        raw_coordinates = cells_features[:, -3:]
        cells_features = cells_features[:, :-3]
        data = ME.SparseTensor(
            coordinates=cell_coordinates,
            features=cells_features,
            device=self.device,
        )
        # print(f"cell_coordinates: {cell_coordinates.shape}, cells_features: {cells_features.shape}")
        # print(data.decomposed_features[0].shape)
        for i in range(len(target)):
            for key in target[i]:
                # print(key)
                target[i][key] = target[i][key].to(device=self.device)
        # print(data.shape)

        if freeze_cell_encoder:
            with torch.no_grad():
                output_dict = self.cell_encoder(data, point2segment=[
                    target[i]["point2segment"] for i in range(len(target))], raw_coordinates=raw_coordinates,
                                                is_eval=False)
        else:
            output_dict = self.cell_encoder(data, point2segment=[
                target[i]["point2segment"] for i in range(len(target))], raw_coordinates=raw_coordinates, is_eval=False)

        queries = output_dict['queries']  # B, N, 128
        pred_class_list = []
        pred_mask = output_dict['pred_masks']
        mask_queries_list = []
        center_queries_list = []
        color_queries_list = []
        rgb_queries_list = []

        queries_feature_list = []
        for i in range(len(output_dict['pred_masks'])):
            # print(f"pred_masks: {output_dict['pred_masks'][i].shape}") # shape [N, num_query]
            pred_class = []
            mask_queries = []
            center_queries = []
            color_queries = []
            rgb_queries = []
            queries_feature = []

            gt_class = target[i]['labels']
            # print(f"gt_class: {gt_class}")
            for j in range(num_query):
                p_masks = torch.sigmoid(pred_mask[i][:, j])  # get sigmoid for instance masks
                m = p_masks > 0.5  # get the instance mask
                c_m = p_masks[m].sum() / (m.sum() + 1e-8)  # get the confidence of the instance mask
                c_label = torch.max(output_dict['pred_logits'][i][j])
                c_label_i = torch.argmax(output_dict['pred_logits'][i][j], dim=-1)
                c = c_label * c_m  # combine the confidence of the semantic label and the instance mask, unused
                # print(f"m shape: {m.shape}")
                # print(f"m: {m.sum()}")
                # print(f"c: {c}")
                # print(f"c_label_i: {c_label_i}", f"c: {c}")
                if c < 4 or m.sum() == 0 or c_label_i == 21:
                    continue
                # print(f"inverse_map[i]: {inverse_map[i].shape}")
                mask = m[inverse_map[i]]  # mapping the mask back to the original point scloud
                coordinates_b = batched_raw_coordinates[i]
                coordinates_b = torch.from_numpy(coordinates_b).to(self.device)
                # print(f"coordinates_b: {coordinates_b.shape}")
                coordinates_b_mask = coordinates_b[mask == 1]
                mask_queries.append(coordinates_b_mask)
                center_queries.append(torch.mean(coordinates_b_mask, dim=0) / 30)
                color_b = batched_raw_color[i]
                # print(f"color_b: {color_b.shape}")
                # print(f"color_b: {color_b}")
                indices = mask == 1
                # print(f"indices: {indices}")
                # print(f"indices: {sum(indices)}")
                color_b_mask = color_b[indices.cpu().numpy()]
                color_b_mask_mean = np.mean(color_b_mask, axis=0)
                dists = np.linalg.norm(color_b_mask_mean - COLORS, axis=1)
                color_name = COLOR_NAMES[np.argmin(dists)]
                if offset:
                    c_label_i += 1
                # print(f"c_label_i: {c_label_i}, color_name: {color_name}", "center: ", torch.mean(coordinates_b_mask, dim=0)/30, f"c: {c}")
                color_name_index = COLOR_NAMES.index(color_name)
                color_queries.append(torch.from_numpy(np.array([color_name_index])).to(self.device))
                rgb_queries.append(torch.from_numpy(color_b_mask_mean).to(self.device))
                pred_class.append(c_label_i)
                # print(f"coordinates_b_mask: {coordinates_b_mask.shape}")
                # print(f"mask: {mask.shape}")
                # print(f"mask: {mask.sum()}")
                # print(f"confidence: {c}")
                queries_feature.append(queries[i, j])
            assert len(mask_queries) == len(center_queries) == len(color_queries) == len(rgb_queries) == len(
                pred_class) == len(queries_feature)
            mask_queries_list.append(mask_queries)  # mask 非等长 无法stack
            center_queries_list.append(center_queries)  # center 等长 可以stack
            color_queries_list.append(color_queries)  # color 等长 可以stack
            rgb_queries_list.append(rgb_queries)
            pred_class_list.append(pred_class)
            queries_feature_list.append(queries_feature)
        assert len(mask_queries_list) == len(center_queries_list) == len(color_queries_list) == len(
            rgb_queries_list) == len(pred_class_list) == len(queries_feature_list)


        """
        Object encoder
        """
        if self.use_queries:
            object_encodings, class_embedding, color_embedding, pos_embedding, num_points_embedding, relation_embedding = self.object_encoder.forward_cell(class_indices=pred_class_list, color_indices=color_queries_list, rgbs=rgb_queries_list, positions=center_queries_list, queries=queries_feature_list)
        else:
            object_encodings, class_embedding, color_embedding, pos_embedding, num_points_embedding, relation_embedding = self.object_encoder.forward_cell(class_indices=pred_class_list, color_indices=color_queries_list, rgbs=rgb_queries_list, positions=center_queries_list)

        extracted_objects = object_encodings

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

        if self.args.no_superglue:
            hint_encodings = self.final_attn_pool(hint_encodings.transpose(0, 1))
        """
        Predict offsets from hints
        """
        offsets = self.mlp_offsets(hint_encodings)  # [B, num_hints, 2]
        outputs = EasyDict()
        outputs.offsets = offsets

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
