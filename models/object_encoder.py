from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from models.modules import get_mlp
from models.pointcloud.pointnet2 import PointNet2

from datapreparation.kitti360pose.imports import Object3d
from datapreparation.kitti360pose.utils import COLOR_NAMES


class ObjectEncoder(torch.nn.Module):
    def __init__(self, embed_dim: int, known_classes: List[str], known_colors: List[str], args):
        """Module to encode a set of objects

        Args:
            embed_dim (int): Embedding dimension
            known_classes (List[str]): List of known classes, only used for embedding ablation
            known_colors (List[str]): List of known colors, only used for embedding ablation
            args: Global training arguments
        """
        super(ObjectEncoder, self).__init__()

        self.embed_dim = embed_dim
        self.args = args

        # Set idx=0 for padding, class to idx mapping, then embedding
        if args.no_objects:
            self.known_classes = {c: i for i, c in enumerate(known_classes)}
            print(f"known_classes: {self.known_classes}")
            self.class_embedding = nn.Embedding(len(self.known_classes), embed_dim, padding_idx=0)
        else:
            self.known_classes = {c: (i + 1) for i, c in enumerate(known_classes)}
            self.known_classes["<unk>"] = 0
            # print(f"known_classes: {self.known_classes}")
            self.class_embedding = nn.Embedding(len(self.known_classes), embed_dim, padding_idx=0)
            # print(f"class_embedding: {self.class_embedding}")

        # Set idx=0 for padding, color to idx mapping, then embedding
        self.known_colors = {c: i for i, c in enumerate(COLOR_NAMES)}
        self.known_colors["<unk>"] = 0
        self.color_embedding = nn.Embedding(len(self.known_colors), embed_dim, padding_idx=0)

        if self.args.use_relation_transformer and not (self.args.only_clip_semantic_feature or self.args.use_clip_semantic_feature):
            self.relation_encoder = get_mlp([2, 64, embed_dim])
        elif self.args.use_relation_transformer and (self.args.only_clip_semantic_feature or self.args.use_clip_semantic_feature):
            self.relation_encoder = get_mlp([2, 64, embed_dim + 512])
        # self.relation_encoder = get_mlp([2, 64, embed_dim])  # OPTION: relation_encoder layers
        self.num_encoder = get_mlp([1, 64, embed_dim])  # OPTION: num_encoder layers
        self.pos_encoder = get_mlp([3, 64, embed_dim])  # OPTION: pos_encoder layers
        self.color_encoder = get_mlp([3, 64, embed_dim])  # OPTION: color_encoder layers

        # use pointnet++ to extract semantic features
        if not args.only_clip_semantic_feature or not args.no_objects:
            self.pointnet = PointNet2(
                len(known_classes), len(known_colors), args
            )  # The known classes are all the same now, at least for K360
            self.pointnet.load_state_dict(torch.load(args.pointnet_path))
            # self.pointnet_dim = self.pointnet.lin2.weight.size(0)

            if args.pointnet_freeze:
                print("CARE: freezing PN")
                self.pointnet.requires_grad_(False)
            if args.pointnet_features == 0:
                self.mlp_pointnet = get_mlp([self.pointnet.dim0, self.embed_dim])
            elif args.pointnet_features == 1:
                self.mlp_pointnet = get_mlp([self.pointnet.dim1, self.embed_dim])
            elif args.pointnet_features == 2:
                self.mlp_pointnet = get_mlp([self.pointnet.dim2, self.embed_dim])

        merged_embedding_dim = len(args.use_features) * embed_dim
        if "class_embed" in args and args.class_embed:
            merged_embedding_dim += embed_dim
        if "color_embed" in args and args.color_embed:
            merged_embedding_dim += embed_dim
        if "num_embed" in args and args.num_embed:
            merged_embedding_dim += embed_dim
        print(f"merged_embedding_dim, {merged_embedding_dim}")
        if args.no_objects:
            merged_embedding_dim -= 256
        if args.no_superglue:
            merged_embedding_dim += 256
        if args.use_queries:
            self.query_linear = nn.Linear(128, embed_dim)
            merged_embedding_dim += embed_dim
        if args.distill_query_embedding:
            self.query_linear = nn.Linear(128, embed_dim)
        self.mlp_merge = get_mlp([merged_embedding_dim, embed_dim])

        if args.use_semantic_head:
            self.semantic_head = SemanticHead(embed_dim, len(known_classes))


    def forward(self, objects: List[Object3d], object_points):
        """Features are currently normed before merging but not at the end.

        Args:
            objects (List[List[Object3d]]): List of lists of objects
            object_points (List[Batch]): List of PyG-Batches of object-points
        """

        if not hasattr(self, "color_encoder"):
            self.color_encoder = self.color_embedding
        if not hasattr(self, "pos_encoder"):
            self.pos_encoder = self.pos_embedding

        if ("class_embed" in self.args and self.args.class_embed) or (
            "color_embed" in self.args and self.args.color_embed
        ):
            class_indices = []
            color_indices = []
            for i_batch, objects_sample in enumerate(objects):
                for obj in objects_sample:
                    class_idx = self.known_classes.get(obj.label, 0)
                    class_indices.append(class_idx)
                    color_idx = self.known_colors[obj.get_color_text()]
                    color_indices.append(color_idx)

        # if "class_embed" not in self.args or self.args.class_embed == False:
        #     # Void all colors for ablation
        if "color" not in self.args.use_features:
            for pyg_batch in object_points:
                pyg_batch.x[:] = 0.0  # x is color, pos is xyz

        if not self.args.only_clip_semantic_feature:
            object_features = [
                self.pointnet(pyg_batch.to(self.get_device())).features2
                for pyg_batch in object_points
            ]  # [B, obj_counts, PN_dim]

            object_features_p = torch.cat(object_features, dim=0)  # [total_objects, PN_dim]
            object_features = self.mlp_pointnet(object_features_p)

            if self.args.use_semantic_head:
                object_features_sem = self.semantic_head(object_features_p)  # [total_objects, num_classes]
                # 还原batch [B, obj_counts, PN_dim]
                # object_features_sem = torch.split(object_features, [pyg_batch.num_nodes for pyg_batch in object_points], dim=0)
                # shape: [B, obj_counts, num_classes]

        embeddings = []

        class_embedding = None
        if "class" in self.args.use_features:
            if (
                "class_embed" in self.args and self.args.class_embed
            ):  # Use fixed embedding (ground-truth data!)
                # print(torch.tensor(class_indices, dtype=torch.long, device=self.get_device()).shape)
                class_embedding = self.class_embedding(
                    torch.tensor(class_indices, dtype=torch.long, device=self.get_device())
                )
                class_embedding = F.normalize(class_embedding, dim=-1)
                # print(f"class_embedding: {class_embedding.shape}")
                # print(f"class_embedding: {class_embedding}")
                # NOW: using both class embedding and object features
                embeddings.append(class_embedding)
                embeddings.append(
                    F.normalize(object_features, dim=-1)
                )  # Use features from PointNet
            elif not self.args.only_clip_semantic_feature:
                embeddings.append(
                    F.normalize(object_features, dim=-1)
                )  # Use features from PointNet

        color_embedding = None
        if "color" in self.args.use_features:
            if "color_embed" in self.args and self.args.color_embed:
                # print(torch.tensor(color_indices, dtype=torch.long, device=self.get_device()).shape)
                color_label_embedding = self.color_embedding(
                    torch.tensor(color_indices, dtype=torch.long, device=self.get_device())
                )
                # print(f"color_label_embedding: {color_label_embedding.shape}")
                # NOW: use both color embedding and object.rgb features
                embeddings.append(F.normalize(color_label_embedding, dim=-1))
                colors = []
                for objects_sample in objects:
                    colors.extend([obj.get_color_rgb() for obj in objects_sample])
                colors = np.array(colors)
                # print(f"colors: {colors.shape}")
                color_embedding = self.color_encoder(
                    torch.tensor(colors, dtype=torch.float, device=self.get_device())
                )
                color_embedding = F.normalize(color_embedding, dim=-1)
                # print(f"color_embedding: {color_embedding.shape}")
                # print(f"color_embedding: {color_embedding}")
                embeddings.append(color_embedding)
            else:
                colors = []
                for objects_sample in objects:
                    colors.extend([obj.get_color_rgb() for obj in objects_sample])
                colors = np.array(colors)
                color_embedding = self.color_encoder(
                    torch.tensor(colors, dtype=torch.float, device=self.get_device())
                )
                color_embedding = F.normalize(color_embedding, dim=-1)
                embeddings.append(color_embedding)

        pos_embedding = None
        if "position" in self.args.use_features:
            positions = []
            for objects_sample in objects:
                positions.extend([obj.get_center() for obj in objects_sample])
            positions = np.array(positions)
            # print(f"positions: {torch.tensor(positions, dtype=torch.float, device=self.get_device()).shape}")
            # print(f"positions: {torch.tensor(positions, dtype=torch.float, device=self.get_device())}")
            pos_embedding = self.pos_encoder(
                torch.tensor(positions, dtype=torch.float, device=self.get_device())
            )
            # print(f"pos_embedding: {pos_embedding.shape}")
            embeddings.append(F.normalize(pos_embedding, dim=-1))

        num_points_embedding = None
        if "num_points" in self.args.use_features:
            num_points = []
            for objects_sample in objects:
                num_points.extend([obj.xyz.shape[0] for obj in objects_sample])
            num_points = np.array(num_points)
            num_points_embedding = self.num_encoder(
                torch.tensor(num_points, dtype=torch.float, device=self.get_device())
            )
            embeddings.append(F.normalize(num_points_embedding, dim=-1))

        relation_embedding = None
        if self.args.use_relation_transformer:
            # get relation matrix
            relations = []
            for i_batch, objects_sample in enumerate(objects):
                centers_i = torch.tensor(np.array([obj.get_center()[0:2] for obj in objects_sample]), dtype=torch.float, device=self.get_device())
                centers_j = torch.tensor(np.array([obj.get_center()[0:2] for obj in objects_sample]), dtype=torch.float, device=self.get_device())
                # Broadcasting the subtraction operation over the two tensors.
                # The unsqueeze method is used to add the necessary dimension for broadcasting.
                relation_matrix = centers_i.unsqueeze(1) - centers_j.unsqueeze(0)  # [num_obj, num_obj, 2] tensor
                relations.append(relation_matrix)
            # pad relations to the same size (the max number of objects in a batch)
            max_num_obj = max([len(objs) for objs in objects])
            for i, relation in enumerate(relations):
                relations[i] = F.pad(relation, (0, 0, 0, max_num_obj - relation.shape[0], 0, max_num_obj - relation.shape[0]))
            relations = torch.stack(relations, dim=0)

            B, max_n, _, _ = relations.shape
            relations = relations.reshape(B*max_n*max_n, 2)
            relation_embedding = self.relation_encoder(relations)
            if self.args.use_clip_semantic_feature or self.args.only_clip_semantic_feature:
                relation_embedding = relation_embedding.reshape(B, max_n, max_n, self.embed_dim + 512)
            else:
                relation_embedding = relation_embedding.reshape(B, max_n, max_n, self.embed_dim)
            # print("relation_embedding", relation_embedding.shape)

        if len(embeddings) > 1:
            # for i, embedding in enumerate(embeddings):
            #     print(f"embedding {i}: {embedding.shape}")
            embeddings = self.mlp_merge(torch.cat(embeddings, dim=-1))
        else:
            embeddings = embeddings[0]

        # if use clip feature as instance semantic feature, concat clip feature with object embedding
        if self.args.only_clip_semantic_feature or self.args.use_clip_semantic_feature:
            clip_2d_features = []
            for objects_sample in objects:
                clip_2d_features.extend([obj.feature_2d for obj in objects_sample])
            clip_2d_features = np.array(clip_2d_features)
            # print(f"clip_2d_features: {clip_2d_features.shape}")
            clip_2d_features = torch.tensor(clip_2d_features, dtype=torch.float, device=self.get_device())
            clip_2d_features = clip_2d_features.squeeze(1)
            # print(f"clip_2d_features: {clip_2d_features.shape}")
            # print(f"embeddings: {embeddings.shape}")
            # normalize embeddings
            embeddings = F.normalize(embeddings, dim=-1)
            # print(f"embeddings: {embeddings.max()}, {embeddings.min()}")
            # print(f"clip_2d_features: {clip_2d_features.max()}, {clip_2d_features.min()}")
            embeddings = torch.cat((embeddings, clip_2d_features), dim=-1)  # [N, embedding_dim + clip_dim]
        # print(f"embeddings: {embeddings.shape}")
        if self.args.use_semantic_head:
            return embeddings, class_embedding, color_embedding, pos_embedding, num_points_embedding, relation_embedding, object_features_sem
        return embeddings, class_embedding, color_embedding, pos_embedding, num_points_embedding, relation_embedding


    def     forward_cell(self, class_indices, color_indices, rgbs, positions, queries=None, num_points=None):
        """Features are currently normed before merging but not at the end.

        Args:
            objects (List[List[Object3d]]): List of lists of objects
            object_points (List[Batch]): List of PyG-Batches of object-points
        """

        # if not hasattr(self, "color_encoder"):
        #     self.color_encoder = self.color_embedding
        # if not hasattr(self, "pos_encoder"):
        #     self.pos_encoder = self.pos_embedding

        # class_indices_nobatch = []
        # color_indices_nobatch = []
        # rgb_nobatch = []
        # positions_nobatch = []
        # for i_batch, in range(len(class_indices)):
        #     for class_id in class_indices[i_batch]:
        #         color_indices_nobatch.append(class_id)
        #     for color_id in color_indices[i_batch]:
        #         color_indices_nobatch.append(color_id)
        #     for rgb in rgbs[i_batch]:
        #         rgb_nobatch.append(rgb)
        #     for position in positions[i_batch]:
        #         positions_nobatch.append(position)
        # for i_batch in range(len(class_indices)):
        #     print(f"in batch: {i_batch}, there are objs: {len(class_indices[i_batch])}")

        class_indices_nobatch = [class_id for class_list in class_indices for class_id in class_list]
        color_indices_nobatch = [color_id for color_list in color_indices for color_id in color_list]
        rgb_nobatch = [rgb for rgb_list in rgbs for rgb in rgb_list]
        positions_nobatch = [position for position_list in positions for position in position_list]
        if queries is not None:
            queries_nobatch = [query for query_list in queries for query in query_list]

        embeddings = []

        class_embedding_ori = class_embedding = None
        if "class" in self.args.use_features:
            if (
                "class_embed" in self.args and self.args.class_embed
            ):  # Use fixed embedding (ground-truth data!)
                class_embedding = self.class_embedding(
                    torch.tensor(class_indices_nobatch, dtype=torch.long, device=self.get_device())
                )
                class_embedding_ori = class_embedding
                class_embedding = F.normalize(class_embedding, dim=-1)
                # print(f"class_embedding: {class_embedding.shape}")
                # print(f"class_embedding: {class_embedding}")
                # NOW: using both class embedding and object features
                embeddings.append(class_embedding)
        # print(class_embedding.shape)

        queries_embedding_ori = queries_embedding = None
        if self.args.use_queries or self.args.distill_query_embedding:
            queries_nobatch_tensor = torch.stack(queries_nobatch)
            queries_embedding = self.query_linear(
                queries_nobatch_tensor.float()
            )
            queries_embedding_ori = queries_embedding
            queries_embedding = F.normalize(queries_embedding, dim=-1)
            if self.args.use_queries:
                embeddings.append(queries_embedding)
            # embeddings.append(F.normalize(queries_embedding, dim=-1))
        #print(queries_embedding.shape)

        color_embedding = None
        if "color" in self.args.use_features:
            if "color_embed" in self.args and self.args.color_embed:
                color_label_embedding = self.color_embedding(
                    # color_indices.long()
                    torch.tensor(color_indices_nobatch, dtype=torch.long, device=self.get_device())
                )
                # NOW: use both color embedding and object.rgb features
                # print(f"color_label_embedding: {color_label_embedding.shape}")
                # print(f"color_label_embedding: {color_label_embedding}")
                embeddings.append(F.normalize(color_label_embedding, dim=-1))
                rgb_nobatch_tensor = torch.stack(rgb_nobatch)
                # print(f"rgb_nobatch_tensor: {rgb_nobatch_tensor}")
                color_embedding = self.color_encoder(
                   rgb_nobatch_tensor.float()
                   # torch.tensor(rgb_nobatch, dtype=torch.float, device=self.get_device())
                )
                color_embedding = F.normalize(color_embedding, dim=-1)
                # print(f"color_embedding: {color_embedding.shape}")
                # print(f"color_embedding: {color_embedding}")
                embeddings.append(color_embedding)
            else:
                color_embedding = self.color_encoder(
                   rgbs.float()
                )
                color_embedding = F.normalize(color_embedding, dim=-1)
                embeddings.append(color_embedding)

        pos_embedding = None
        if "position" in self.args.use_features:
            # print(f"positions_nobatch: {positions_nobatch}")
            positions_nobatch_tensor = torch.stack(positions_nobatch)
            pos_embedding = self.pos_encoder(
                positions_nobatch_tensor.float()
                # torch.tensor(positions_nobatch, dtype=torch.float, device=self.get_device())
            )
            # print(f"pos_embedding: {pos_embedding.shape}")
            # print(f"pos_embedding: {pos_embedding}")
            embeddings.append(F.normalize(pos_embedding, dim=-1))

        num_points_embedding = None
        # if "num_points" in self.args.use_features:
        #     b, n, e = num_points.shape
        #     num_points = num_points.view(b * n, e)
        #     num_points_embedding = self.num_encoder(
        #         num_points.float()
        #     )
        #     embeddings.append(F.normalize(num_points_embedding, dim=-1))

        relation_embedding = None
        if self.args.use_relation_transformer:
            # # get relation matrix
            # # print("positions", positions.shape)
            # positions = positions.float()
            # positions = positions[:, :, :2]
            # b, n, _ = positions.size()
            # expanded_positions = positions.unsqueeze(2)  # [b, n, 1, 2]
            # expanded_positions = expanded_positions.expand(b, n, n, 2)  # [b, n, n, 2]
            # relations = expanded_positions - expanded_positions.transpose(1, 2)  # [b, n, n, 2]
            #
            # relations = relations.reshape(b*n*n, 2)
            # relation_embedding = self.relation_encoder(relations)
            # relation_embedding = relation_embedding.reshape(b, n, n, self.embed_dim)
            # # print("relation_embedding", relation_embedding.shape)
            # get relation matrix
            relations = []
            # print(positions)
            for i_batch, objects_sample in enumerate(positions):
                centers_i = torch.stack([obj[:2] for obj in objects_sample]).float()
                centers_j = torch.stack([obj[:2] for obj in objects_sample]).float()
                # Broadcasting the subtraction operation over the two tensors.
                # The unsqueeze method is used to add the necessary dimension for broadcasting.
                relation_matrix = centers_i.unsqueeze(1) - centers_j.unsqueeze(0)  # [num_obj, num_obj, 2] tensor
                # print("relation_matrix", relation_matrix.shape)
                relations.append(relation_matrix)
            # pad relations to the same size (the max number of objects in a batch)
            max_num_obj = max([len(objs) for objs in positions])
            for i, relation in enumerate(relations):
                relations[i] = F.pad(relation,
                                     (0, 0, 0, max_num_obj - relation.shape[0], 0, max_num_obj - relation.shape[0]))
            relations = torch.stack(relations, dim=0)

            B, max_n, _, _ = relations.shape
            relations = relations.reshape(B * max_n * max_n, 2)
            relation_embedding = self.relation_encoder(relations)
            relation_embedding = relation_embedding.reshape(B, max_n, max_n, self.embed_dim)
            # print("relation_embedding", relation_embedding.shape)

        if len(embeddings) > 1:
            # for i, embedding in enumerate(embeddings):
            #     print(f"embedding {i}: {embedding.shape}")
            embeddings = self.mlp_merge(torch.cat(embeddings, dim=-1))
        else:
            embeddings = embeddings[0]
        if self.args.use_queries or self.args.distill_query_embedding:
            return embeddings, class_embedding_ori, color_embedding, pos_embedding, num_points_embedding, relation_embedding, queries_embedding_ori
        else:
            return embeddings, class_embedding, color_embedding, pos_embedding, num_points_embedding, relation_embedding


    @property
    def device(self):
        return next(self.class_embedding.parameters()).device

    def get_device(self):
        return next(self.class_embedding.parameters()).device


class SemanticHead(nn.Module):
    def __init__(self, input_dim,  output_dim):
        super(SemanticHead, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x

