from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
import os
import pickle
from easydict import EasyDict
import sys
sys.path.append("/home/wanglichao/Text2Position")
from models.modules import get_mlp, LanguageEncoder, Clip_LanguageEncoder, MaxPoolMultiHeadSelfAttention, Clip_LanguageEncoder_TransformerFuser, MaxPoolRelationMultiHeadSelfAttention, T5_LanguageEncoder_TransformerFuser
from models.minkowski import *
from models.object_encoder import ObjectEncoder
import clip
import MinkowskiEngine as ME

#from dataloading.semantic3d.semantic3d import Semantic3dCellRetrievalDataset
#from dataloading.semantic3d.semantic3d_poses import Semantic3dPosesDataset
from models.mask3d import Mask3D


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


class CellRetrievalNetwork(torch.nn.Module):
    def __init__(
        self, known_classes: List[str], known_colors: List[str], known_words: List[str], args
    ):
        """Coarse module for cell retrieval.
        Implemented as a language encode and an object encoder.
        The object encoder aggregates information about a varying count of multiple objects through a DGCNN architecture.
        """
        super(CellRetrievalNetwork, self).__init__()
        self.embed_dim = args.embed_dim
        self.use_features = args.use_features
        self.variation = args.variation
        self.use_edge_conv = args.use_edge_conv
        self.only_clip_semantic_feature = args.only_clip_semantic_feature
        self.use_clip_semantic_feature = args.use_clip_semantic_feature
        self.use_relation_transformer = args.use_relation_transformer
        self.args = args
        embed_dim = self.embed_dim

        assert args.variation in (0, 1)

        """
        Object path
        """
        self.lin = get_mlp([embed_dim, embed_dim, embed_dim])
        print("use pyg edgeconv: ", args.use_edge_conv)
        if args.use_edge_conv:
        # CARE: possibly handle variation in forward()!
            if self.variation == 0:
                self.graph1 = gnn.DynamicEdgeConv(
                    get_mlp([2 * embed_dim, embed_dim, embed_dim], add_batchnorm=True), k=8, aggr="max"
                )  # Originally: k=4
                self.lin = get_mlp([embed_dim, embed_dim, embed_dim])
            elif self.variation == 1:
                self.graph1 = gnn.DynamicEdgeConv(
                    get_mlp([2 * embed_dim, embed_dim, embed_dim], add_batchnorm=True), k=8, aggr="mean"
                )  # Originally: k=4
                self.lin = get_mlp([embed_dim, embed_dim, embed_dim])
        elif (args.only_clip_semantic_feature or args.use_clip_semantic_feature) and not args.use_relation_transformer:
            self.attn_pooling = MaxPoolMultiHeadSelfAttention(embed_dim+512, num_heads=8)
            self.final_linear = nn.Linear(embed_dim + 512, embed_dim)
        elif (args.only_clip_semantic_feature or args.use_clip_semantic_feature) and args.use_relation_transformer:
            self.attn_pooling = MaxPoolRelationMultiHeadSelfAttention(embed_dim+512, num_heads=8)
            self.final_linear = nn.Linear(embed_dim + 512, embed_dim)
        elif args.use_relation_transformer and not (args.only_clip_semantic_feature or args.use_clip_semantic_feature):
            self.attn_pooling = MaxPoolRelationMultiHeadSelfAttention(embed_dim, num_heads=8)


        elif args.no_objects and not args.use_relation_transformer:
            self.attn_pooling = MaxPoolMultiHeadSelfAttention(embed_dim, num_heads=8)
            # self.final_linear = nn.Linear(128, embed_dim)

        elif args.no_objects and args.use_relation_transformer:
            self.attn_pooling = MaxPoolRelationMultiHeadSelfAttention(embed_dim, num_heads=8)
            # self.final_linear = nn.Linear(128, embed_dim)

        else:  # use attention + pooling
            self.attn_pooling = MaxPoolMultiHeadSelfAttention(embed_dim, num_heads=8)

        self.object_encoder = ObjectEncoder(embed_dim, known_classes, known_colors, args)
        if self.args.no_objects:
            # self.cell_encoder = ResNet34(in_channels=3, out_channels=embed_dim, D=3)
            self.cell_encoder = Mask3D()


        """
        Textual path
        """
        if args.language_encoder == "CLIP_text":
            self.language_encoder = Clip_LanguageEncoder(clip_version="ViT-B/32")
            self.language_linear = nn.Linear(512, embed_dim)
            # self.language_linear_object = nn.Linear(512, embed_dim)
            self.language_linear_submap = nn.Linear(512, embed_dim)

        elif args.language_encoder == "CLIP_text_transformer":
            self.language_encoder = Clip_LanguageEncoder_TransformerFuser(clip_version="ViT-B/32", clip_text_freeze=self.args.text_freeze)
            self.language_linear = nn.Linear(512, embed_dim)
            # self.language_linear_object = nn.Linear(512, embed_dim)
            self.language_linear_submap = nn.Linear(512, embed_dim)

        elif args.language_encoder == "T5_text_transformer":
            self.language_encoder = T5_LanguageEncoder_TransformerFuser(T5_model_path=args.T5_model_path, T5_model_freeze=self.args.text_freeze)
            self.language_linear_submap = nn.Linear(512, embed_dim)
        else:
            self.language_encoder = LanguageEncoder(known_words, embed_dim, bi_dir=True)

        """
        Obj Textual path 
        """
        # ## TODO: add a new path for object and text
        # self.obj_language_encoder = Clip_LanguageEncoder_TransformerFuser(clip_version="ViT-B/32")


        print(
            f"CellRetrievalNetwork, class embed {args.class_embed}, color embed {args.color_embed}, variation: {self.variation}, dim: {embed_dim}, features: {self.use_features}"
        )

        self.printed = False

    def encode_text(self, descriptions):
        batch_size = len(descriptions)
        # print(batch_size)
        # print(descriptions[0])
        # print(descriptions)
        # quit()
        if self.args.language_encoder == "CLIP_text" or self.args.language_encoder == "CLIP_text_transformer":
            description_clip_features = self.language_encoder(descriptions)
            description_encodings = self.language_linear(description_clip_features)
            description_encodings = F.normalize(description_encodings)
        else:
            description_encodings = self.language_encoder(descriptions)  # [B, DIM]
            description_encodings = F.normalize(description_encodings)
        return description_encodings

    def encode_text_objects(self, descriptions):
        description_encodings = self.obj_language_encoder(descriptions)  # [B, DIM]
        description_encodings = F.normalize(description_encodings)
        return description_encodings, None

    def encode_text_submap(self, descriptions):
        batch_size = len(descriptions)
        if self.args.language_encoder == "CLIP_text" or self.args.language_encoder == "CLIP_text_transformer" or self.args.language_encoder == "T5_text_transformer":
            description_clip_features = self.language_encoder(descriptions)
            description_encodings = self.language_linear_submap(description_clip_features)
            description_encodings, description_clip_features = F.normalize(description_encodings), F.normalize(description_clip_features)
            return description_encodings, description_clip_features
        else:
            description_encodings = self.language_encoder(descriptions)  # [B, DIM]
            description_encodings = F.normalize(description_encodings)

        return description_encodings, None

    def encode_objects(self, objects, object_points):
        """
        Process the objects in a flattened way to allow for the processing of batches with uneven sample counts
        """
        batch_size = len(objects)
        class_indices = []
        batch = []  # Batch tensor to send into PyG
        for i_batch, objects_sample in enumerate(objects):
            for obj in objects_sample:
                #  class_idx = self.known_classes.get(obj.label, 0)
                # class_indices.append(class_idx)
                batch.append(i_batch)
        batch = torch.tensor(batch, dtype=torch.long, device=self.device)

        # TODO: Norm embeddings or not?
        # use semantic head or not
        if self.args.use_semantic_head:
            embeddings, class_embedding, color_embedding, pos_embedding, num_points_embedding, relation_embedding, object_features_sem = self.object_encoder(objects, object_points)
        else:
            embeddings, class_embeddings, color_embeddings, pos_embeddings, num_points_embeddings, relation_embedding = self.object_encoder(objects, object_points)
        # print("embeddings", embeddings.shape, "class_embeddings", class_embeddings.shape, "color_embeddings", color_embeddings.shape)
        embeddings = F.normalize(embeddings, dim=-1)  # OPTION: normalize, this is new

        if self.use_edge_conv:
            if self.variation == 0:
                x = self.graph1(embeddings, batch)
                x = gnn.global_max_pool(x, batch)
                x = self.lin(x)
            elif self.variation == 1:
                x = self.graph1(embeddings, batch)
                x = gnn.global_mean_pool(x, batch)
                x = self.lin(x)
        elif (self.only_clip_semantic_feature or self.use_clip_semantic_feature) and not self.use_relation_transformer:
            x = embeddings.to(self.device)
            x = self.attn_pooling(x, batch)
            x = self.final_linear(x)
        elif (self.only_clip_semantic_feature or self.use_clip_semantic_feature) and self.use_relation_transformer:
            x = embeddings.to(self.device)
            x = self.attn_pooling(x, batch, relation_embedding)
            x = self.final_linear(x)
        elif self.use_relation_transformer:
            embeddings = embeddings.to(self.device)
            x = self.attn_pooling(embeddings, batch, relation_embedding)
        else:
            embeddings = embeddings.to(self.device)
            x = self.attn_pooling(embeddings, batch)
        x = F.normalize(x)
        # print("x", x.shape)
        if self.args.use_semantic_head:
            return x, object_features_sem
        else:
            return x

    # def encode_cell(self, cells_xyz, cells_rgb):
    #     """
    #     Process the cell in a flattened way to allow for the processing of batches with uneven sample counts
    #     """
    #     assert len(cells_xyz) == len(cells_rgb)
    #     batch_size = len(cells_xyz)
    #     batched_coords = []
    #     batched_feats = []
    #     for b in range(batch_size):
    #         coords = cells_xyz[b]
    #         coords = torch.tensor(coords, dtype=torch.float)
    #         feats = cells_rgb[b]
    #         feats = torch.tensor(feats, dtype=torch.float)
    #         batched_coords.append(coords)
    #         batched_feats.append(feats)
    #     batched_coords = ME.utils.batched_coordinates(batched_coords)
    #     batched_feats = torch.cat(batched_feats, dim=0)
    #     x_sp = ME.SparseTensor(features=batched_feats, coordinates=batched_coords, device=self.device)
    #     output, feature = self.cell_encoder(x_sp)
    #     output = F.normalize(output.F)
    #     return output

    def encode_cell(self, cell_coordinates, cells_features, target, inverse_map, batched_raw_coordinates=None, batched_raw_color=None, num_query = 24, freeze_cell_encoder=False, use_queries=False, offset=True):
        # print(f"cell_coordinates: {cell_coordinates.shape}, cells_features: {cells_features.shape}")
        raw_coordinates =cells_features[:, -3:]
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
                    target[i]["point2segment"] for i in range(len(target))], raw_coordinates=raw_coordinates, is_eval=False)
        else:
            output_dict = self.cell_encoder(data, point2segment=[
                        target[i]["point2segment"] for i in range(len(target))], raw_coordinates=raw_coordinates, is_eval=False)

        queries = output_dict['queries']  # B, N, 128
        # print(f"queries: {output_dict['queries'].shape}")
        # print(f"pred_logits: {output_dict['pred_logits'].shape}")  # [B, N, Class_num]
        # # argmax on the class_num dimension
        # pred_class = torch.argmax(output_dict['pred_logits'], dim=-1)
        # print(f"pred_logits[0]: {pred_class[0]}")
        # print(f"pred_logits[1]: {pred_class[1]}")
        # # print(f"pred_logits[1]: {output_dict['pred_logits'][1]}")
        # # for i in range(len(output_dict['pred_masks'])):
        # #     print(f"pred_masks: {output_dict['pred_masks'][i].shape}")
        # quit()

        # get pred class
        # pred_class = torch.argmax(output_dict['pred_logits'], dim=-1, keepdim=True)
        # print(f"pred_class shape: {pred_class.shape}")
        # print(f"pred_logits[0]: {pred_class[0]}")
        # print(f"pred_logits[1]: {pred_class[1]}")

        # get instance mask center and instance color and instance
        # TODO: add instance point num
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
            #print(f"gt_class: {gt_class}")
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
                if c<4 or m.sum()==0 or c_label_i==21:
                    continue
                # print(f"inverse_map[i]: {inverse_map[i].shape}")
                mask = m[inverse_map[i]]  # mapping the mask back to the original point scloud
                coordinates_b = batched_raw_coordinates[i]
                coordinates_b = torch.from_numpy(coordinates_b).to(self.device)
                # print(f"coordinates_b: {coordinates_b.shape}")
                coordinates_b_mask = coordinates_b[mask==1]
                mask_queries.append(coordinates_b_mask)
                center_queries.append(torch.mean(coordinates_b_mask, dim=0)/30)
                color_b = batched_raw_color[i]
                # print(f"color_b: {color_b.shape}")
                # print(f"color_b: {color_b}")
                indices = mask==1
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
            assert len(mask_queries) == len(center_queries) == len(color_queries) == len(rgb_queries) == len(pred_class) == len(queries_feature)
            mask_queries_list.append(mask_queries)  # mask 非等长 无法stack
            center_queries_list.append(center_queries)  # center 等长 可以stack
            color_queries_list.append(color_queries)  # color 等长 可以stack
            rgb_queries_list.append(rgb_queries)
            pred_class_list.append(pred_class)
            queries_feature_list.append(queries_feature)
        assert len(mask_queries_list) == len(center_queries_list) == len(color_queries_list) == len(rgb_queries_list) == len(pred_class_list) == len(queries_feature_list)
        # center_queries_list_tensor = torch.stack([torch.stack(center_queries) for center_queries in center_queries_list])
        # color_queries_list_tensor = torch.stack([torch.stack(color_queries) for color_queries in color_queries_list])
        # rgb_queries_list_tensor = torch.stack([torch.stack(rgb_queries) for rgb_queries in rgb_queries_list])
        # pred_class_list_tensor = torch.stack([torch.stack(pred_class) for pred_class in pred_class_list])
        # print("color_queries_tensor shape：", color_queries_list_tensor.shape)
        # print("color_queries_tensor[0]：", color_queries_list_tensor[0])
        # print("center_queries_tensor shape：", center_queries_list_tensor.shape)
        # print("center_queries_tensor[0]：", center_queries_list_tensor[0])
        # print("rgb_queries_tensor shape：", rgb_queries_list_tensor.shape)
        # print(f"rgb_queries_list_tensor[0]: {rgb_queries_list_tensor[0]}")
        # quit()

        batch = []  # Batch tensor to send into PyG

        for i_batch, objects_sample in enumerate(mask_queries_list):
            for obj in objects_sample:
                batch.append(i_batch)
        batch = torch.tensor(batch, dtype=torch.long, device=self.device)

        if self.args.use_queries:
            embeddings, class_embeddings, color_embeddings, pos_embeddings, num_points_embeddings, relation_embedding = self.object_encoder.forward_cell(class_indices=pred_class_list, color_indices=color_queries_list, rgbs=rgb_queries_list, positions=center_queries_list, queries=queries_feature_list)
        else:
            embeddings, class_embeddings, color_embeddings, pos_embeddings, num_points_embeddings, relation_embedding = self.object_encoder.forward_cell(class_indices=pred_class_list, color_indices=color_queries_list, rgbs=rgb_queries_list, positions=center_queries_list)
        # print(embeddings.shape)
        # print(f"embeddings: {embeddings.shape}, class_embeddings: {class_embeddings.shape}, color_embeddings: {color_embeddings.shape}, pos_embeddings: {pos_embeddings.shape}, relation_embedding: {relation_embedding.shape}")
        # quit()

        if self.use_relation_transformer:
            embeddings = embeddings.to(self.device)
            x = self.attn_pooling(embeddings, batch, relation_embedding)
        else:
            embeddings = embeddings.to(self.device)
            x = self.attn_pooling(embeddings, batch)
        # else:
        #     queries = queries.reshape(-1, 128)
        #     embeddings = queries.to(self.device)
        #     x = self.attn_pooling(embeddings, batch)
        #     x = self.final_linear(x)
        x = F.normalize(x)
        # print(f"x shape: {x.shape}")
        # quit()

        return x, output_dict

    def forward(self):
        raise Exception("Not implemented.")

    @property
    def device(self):
        return next(self.lin.parameters()).device

    def get_device(self):
        return next(self.lin.parameters()).device




if __name__ == "__main__":
    from training.args import parse_arguments

    args = parse_arguments()
    print(args)

    model = CellRetrievalNetwork(
        ["high vegetation", "low vegetation", "buildings", "hard scape", "cars"],
        ["green", "brown", "grey", "white", "blue", "red"],
        "a b c d e".split(),
        args
    )


    # dataset = Semantic3dPosesDataset("./data/numpy_merged/", "./data/semantic3d")
    # dataloader = DataLoader(dataset, batch_size=2, collate_fn=Semantic3dPosesDataset.collate_fn)
    # data = dataset[0]
    # batch = next(iter(dataloader))
