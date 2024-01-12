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

from models.modules import get_mlp, LanguageEncoder, Clip_LanguageEncoder, MaxPoolMultiHeadSelfAttention, Clip_LanguageEncoder_TransformerFuser, MaxPoolRelationMultiHeadSelfAttention

from models.object_encoder import ObjectEncoder
import clip


#from dataloading.semantic3d.semantic3d import Semantic3dCellRetrievalDataset
#from dataloading.semantic3d.semantic3d_poses import Semantic3dPosesDataset


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
        elif args.only_clip_semantic_feature:
            self.mlp_pos_num = get_mlp([2 * embed_dim, embed_dim])
            self.attn_pooling = MaxPoolMultiHeadSelfAttention(embed_dim, num_heads=8)
        elif args.use_relation_transformer:
            self.attn_pooling = MaxPoolRelationMultiHeadSelfAttention(embed_dim, num_heads=8)
        else:  # use attention + pooling
            self.attn_pooling = MaxPoolMultiHeadSelfAttention(embed_dim, num_heads=8)

        self.object_encoder = ObjectEncoder(embed_dim, known_classes, known_colors, args)


        """
        Textual path
        """
        if args.language_encoder == "CLIP_text":
            self.language_encoder = Clip_LanguageEncoder(clip_version="ViT-B/32")
            self.language_linear = nn.Linear(512, embed_dim)
            # self.language_linear_object = nn.Linear(512, embed_dim)
            self.language_linear_submap = nn.Linear(512, embed_dim)

        elif args.language_encoder == "CLIP_text_transformer":
            self.language_encoder = Clip_LanguageEncoder_TransformerFuser(clip_version="ViT-B/32")
            self.language_linear = nn.Linear(512, embed_dim)
            # self.language_linear_object = nn.Linear(512, embed_dim)
            self.language_linear_submap = nn.Linear(512, embed_dim)

        else:
            self.language_encoder = LanguageEncoder(known_words, embed_dim, bi_dir=True)

        """
        Obj Textual path 
        """
        ## TODO: add a new path for object and text
        self.obj_language_encoder = Clip_LanguageEncoder_TransformerFuser(clip_version="ViT-B/32")


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
        if self.args.language_encoder == "CLIP_text" or self.args.language_encoder == "CLIP_text_transformer":
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
                # class_idx = self.known_classes.get(obj.label, 0)
                # class_indices.append(class_idx)
                batch.append(i_batch)
        batch = torch.tensor(batch, dtype=torch.long, device=self.device)
        # TODO: Norm embeddings or not?
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
        elif self.only_clip_semantic_feature:
            x = [pos_embeddings, num_points_embeddings]
            x = torch.cat(x, dim=-1)  # [B, 2*DIM]
            x = self.mlp_pos_num(x.to(self.device))
            x = self.attn_pooling(x, batch)
        elif self.use_relation_transformer:
            embeddings = embeddings.to(self.device)
            x = self.attn_pooling(embeddings, batch, relation_embedding)
        else:
            embeddings = embeddings.to(self.device)
            x = self.attn_pooling(embeddings, batch)
        x = F.normalize(x)
        # print("x", x.shape)
        return x

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
