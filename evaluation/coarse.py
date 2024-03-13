from datapreparation.kitti360pose.imports import Object3d
import numpy as np
import os
import os.path as osp
import cv2
from easydict import EasyDict
from copy import deepcopy
import pickle

import torch
from torch.utils.data import DataLoader
import time

from scipy.spatial.distance import cdist

# from training.losses import calc_pose_error # This cannot be used here anymore!!
from evaluation.args import parse_arguments
from evaluation.utils import calc_sample_accuracies, print_accuracies

from dataloading.kitti360pose.cells import Kitti360CoarseDataset, Kitti360CoarseDatasetMulti
from dataloading.kitti360pose.eval import Kitti360TopKDataset

from datapreparation.kitti360pose.utils import SCENE_NAMES_TEST, SCENE_NAMES_VAL

from training.coarse import eval_epoch as eval_epoch_retrieval
from training.utils import plot_retrievals
from models.superglue_matcher import get_pos_in_cell

import torch_geometric.transforms as T

"""
TODO:
- Try to add num_matches*10 + sum(match_scores[correctly_matched])
"""


@torch.no_grad()
def run_coarse(model, dataloader, args):
    """Run text-to-cell retrieval to obtain the top-cells and coarse pose accuracies.

    Args:
        model: retrieval model
        dataloader: retrieval dataset
        args: global arguments

    Returns:
        [List]: retrievals as [(cell_indices_i_0, cell_indices_i_1, ...), (cell_indices_i+1, ...), ...] with i ∈ [0, len(poses)-1], j ∈ [0, max(top_k)-1]
        [Dict]: accuracies
    """
    # model.eval()

    all_cells_dict = {cell.id: cell for cell in dataloader.dataset.all_cells}

    if (
        args.coarse_oracle
    ):  # Build retrievals as [[best_cell, best_cell, ...], [best_cell, best_cell, ...]]
        retrievals = []
        max_k = max(args.top_k)
        for pose in dataloader.dataset.all_poses:
            retrievals.append([pose.cell_id for i in range(max_k)])
    elif args.coarse_random:
        retrievals = []
        max_k = max(args.top_k)
        all_cell_ids = list(all_cells_dict.keys())
        num_poses = len(dataloader.dataset.all_poses)
        retrievals = np.random.choice(all_cell_ids, size=(num_poses, max_k))
        retrievals = retrievals.tolist()
    elif args.street_oracle:
        # Calculate scores here, remove retrieved cells from incorrect streets
        # TODO: Verify this leads to same results
        assert args.use_test_set == False

        _, _, _, cell_encodings, text_encodings = eval_epoch_retrieval(
            model, dataloader, args, return_encodings=True
        )

        with open(
            osp.join(args.base_path, "street_centers", "2013_05_28_drive_0010_sync.pkl"), "rb"
        ) as f:
            street_centers = pickle.load(f)

        cells = dataloader.dataset.all_cells
        poses = dataloader.dataset.all_poses
        assert len(cells) == len(cell_encodings) and len(poses) == len(text_encodings)

        cell_centers = np.array([cell.get_center() for cell in cells])
        cell_street_dists = cdist(cell_centers, street_centers)  # [num_cells, num_streets]
        cell_street_indices = np.argmin(cell_street_dists, axis=1)
        assert len(cell_street_indices) == len(cell_centers)
        cell_ids = np.array([cell.id for cell in cells])

        retrievals = []  # [[cell_ids0], [cell_ids1], ...]
        for query_idx in range(len(text_encodings)):
            pose = poses[query_idx]
            scores = (
                cell_encodings[:] @ text_encodings[query_idx]
            )  # NOTE: Always uses cosine-similarity here, not correct for other losses

            # Remove retrievals from incorrect streets
            pose_street_dist = np.linalg.norm(street_centers - pose.pose_w, axis=1)
            pose_street_idx = np.argmin(pose_street_dist)
            scores[cell_street_indices != pose_street_idx] = -np.inf

            sorted_indices = np.argsort(-1.0 * scores)  # High -> low
            sorted_indices = sorted_indices[0 : np.max(args.top_k)]

            retrieved_cell_ids = cell_ids[sorted_indices]
            retrievals.append(retrieved_cell_ids)
    else:
        # Run retrieval model to obtain top-cells
        retrieval_accuracies, retrieval_accuracies_close, retrievals = eval_epoch_retrieval(
            model, dataloader, args
        )
        retrievals = [retrievals[idx] for idx in range(len(retrievals))]  # Dict -> list
        print("Retrieval Accs:")
        print(retrieval_accuracies)
        print("Retrieval Accs Close:")
        print(retrieval_accuracies_close)
        assert len(retrievals) == len(dataloader.dataset.all_poses)

    # Gather the accuracies for each sample
    accuracies = {k: {t: [] for t in args.threshs} for k in args.top_k}
    for i_sample in range(len(retrievals)):
        pose = dataloader.dataset.all_poses[i_sample]
        top_cells = [all_cells_dict[cell_id] for cell_id in retrievals[i_sample]]
        pos_in_cells = 0.5 * np.ones((len(top_cells), 2))  # Predict cell-centers
        accs, wrong_cell_pairs = calc_sample_accuracies(pose, top_cells, pos_in_cells, args.top_k, args.threshs)
        print(f"wrong_cell_name: {wrong_cell_pairs}")
        print(f"gt_pose.cell_id: {pose.cell_id}")
        print(f"pred_cell_ids: {retrievals[i_sample]}")
        for k in args.top_k:
            for t in args.threshs:
                accuracies[k][t].append(accs[k][t])

    for k in args.top_k:
        for t in args.threshs:
            accuracies[k][t] = np.mean(accuracies[k][t])

    return retrievals, accuracies

if __name__ == "__main__":
    args = parse_arguments()
    print(str(args).replace(",", "\n"), "\n")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device, torch.cuda.get_device_name(0))

    # Load datasets
    if args.no_pc_augment:
        transform = T.FixedPoints(args.pointnet_numpoints)
    else:
        transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])

    if args.use_test_set:
        dataset_retrieval = Kitti360CoarseDatasetMulti(
            args.base_path, SCENE_NAMES_TEST, transform, shuffle_hints=False, flip_poses=False, use_alt_descriptions=args.use_alt_descriptions
        )
    else:
        dataset_retrieval = Kitti360CoarseDatasetMulti(
            args.base_path, SCENE_NAMES_VAL, transform, shuffle_hints=False, flip_poses=False, use_alt_descriptions=args.use_alt_descriptions
        )
    dataloader_retrieval = DataLoader(
        dataset_retrieval,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )

    # dataset_cell_only = dataset_retrieval.get_cell_dataset()

    # Load models
    model_retrieval = torch.load(args.path_coarse, map_location=torch.device("cpu"))
    model_retrieval.to(device)


    # Run coarse
    retrievals, coarse_accuracies = run_coarse(model_retrieval, dataloader_retrieval, args)
    print_accuracies(coarse_accuracies, "Coarse")

    if args.plot_retrievals:
        plot_retrievals(retrievals, dataset_retrieval)
    if args.plot_retrievals or args.coarse_only:
        quit()
