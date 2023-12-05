"""Module for training the coarse cell-retrieval module
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torch_geometric.transforms as T

import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from easydict import EasyDict
import os
import os.path as osp

from models.cell_retrieval import CellRetrievalNetwork

from datapreparation.kitti360pose.utils import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL
from datapreparation.kitti360pose.utils import COLOR_NAMES as COLOR_NAMES_K360
from dataloading.kitti360pose.cells import Kitti360CoarseDatasetMulti, Kitti360CoarseDataset

from training.args import parse_arguments
from training.plots import plot_metrics
from training.losses import MatchingLoss, PairwiseRankingLoss, HardestRankingLoss, ClipLoss
from training.utils import plot_retrievals


def train_epoch(model, dataloader, args):
    model.train()
    epoch_losses = []

    batches = []
    printed = False
    # time_start_epoch = time.time()
    for i_batch, batch in enumerate(dataloader):
        # time_end_load_batch = time.time()
        # print(f"load batch {i_batch} in {time_end_load_batch - time_start_epoch:0.2f}")

        # time_start_forward = time.time()
        if args.max_batches is not None and i_batch >= args.max_batches:
            break
        # print(f"\r{i_batch}/{len(dataloader)}", end="", flush=True)
        batch_size = len(batch["texts"])
        optimizer.zero_grad()
        anchor = model.encode_text(batch["texts"])
        anchor_objects = model.encode_text_objects(batch["texts_objects"])
        anchor_submap = model.encode_text_submap(batch["texts_submap"])
        positive = model.encode_objects(batch["objects"], batch["object_points"])
        # print("anchor_object: ", anchor_objects.shape)
        # print("anchor_submap: ", anchor_submap.shape)
        # print("anchor", anchor.shape)
        # print("positive", positive.shape)
        # quit()
        # time_end_forward = time.time()
        # print(f"forward {i_batch} in {time_end_forward - time_start_forward:0.2f}")

        if args.ranking_loss == "triplet":
            negative_cell_objects = [cell.objects for cell in batch["negative_cells"]]
            negative = model.encode_objects(negative_cell_objects)
            loss = criterion(anchor, positive, negative)

        elif args.ranking_loss == "obj_submap":
            loss_obj = criterion(anchor_objects, positive)
            loss_submap = criterion(anchor_submap, positive)

        else:
            loss = criterion(anchor, positive)

        loss = loss

        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        batches.append(batch)

        if i_batch % 80 == 0:
            print(f"\r{i_batch}/{len(dataloader)} loss {loss.item():0.2f}", end="", flush=True)
    return np.mean(epoch_losses), batches


@torch.no_grad()
def eval_epoch(model, dataloader, args, return_encodings=False):
    """Top-k retrieval for each pose against all cells in the dataset.

    Args:
        model: The model
        dataloader: Train or test dataloader
        args: Global arguments

    Returns:
        Dict: Top-k accuracies
        Dict: Top retrievals as {query_idx: top_cell_ids}
    """
    assert args.ranking_loss != "triplet"  # Else also update evaluation.pipeline

    model.eval()  # Now eval() seems to increase results
    accuracies = {k: [] for k in args.top_k}
    accuracies_close = {k: [] for k in args.top_k}

    # TODO: Use this again if batches!
    # num_samples = len(dataloader.dataset) if isinstance(dataloader, DataLoader) else np.sum([len(batch['texts']) for batch in dataloader])

    cells_dataset = dataloader.dataset.get_cell_dataset()
    cells_dataloader = DataLoader(
        cells_dataset,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )
    cells_dict = {cell.id: cell for cell in cells_dataset.cells}
    cell_size = cells_dataset.cells[0].cell_size

    cell_encodings = np.zeros((len(cells_dataset), model.embed_dim))
    db_cell_ids = np.zeros(len(cells_dataset), dtype="<U32")

    text_encodings = np.zeros((len(dataloader.dataset), model.embed_dim))
    query_cell_ids = np.zeros(len(dataloader.dataset), dtype="<U32")
    query_poses_w = np.array([pose.pose_w[0:2] for pose in dataloader.dataset.all_poses])

    # Encode the query side
    # t0 = time.time()
    index_offset = 0
    for batch in dataloader:
        #print(f"eval: \r{index_offset}/{len(dataloader.dataset)}", end="", flush=True)
        text_enc = model.encode_text(batch["texts"])
        batch_size = len(text_enc)

        text_encodings[index_offset : index_offset + batch_size, :] = (
            text_enc.cpu().detach().numpy()
        )
        query_cell_ids[index_offset : index_offset + batch_size] = np.array(batch["cell_ids"])
        index_offset += batch_size
    # print(f"Encoded {len(text_encodings)} query texts in {time.time() - t0:0.2f}.")

    # Encode the database side
    index_offset = 0
    for batch in cells_dataloader:
        cell_enc = model.encode_objects(batch["objects"], batch["object_points"])
        batch_size = len(cell_enc)

        cell_encodings[index_offset : index_offset + batch_size, :] = (
            cell_enc.cpu().detach().numpy()
        )
        db_cell_ids[index_offset : index_offset + batch_size] = np.array(batch["cell_ids"])
        index_offset += batch_size

    top_retrievals = {}  # {query_idx: top_cell_ids}
    for query_idx in range(len(text_encodings)):
        if args.ranking_loss != "triplet":  # Asserted above
            scores = cell_encodings[:] @ text_encodings[query_idx]
            assert len(scores) == len(dataloader.dataset.all_cells)  # TODO: remove
            sorted_indices = np.argsort(-1.0 * scores)  # High -> low

        sorted_indices = sorted_indices[0 : np.max(args.top_k)]

        # Best-cell hit accuracy
        retrieved_cell_ids = db_cell_ids[sorted_indices]
        target_cell_id = query_cell_ids[query_idx]

        for k in args.top_k:
            accuracies[k].append(target_cell_id in retrieved_cell_ids[0:k])
        top_retrievals[query_idx] = retrieved_cell_ids

        # Close-by accuracy
        # CARE/TODO: can be wrong across scenes!
        target_pose_w = query_poses_w[query_idx]
        retrieved_cell_poses = [
            cells_dict[cell_id].get_center()[0:2] for cell_id in retrieved_cell_ids
        ]
        dists = np.linalg.norm(target_pose_w - retrieved_cell_poses, axis=1)
        for k in args.top_k:
            accuracies_close[k].append(np.any(dists[0:k] <= cell_size / 2))

    for k in args.top_k:
        accuracies[k] = np.mean(accuracies[k])
        accuracies_close[k] = np.mean(accuracies_close[k])

    if return_encodings:
        return accuracies, accuracies_close, top_retrievals, cell_encodings, text_encodings
    else:
        return accuracies, accuracies_close, top_retrievals


if __name__ == "__main__":
    args = parse_arguments()
    print(str(args).replace(",", "\n"), "\n")

    dataset_name = args.base_path[:-1] if args.base_path.endswith("/") else args.base_path
    dataset_name = dataset_name.split("/")[-1]
    print(f"Directory: {dataset_name}")

    cont = "Y" if bool(args.continue_path) else "N"
    feats = "all" if len(args.use_features) == 3 else "-".join(args.use_features)
    plot_path = f"./plots/{dataset_name}/Coarse_cont{cont}_bs{args.batch_size}_lr{args.lr_idx}_e{args.embed_dim}_ecl{int(args.class_embed)}_eco{int(args.color_embed)}_p{args.pointnet_numpoints}_m{args.margin:0.2f}_s{int(args.shuffle)}_g{args.lr_gamma}_npa{int(args.no_pc_augment)}_nca{int(args.no_cell_augment)}_f-{feats}.png"
    print("Plot:", plot_path, "\n")

    # time_start_create_data_loader = time.time()
    """
    Create data loaders
    """
    if args.dataset == "K360":
        # ['2013_05_28_drive_0003_sync', ]
        if args.no_pc_augment:
            # train_transform = T.FixedPoints(args.pointnet_numpoints)
            # val_transform = T.FixedPoints(args.pointnet_numpoints)
            train_transform = T.Compose(
                [
                    T.FixedPoints(args.pointnet_numpoints),
                    # T.RandomRotate(120, axis=2),
                    T.NormalizeScale(),
                ]
            )
            val_transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])
        else:
            train_transform = T.Compose(
                [
                    T.FixedPoints(args.pointnet_numpoints),
                    T.RandomRotate(120, axis=2),
                    T.NormalizeScale(),
                ]
            )
            val_transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])

        if args.no_cell_augment:
            dataset_train = Kitti360CoarseDatasetMulti(
                args.base_path,
                SCENE_NAMES_TRAIN,
                train_transform,
                shuffle_hints=False,
                flip_poses=False,
            )
        else:
            dataset_train = Kitti360CoarseDatasetMulti(
                args.base_path,
                SCENE_NAMES_TRAIN,
                train_transform,
                shuffle_hints=True,
                flip_poses=True,
            )
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            collate_fn=Kitti360CoarseDataset.collate_fn,
            shuffle=args.shuffle,
            pin_memory=True
        )

        dataset_val = Kitti360CoarseDatasetMulti(args.base_path, SCENE_NAMES_VAL, val_transform)
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            collate_fn=Kitti360CoarseDataset.collate_fn,
            shuffle=False,
            pin_memory=True
        )

    print(
        "Words-diff:",
        set(dataset_train.get_known_words()).difference(set(dataset_val.get_known_words())),
    )
    assert sorted(dataset_train.get_known_classes()) == sorted(dataset_val.get_known_classes())

    # time_end_create_data_loader = time.time()
    # print(f"Created data loaders in {time_end_create_data_loader - time_start_create_data_loader:0.2f}.")

    data = dataset_train[0]
    assert len(data["debug_hint_descriptions"]) == args.num_mentioned

    batch = next(iter(dataloader_train))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device, torch.cuda.get_device_name(0))
    torch.autograd.set_detect_anomaly(True)

    if args.lr_idx is not None:
        learning_rates = np.logspace(-2.5, -3.5, 3)[args.lr_idx : args.lr_idx + 1]
    else:
        learning_rates = [
            args.learning_rate,
        ]
    dict_loss = {lr: [] for lr in learning_rates}
    dict_acc = {k: {lr: [] for lr in learning_rates} for k in args.top_k}
    dict_acc_val = {k: {lr: [] for lr in learning_rates} for k in args.top_k}
    dict_acc_val_close = {k: {lr: [] for lr in learning_rates} for k in args.top_k}

    best_val_accuracy = -1
    last_model_save_path = None

    for lr in learning_rates:
        if args.continue_path:
            model = torch.load(args.continue_path)
        else:
            model = CellRetrievalNetwork(
                dataset_train.get_known_classes(),
                COLOR_NAMES_K360,
                dataset_train.get_known_words(),
                args,
            )
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        if args.ranking_loss == "pairwise":
            criterion = PairwiseRankingLoss(margin=args.margin)
        if args.ranking_loss == "hardest":
            criterion = HardestRankingLoss(margin=args.margin)
        if args.ranking_loss == "triplet":
            criterion = nn.TripletMarginLoss(margin=args.margin)
        if args.ranking_loss == "CLIP":
            criterion = ClipLoss()

        criterion_class = nn.CrossEntropyLoss()
        criterion_color = nn.CrossEntropyLoss()

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)

        for epoch in range(1, args.epochs + 1):
            # dataset_train.reset_seed() #OPTION: re-setting seed leads to equal data at every epoch
            time_start_epoch = time.time()
            loss, train_batches = train_epoch(model, dataloader_train, args)
            # train_acc, train_retrievals = eval_epoch(model, train_batches, args)
            time_end_epoch = time.time()
            print(f"Epoch {epoch} in {time_end_epoch - time_start_epoch:0.2f}.")

            time_start_eval = time.time()
            train_acc, train_acc_close, train_retrievals = eval_epoch(
                model, dataloader_train, args
            )  # TODO/CARE: Is this ok? Send in batches again?
            val_acc, val_acc_close, val_retrievals = eval_epoch(model, dataloader_val, args)
            time_end_eval = time.time()
            print(f"Evaluation in {time_end_eval - time_start_eval:0.2f}.")

            key = lr
            dict_loss[key].append(loss)
            for k in args.top_k:
                dict_acc[k][key].append(train_acc[k])
                dict_acc_val[k][key].append(val_acc[k])
                dict_acc_val_close[k][key].append(val_acc_close[k])

            scheduler.step()
            print(f"\t lr {lr:0.4} loss {loss:0.2f} epoch {epoch} train-acc: ", end="")
            for k, v in train_acc.items():
                print(f"{k}-{v:0.2f} ", end="")
            print("val-acc: ", end="")
            for k, v in val_acc.items():
                print(f"{k}-{v:0.2f} ", end="")
            print("val-acc-close: ", end="")
            for k, v in val_acc_close.items():
                print(f"{k}-{v:0.2f} ", end="")
            print("\n", flush=True)

            # Saving best model (w/ early stopping)
            if epoch >= args.epochs // 4:
                acc = val_acc[max(args.top_k)]
                if acc > best_val_accuracy:
                    model_path = f"./checkpoints/{dataset_name}/coarse_cont{cont}_acc{acc:0.2f}_lr{args.lr_idx}_ecl{int(args.class_embed)}_eco{int(args.color_embed)}_p{args.pointnet_numpoints}_npa{int(args.no_pc_augment)}_nca{int(args.no_cell_augment)}_f-{feats}.pth"
                    if not osp.isdir(osp.dirname(model_path)):
                        os.mkdir(osp.dirname(model_path))

                    print(f"Saving model at {acc:0.2f} to {model_path}")
                    try:
                        torch.save(model, model_path)
                        if (
                            last_model_save_path is not None
                            and last_model_save_path != model_path
                            and osp.isfile(last_model_save_path)
                        ):
                            print("Removing", last_model_save_path)
                            os.remove(last_model_save_path)
                        last_model_save_path = model_path
                    except Exception as e:
                        print(f"Error saving model!", str(e))
                    best_val_accuracy = acc

    """
    Save plots
    """
    # plot_name = f'Cells-{args.dataset}_s{scene_name.split('_')[-2]}_bs{args.batch_size}_mb{args.max_batches}_e{args.embed_dim}_l-{args.ranking_loss}_m{args.margin}_f{"-".join(args.use_features)}.png'
    train_accs = {f"train-acc-{k}": dict_acc[k] for k in args.top_k}
    val_accs = {f"val-acc-{k}": dict_acc_val[k] for k in args.top_k}
    val_accs_close = {f"val-close-{k}": dict_acc_val_close[k] for k in args.top_k}

    metrics = {
        "train-loss": dict_loss,
        **train_accs,
        **val_accs,
        **val_accs_close,
    }
    if not osp.isdir(osp.dirname(plot_path)):
        os.mkdir(osp.dirname(plot_path))
    plot_metrics(metrics, plot_path)
