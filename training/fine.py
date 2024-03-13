"""Module for training the fine matching module
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torch_geometric.transforms as T

import time
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict
import os
import os.path as osp

from models.superglue_matcher import SuperGlueMatch

from dataloading.kitti360pose.poses import Kitti360FineDataset, Kitti360FineDatasetMulti


# from datapreparation.semantic3d.imports import COLORS as COLORS_S3D, COLOR_NAMES as COLOR_NAMES_S3D
from datapreparation.kitti360pose.utils import (
    COLORS as COLORS_K360,
    COLOR_NAMES as COLOR_NAMES_K360,
    SCENE_NAMES_TEST,
)
from datapreparation.kitti360pose.utils import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL

from training.args import parse_arguments
from training.plots import plot_metrics
from training.losses import MatchingLoss, calc_recall_precision, calc_pose_error, calc_pose_error_no_superglue


def train_epoch(model, dataloader, args):
    model.train()
    stats = EasyDict(
        loss=[],
        loss_offsets=[],
        recall=[],
        precision=[],
        pose_mid=[],
        pose_mean=[],
        pose_offsets=[],
    )

    printed = False
    for i_batch, batch in enumerate(dataloader):
        if args.max_batches is not None and i_batch >= args.max_batches:
            break

        optimizer.zero_grad()
        # print(f"hints: {batch['hint_descriptions']}")
        if not args.no_objects:
            if args.language_encoder == "CLIP_text" or args.language_encoder == "T5":
                output = model.encode_cell(batch["text"],batch["cells_coordinates"], batch["cells_features"], batch["cells_targets"], batch["cells_inverse_map"], batch["cells_points"], batch["cells_colors"], num_query = 24, freeze_cell_encoder=args.freeze_cell_encoder, use_queries=args.use_queries, offset=not args.no_offset_idx)
            else:
                output = model.encode_cell(batch["hint_descriptions"], batch["cells_coordinates"], batch["cells_features"], batch["cells_targets"], batch["cells_inverse_map"], batch["cells_points"], batch["cells_colors"], num_query = 24, freeze_cell_encoder=args.freeze_cell_encoder, use_queries=args.use_queries, offset=not args.no_offset_idx)
        else:
            # TODO: add support for no-objects
            if args.language_encoder == "CLIP_text" or args.language_encoder == "T5":
                output = model(batch["objects"], batch["texts"], batch["object_points"])
            else:
                output = model(batch["objects"], batch["hint_descriptions"], batch["object_points"])

        if args.only_matcher == True:
            loss_matching = criterion_matching(output.P, batch["all_matches"])
            loss_offsets = torch.tensor(-1, dtype=torch.float, device=DEVICE)
            loss = loss_matching

        elif args.no_superglue == True:
            loss_matching = torch.tensor(0, dtype=torch.float, device=DEVICE)
            # print(f"batch[pose.pose]: {batch['poses_coords']}")
            gt_coords = torch.tensor(batch['poses_coords'], dtype=torch.float, device=DEVICE)
            # print(f"gt_coords: {gt_coords.shape}")
            loss_offsets = criterion_offsets(
                output.offsets, gt_coords[:, :2]
            )
            loss = (
                    loss_matching + 5 * loss_offsets
            )  # Currently fixed alpha seems enough, cell normed ∈ [0, 1]

        else:
            loss_matching = criterion_matching(output.P, batch["all_matches"])
            loss_offsets = criterion_offsets(
                output.offsets, torch.tensor(batch["offsets"], dtype=torch.float, device=DEVICE)
            )

            loss = (
                loss_matching + 5 * loss_offsets
            )  # Currently fixed alpha seems enough, cell normed ∈ [0, 1]

        if not printed:
            print(f"Losses: {loss_matching.item():0.3f} {loss_offsets.item():0.3f}")
            printed = True

        try:
            loss.backward()
            optimizer.step()
        except Exception as e:
            print()
            print(str(e))
            print()
            print(batch["all_matches"])
        if not args.no_superglue:
            recall, precision = calc_recall_precision(
                batch["matches"],
                output.matches0.cpu().detach().numpy(),
                output.matches1.cpu().detach().numpy(),
            )
        else:
            recall = precision = -1

        stats.loss.append(loss.item())
        stats.loss_offsets.append(loss_offsets.item())
        stats.recall.append(recall)
        stats.precision.append(precision)
        if not args.no_superglue:
            stats.pose_mid.append(
                calc_pose_error(
                    batch["objects"],
                    output.matches0.detach().cpu().numpy(),
                    batch["poses"],
                    offsets=output.offsets.detach().cpu().numpy(),
                    use_mid_pred=True,
                )
            )
            stats.pose_mean.append(
                calc_pose_error(
                    batch["objects"],
                    output.matches0.detach().cpu().numpy(),
                    batch["poses"],
                    offsets=None,
                )
            )
            stats.pose_offsets.append(
                calc_pose_error(
                    batch["objects"],
                    output.matches0.detach().cpu().numpy(),
                    batch["poses"],
                    offsets=output.offsets.detach().cpu().numpy(),
                )
            )
        else:
            stats.pose_mid.append(
                calc_pose_error_no_superglue(
                    gt_coords = gt_coords[:, :2].detach().cpu().numpy(),
                    pred_coords = output.offsets.detach().cpu().numpy(),
                    use_mid_pred=True,
                )
            )
            stats.pose_mean.append(
                calc_pose_error_no_superglue(
                    gt_coords=gt_coords[:, :2].detach().cpu().numpy(),
                    pred_coords=output.offsets.detach().cpu().numpy(),
                )
            )
            stats.pose_offsets.append(
                calc_pose_error_no_superglue(
                    gt_coords=gt_coords[:, :2].detach().cpu().numpy(),
                    pred_coords=output.offsets.detach().cpu().numpy(),
                )
            )
            # print(f"Pose errors: {stats.pose_mid[-1]:0.3f} {stats.pose_mean[-1]:0.3f} {stats.pose_offsets[-1]:0.3f}")

    for key in stats.keys():
        stats[key] = np.mean(stats[key])
    return stats


@torch.no_grad()
def eval_epoch(model, dataloader, args):
    # model.eval() #TODO/NOTE: set eval() or not?
    print("Evaluating")
    stats = EasyDict(
        recall=[],
        precision=[],
        pose_mid=[],
        pose_mean=[],
        pose_offsets=[],
    )

    for i_batch, batch in enumerate(dataloader):
        if not args.no_objects:
            if args.language_encoder == "CLIP_text" or args.language_encoder == "T5":
                output = model.encode_cell(batch["text"], batch["cells_coordinates"], batch["cells_features"],
                                           batch["cells_targets"], batch["cells_inverse_map"], batch["cells_points"],
                                           batch["cells_colors"], num_query=24,
                                           freeze_cell_encoder=args.freeze_cell_encoder, use_queries=args.use_queries, offset=not args.no_offset_idx)
            else:
                output = model.encode_cell(batch["hint_descriptions"], batch["cells_coordinates"],
                                           batch["cells_features"], batch["cells_targets"], batch["cells_inverse_map"],
                                           batch["cells_points"], batch["cells_colors"], num_query=24,
                                           freeze_cell_encoder=args.freeze_cell_encoder, use_queries=args.use_queries, offset=not args.no_offset_idx)
        else:
            # TODO: add support for no-objects
            if args.language_encoder == "CLIP_text" or args.language_encoder == "T5":
                output = model(batch["objects"], batch["texts"], batch["object_points"])
            else:
                output = model(batch["objects"], batch["hint_descriptions"], batch["object_points"])

        if not args.no_superglue:
            recall, precision = calc_recall_precision(
                batch["matches"],
                output.matches0.cpu().detach().numpy(),
                output.matches1.cpu().detach().numpy(),
            )
            stats.recall.append(recall)
            stats.precision.append(precision)

            stats.pose_mid.append(
                calc_pose_error(
                    batch["objects"],
                    output.matches0.detach().cpu().numpy(),
                    batch["poses"],
                    offsets=output.offsets.detach().cpu().numpy(),
                    use_mid_pred=True,
                )
            )
            stats.pose_mean.append(
                calc_pose_error(
                    batch["objects"],
                    output.matches0.detach().cpu().numpy(),
                    batch["poses"],
                    offsets=None,
                )
            )
            stats.pose_offsets.append(
                calc_pose_error(
                    batch["objects"],
                    output.matches0.detach().cpu().numpy(),
                    batch["poses"],
                    offsets=output.offsets.detach().cpu().numpy(),
                )
            )
        else:
            stats.recall.append(-1)
            stats.precision.append(-1)

            gt_coords = torch.tensor(batch['poses_coords'], dtype=torch.float, device=DEVICE)
            stats.pose_mid.append(
                calc_pose_error_no_superglue(
                    gt_coords = gt_coords[:, :2].detach().cpu().numpy(),
                    pred_coords = output.offsets.detach().cpu().numpy(),
                    use_mid_pred=True,
                )
            )
            stats.pose_mean.append(
                calc_pose_error_no_superglue(
                    gt_coords=gt_coords[:, :2].detach().cpu().numpy(),
                    pred_coords=output.offsets.detach().cpu().numpy(),
                )
            )
            stats.pose_offsets.append(
                calc_pose_error_no_superglue(
                    gt_coords=gt_coords[:, :2].detach().cpu().numpy(),
                    pred_coords=output.offsets.detach().cpu().numpy(),
                )
            )

    for key in stats.keys():
        stats[key] = np.mean(stats[key])
    return stats


@torch.no_grad()
def eval_conf(model, dataset, args):
    accs = []
    accs_old = []
    for i_sample in range(100):
        confs = []

        idx = np.random.randint(len(dataset))
        data = Kitti360FineDataset.collate_fn(
            [
                dataset[idx],
            ]
        )
        if args.language_encoder == "CLIP_text" or args.language_encoder == "T5":
            hints = data["texts"]
        else:
            hints = data["hint_descriptions"]
        output = model(data["objects"], hints, data["object_points"])

        matches = output.matches0.detach().cpu().numpy()
        confs.append(np.sum(matches >= 0))

        for _ in range(4):
            idx = np.random.randint(len(dataset))
            data = Kitti360FineDataset.collate_fn(
                [
                    dataset[idx],
                ]
            )
            if args.language_encoder == "CLIP_text" or args.language_encoder == "T5":
                hints = data["texts"]
            else:
                hints = data["hint_descriptions"]
            output = model(data["objects"], hints, data["object_points"])
            matches = output.matches0.detach().cpu().numpy()
            confs.append(np.sum(matches >= 0))

        accs.append(np.argmax(confs) == 0)
        accs.append(np.argmax(np.flip(confs)) == len(confs) - 1)
        accs_old.append(np.argmax(confs) == 0)

    print("Conf score:", np.mean(accs), np.mean(accs_old))


if __name__ == "__main__":
    args = parse_arguments()
    print(str(args).replace(",", "\n"), "\n")

    dataset_name = args.base_path[:-1] if args.base_path.endswith("/") else args.base_path
    dataset_name = dataset_name.split("/")[-1]
    print(f"Directory: {dataset_name}")

    cont = "Y" if bool(args.continue_path) else "N"
    feats = "all" if len(args.use_features) == 3 else "-".join(args.use_features)
    plot_path = f"./plots/{dataset_name}/Fine_cont{cont}-bs{args.batch_size}_obj-{args.num_mentioned}-{args.pad_size}_e{args.embed_dim}_lr{args.lr_idx}_l{args.num_layers}_i{args.sinkhorn_iters}_v{args.variation}_ecl{int(args.class_embed)}_eco{int(args.color_embed)}_p{args.pointnet_numpoints}_s{args.shuffle}_g{args.lr_gamma}_npa{int(args.no_pc_augment)}_nca{int(args.no_cell_augment)}_f-{feats}.png"
    print("Plot:", plot_path, "\n")

    """
    Create data loaders
    """
    if args.dataset == "K360":
        if args.no_pc_augment:
            train_transform = T.FixedPoints(args.pointnet_numpoints)
            val_transform = T.FixedPoints(args.pointnet_numpoints)
        else:
            train_transform = T.Compose(
                [
                    T.FixedPoints(args.pointnet_numpoints),
                    T.RandomRotate(120, axis=2),
                    T.NormalizeScale(),
                ]
            )
            val_transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])

        dataset_train = Kitti360FineDatasetMulti(
            args.base_path, SCENE_NAMES_TRAIN, train_transform, args, flip_pose=False
        )  # No cell-augment for fine
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            collate_fn=Kitti360FineDataset.collate_fn,
            shuffle=args.shuffle,
        )

        dataset_val = Kitti360FineDatasetMulti(args.base_path, SCENE_NAMES_VAL, val_transform, args)
        dataloader_val = DataLoader(
            dataset_val, batch_size=args.batch_size, collate_fn=Kitti360FineDataset.collate_fn
        )

    elif args.dataset == "K360_cell":
        from dataloading.kitti360pose.poses_sp import Kitti360FineDataset, Kitti360FineDatasetMulti
        if args.no_pc_augment:
            train_transform = T.FixedPoints(args.pointnet_numpoints)
            val_transform = T.FixedPoints(args.pointnet_numpoints)
        else:
            train_transform = T.Compose(
                [
                    T.FixedPoints(args.pointnet_numpoints),
                    T.RandomRotate(120, axis=2),
                    T.NormalizeScale(),
                ]
            )
            val_transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])

        dataset_train = Kitti360FineDatasetMulti(
            args.base_path, SCENE_NAMES_TRAIN, train_transform, args, flip_pose=False
        )  # No cell-augment for fine
        if args.no_offset_idx:
            dataloader_train = DataLoader(
                dataset_train,
                batch_size=args.batch_size,
                collate_fn=Kitti360FineDataset.collate_fn_cell_no_offset,
                shuffle=args.shuffle,
            )
        else:
            dataloader_train = DataLoader(
                dataset_train,
                batch_size=args.batch_size,
                collate_fn=Kitti360FineDataset.collate_fn_cell,
                shuffle=args.shuffle,
            )

        dataset_val = Kitti360FineDatasetMulti(args.base_path, SCENE_NAMES_VAL, val_transform, args)
        if args.no_offset_idx:
            dataloader_val = DataLoader(
                dataset_val, batch_size=args.batch_size, collate_fn=Kitti360FineDataset.collate_fn_cell_no_offset
            )
        else:
            dataloader_val = DataLoader(
                dataset_val, batch_size=args.batch_size, collate_fn=Kitti360FineDataset.collate_fn_cell
            )


    print(sorted(dataset_train.get_known_words()))
    print(sorted(dataset_val.get_known_words()))
    assert sorted(dataset_train.get_known_classes()) == sorted(dataset_val.get_known_classes())

    # TODO: turn back on for multi
    # assert sorted(dataset_train.get_known_classes()) == sorted(dataset_val.get_known_classes()) and sorted(dataset_train.get_known_words()) == sorted(dataset_val.get_known_words())

    data0 = dataset_train[0]
    batch = next(iter(dataloader_train))

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", DEVICE, torch.cuda.get_device_name(0))
    torch.autograd.set_detect_anomaly(True)

    best_val_recallPrecision = -1  # Measured by mean of recall and precision
    last_model_save_path = None

    """
    Start training
    """
    if args.lr_idx is not None:
        learning_rates = np.logspace(-3.0, -4.0, 3)[
            args.lr_idx : args.lr_idx + 1
        ]  # Larger than -3 throws error (even with warm-up)
    else:
        learning_rates = [
            args.learning_rate,
        ]

    train_stats_loss = {lr: [] for lr in learning_rates}
    train_stats_loss_offsets = {lr: [] for lr in learning_rates}
    train_stats_recall = {lr: [] for lr in learning_rates}
    train_stats_precision = {lr: [] for lr in learning_rates}
    train_stats_pose_mid = {lr: [] for lr in learning_rates}
    train_stats_pose_mean = {lr: [] for lr in learning_rates}
    train_stats_pose_offsets = {lr: [] for lr in learning_rates}

    val_stats_recall = {lr: [] for lr in learning_rates}
    val_stats_precision = {lr: [] for lr in learning_rates}
    val_stats_pose_mid = {lr: [] for lr in learning_rates}
    val_stats_pose_mean = {lr: [] for lr in learning_rates}
    val_stats_pose_offsets = {lr: [] for lr in learning_rates}

    for lr in learning_rates:
        if bool(args.continue_path):
            model = torch.load(args.continue_path)
        else:
            model = SuperGlueMatch(
                dataset_train.get_known_classes(),
                COLOR_NAMES_K360,
                dataset_train.get_known_words(),
                args,
            )
        model.to(DEVICE)

        criterion_matching = MatchingLoss()
        criterion_offsets = nn.MSELoss()
        criterion_class = nn.CrossEntropyLoss()
        criterion_color = nn.CrossEntropyLoss()

        # Warm-up
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)
        if args.no_objects:
            best_val_recallPrecision = 99
        else:
            best_val_recallPrecision = -1
        for epoch in range(args.epochs):
            if epoch == 3:
                optimizer = optim.Adam(model.parameters(), lr=lr)
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)

            # test val
            # val_out = eval_epoch(model, dataloader_val, args)  # CARE: which loader for val!
            # val_stats_recall[lr].append(val_out.recall)
            # val_stats_precision[lr].append(val_out.precision)
            # val_stats_pose_mid[lr].append(val_out.pose_mid)
            # val_stats_pose_mean[lr].append(val_out.pose_mean)
            # val_stats_pose_offsets[lr].append(val_out.pose_offsets)
            # print(
            #     (
            #
            #         f"v-recall {val_out.recall:0.2f} v-precision {val_out.precision:0.2f} v-mean {val_out.pose_mean:0.2f} v-offset {val_out.pose_offsets:0.2f} "
            #     ),
            #     flush=True,
            # )
            # # test val
            # quit()

            train_out = train_epoch(model, dataloader_train, args)

            train_stats_loss[lr].append(train_out.loss)
            train_stats_loss_offsets[lr].append(train_out.loss_offsets)
            train_stats_recall[lr].append(train_out.recall)
            train_stats_precision[lr].append(train_out.precision)
            train_stats_pose_mid[lr].append(train_out.pose_mid)
            train_stats_pose_mean[lr].append(train_out.pose_mean)
            train_stats_pose_offsets[lr].append(train_out.pose_offsets)

            val_out = eval_epoch(model, dataloader_val, args)  # CARE: which loader for val!
            val_stats_recall[lr].append(val_out.recall)
            val_stats_precision[lr].append(val_out.precision)
            val_stats_pose_mid[lr].append(val_out.pose_mid)
            val_stats_pose_mean[lr].append(val_out.pose_mean)
            val_stats_pose_offsets[lr].append(val_out.pose_offsets)

            if not args.no_superglue:
                print()
                eval_conf(model, dataset_val, args)
                print()

            if scheduler:
                scheduler.step()

            print(
                (
                    f"\t lr {lr:0.6} epoch {epoch} loss {train_out.loss:0.3f} "
                    f"t-recall {train_out.recall:0.2f} t-precision {train_out.precision:0.2f} t-mean {train_out.pose_mean:0.2f} t-offset {train_out.pose_offsets:0.2f} "
                    f"v-recall {val_out.recall:0.2f} v-precision {val_out.precision:0.2f} v-mean {val_out.pose_mean:0.2f} v-offset {val_out.pose_offsets:0.2f} "
                ),
                flush=True,
            )

            if epoch >= 0:
                if args.no_objects:
                    acc = np.mean((val_out.pose_offsets, val_out.pose_mean))
                    if acc < best_val_recallPrecision:
                        model_path = f"./checkpoints/{dataset_name}/fine_cont{cont}_acc{acc:0.2f}_lr{args.lr_idx}_obj-{args.num_mentioned}-{args.pad_size}_ecl{int(args.class_embed)}_eco{int(args.color_embed)}_p{args.pointnet_numpoints}_npa{int(args.no_pc_augment)}_nca{int(args.no_cell_augment)}_f-{feats}_no_obj_{args.no_objects}.pth"
                        if not osp.isdir(osp.dirname(model_path)):
                            os.mkdir(osp.dirname(model_path))

                        print("Saving model to", model_path)
                        try:
                            # torch.save(model, model_path)
                            torch.save(model.state_dict(), model_path)
                            if (
                                    last_model_save_path is not None
                                    and last_model_save_path != model_path
                                    and osp.isfile(last_model_save_path)
                            ):
                                print("Removing", last_model_save_path)
                                os.remove(last_model_save_path)
                            last_model_save_path = model_path
                        except Exception as e:
                            print("Error saving model!", str(e))
                        best_val_recallPrecision = acc
                else:
                    acc = np.mean((val_out.recall, val_out.precision))
                    if acc > best_val_recallPrecision:
                        model_path = f"./checkpoints/{dataset_name}/fine_cont{cont}_acc{acc:0.2f}_lr{args.lr_idx}_obj-{args.num_mentioned}-{args.pad_size}_ecl{int(args.class_embed)}_eco{int(args.color_embed)}_p{args.pointnet_numpoints}_npa{int(args.no_pc_augment)}_nca{int(args.no_cell_augment)}_f-{feats}.pth"
                        if not osp.isdir(osp.dirname(model_path)):
                            os.mkdir(osp.dirname(model_path))

                        print("Saving model to", model_path)
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
                            print("Error saving model!", str(e))
                        best_val_recallPrecision = acc

        print()

    """
    Save plots
    """
    metrics = {
        "train-loss": train_stats_loss,
        "train-loss_offsets": train_stats_loss_offsets,
        "train-recall": train_stats_recall,
        "train-precision": train_stats_precision,
        "train-pose_mid": train_stats_pose_mid,
        "train-pose_mean": train_stats_pose_mean,
        "train-pose_offsets": train_stats_pose_offsets,
        "val-recall": val_stats_recall,
        "val-precision": val_stats_precision,
        "val-pose_mid": val_stats_pose_mid,
        "val-pose_mean": val_stats_pose_mean,
        "val-pose_offsets": val_stats_pose_offsets,
    }
    if not osp.isdir(osp.dirname(plot_path)):
        os.mkdir(osp.dirname(plot_path))
    plot_metrics(metrics, plot_path)
