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
import torch.nn.functional as F

from models.cell_retrieval import CellRetrievalNetwork

from datapreparation.kitti360pose.utils import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL, CLASS_TO_INDEX
from datapreparation.kitti360pose.utils import COLOR_NAMES as COLOR_NAMES_K360
from dataloading.kitti360pose.cells import Kitti360CoarseDataset, Kitti360CoarseDatasetMulti

from training.args import parse_arguments
from training.plots import plot_metrics
from training.losses import MatchingLoss, PairwiseRankingLoss, HardestRankingLoss, ClipLoss, SemanticLoss

from training.criterion import SetCriterion
from training.matcher import HungarianMatcher


def train_epoch(model, dataloader, args):
    if torch.cuda.device_count() == 1:
        model = nn.DataParallel(model)
    model.train()
    epoch_losses = []

    batches = []
    printed = False
    # time_start_epoch = time.time()
    for i_batch, batch in enumerate(dataloader):
        # time_end_load_batch = time.time()
        # print(f"load batch {i_batch} in {time_end_load_batch - time_start_epoch:0.2f}")
        # time_start_forward = time.time()
        # print(f"coarse_target: {batch['cells_targets']}")
        if args.max_batches is not None and i_batch >= args.max_batches:
            break
        # print(f"\r{i_batch}/{len(dataloader)}", end="", flush=True)
        batch_size = len(batch["texts"])
        if args.two_optimizers:
            optimizer_cell_encoder.zero_grad()
            optimizer_other.zero_grad()
        else:
            optimizer.zero_grad()
        # anchor = model.module.encode_text(batch["texts"])
        # anchor_objects, clip_feature_objects = model.module.encode_text_objects(batch["texts_objects"])  # with linear layer(512->256) / without linear layer
        anchor_submap, clip_feature_submap = model.module.encode_text_submap(batch["texts"])
        # anchor_submap, clip_feature_submap = model.module.encode_text_submap(batch["texts_submap"])
        if args.no_objects:
            if args.distill_query_embedding:
                positive, mask3d_output, query_embedding, class_embedding = model.module.encode_cell(
                    batch["cells_coordinates"], batch["cells_features"],
                    batch["cells_targets"], batch["cells_inverse_map"],
                    batch["cells_points"], batch["cells_colors"],
                    num_query=24,
                    freeze_cell_encoder=args.freeze_cell_encoder,
                    use_queries=args.use_queries)

            else:
                positive, mask3d_output = model.module.encode_cell(batch["cells_coordinates"], batch["cells_features"],
                                                                   batch["cells_targets"], batch["cells_inverse_map"],
                                                                   batch["cells_points"], batch["cells_colors"],
                                                                   num_query=24,
                                                                   freeze_cell_encoder=args.freeze_cell_encoder,
                                                                   use_queries=args.use_queries)
            # positive, mask3d_output = model.module.encode_cell(batch["cells_coordinates"], batch["cells_features"], batch["cells_targets"], batch["cells_inverse_map"], batch["cells_points"], batch["cells_colors"], num_query = 24, freeze_cell_encoder=args.freeze_cell_encoder, use_queries=args.use_queries)
        elif args.use_semantic_head:
            positive, sem_pred = model.module.encode_objects(batch["objects"], batch["object_points"])
        else:
            positive = model.module.encode_objects(batch["objects"], batch["object_points"])
        # quit()

        if args.ranking_loss == "triplet":
            negative_cell_objects = [cell.objects for cell in batch["negative_cells"]]
            negative = model.module.encode_objects(negative_cell_objects)
            loss = criterion(anchor, positive, negative)

        else:
            loss = criterion(anchor_submap, positive)
            # loss = criterion(anchor, positive)

        loss = loss
        # loss = 0

        if args.use_semantic_head:
            # print("use semantic head")
            semantic_loss = criterion_class(sem_pred, batch["objects"])
            loss += 0.5 * semantic_loss

        mask3d_loss = {}
        if args.no_objects:
            # batch["cells_targets"] = batch["cells_targets"].to(mask3d_output.device)
            # print(len(batch["cells_targets"]))
            mask3d_loss = criterion_mask3d(mask3d_output, batch["cells_targets"], mask_type="masks")
            # print(f"mask3d_loss: {mask3d_loss}")
            for k in list(mask3d_loss.keys()):
                # print(k)
                if k in criterion_mask3d.weight_dict:
                    # print(f"mask3d_loss[k]: {mask3d_loss[k]}")
                    mask3d_loss[k] *= criterion_mask3d.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    mask3d_loss.pop(k)
            if not args.freeze_cell_encoder:
                loss += sum(mask3d_loss.values())
            # print(f"mask3d_loss: {sum(mask3d_loss.values())}")
            # print(f"retrieval_loss: {loss}")
            quit()
        if args.distill_query_embedding:
            loss = 0
            normalization_factor = batch_size * 256
            mse_loss = F.mse_loss(query_embedding, class_embedding)
            loss += mse_loss / normalization_factor

        loss.backward()
        if args.two_optimizers:
            optimizer_cell_encoder.step()
            optimizer_other.step()
        else:
            optimizer.step()

        epoch_losses.append(loss.item())
        batches.append(batch)

        if i_batch % 80 == 0 and not args.use_semantic_head and not args.no_objects:
            print(f"\r{i_batch}/{len(dataloader)} loss {loss.item():0.2f}", end="", flush=False)
        elif i_batch % 80 == 0 and args.use_semantic_head:
            print(f"\r{i_batch}/{len(dataloader)} loss {loss.item():0.2f}, semantic loss {semantic_loss.item():0.2f}",
                  end="", flush=False)
        elif i_batch % 5 == 0 and args.no_objects and args.distill_query_embedding:
            print(f"\r{i_batch}/{len(dataloader)} distill_query_embedding loss {mse_loss.item():0.2f}", end="",
                  flush=False)
        elif i_batch % 80 == 0 and args.no_objects and not args.freeze_cell_encoder:
            print(
                f"\r{i_batch}/{len(dataloader)} loss {loss.item():0.2f}, instance segmentation loss {sum(mask3d_loss.values()):0.2f}, retrieval_loss {loss.item() - sum(mask3d_loss.values()):0.2f}",
                end="", flush=False)
        elif i_batch % 80 == 0 and args.no_objects and args.freeze_cell_encoder:
            print(
                f"\r{i_batch}/{len(dataloader)} instance segmentation loss {sum(mask3d_loss.values()):0.2f}, retrieval_loss {loss.item():0.2f}",
                end="", flush=False)
    return np.mean(epoch_losses), batches


@torch.no_grad()
def eval_epoch(model, dataloader, args, return_encodings=False, return_texts=False):
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
    args.use_semantic_head = False

    print("start eval")
    if torch.cuda.device_count() == 1:
        model = nn.DataParallel(model)

    model.eval()  # Now eval() seems to increase results
    accuracies = {k: [] for k in args.top_k}
    accuracies_close = {k: [] for k in args.top_k}

    # TODO: Use this again if batches!
    # num_samples = len(dataloader.dataset) if isinstance(dataloader, DataLoader) else np.sum([len(batch['texts']) for batch in dataloader])
    if args.dataset == "K360_cell":
        cells_dataset = dataloader.dataset.get_cell_dataset()
        cells_dataloader = DataLoader(
            cells_dataset,
            batch_size=args.batch_size,
            collate_fn=Kitti360CoarseDataset.collate_fn_cell,
            shuffle=False,
        )
    else:
        cells_dataset = dataloader.dataset.get_cell_dataset()
        cells_dataloader = DataLoader(
            cells_dataset,
            batch_size=args.batch_size,
            collate_fn=Kitti360CoarseDataset.collate_fn,
            shuffle=False,
        )
    cells_dict = {cell.id: cell for cell in cells_dataset.cells}
    cell_size = cells_dataset.cells[0].cell_size

    cell_encodings = np.zeros((len(cells_dataset), model.module.embed_dim))
    db_cell_ids = np.zeros(len(cells_dataset), dtype="<U32")

    text_encodings = np.zeros((len(dataloader.dataset), model.module.embed_dim))
    query_cell_ids = np.zeros(len(dataloader.dataset), dtype="<U32")
    query_poses_w = np.array([pose.pose_w[0:2] for pose in dataloader.dataset.all_poses])

    # Encode the query side
    # t0 = time.time()
    index_offset = 0
    texts_eval = []
    for batch in dataloader:
        # text_enc = model.encode_text(batch["texts"])
        # print("batch['texts_submap']:", batch["texts_submap"])
        text_enc, text_clip_feature = model.module.encode_text_submap(batch["texts"])
        texts_eval.extend(batch["texts"])
        # text_enc, text_clip_feature = model.encode_text_submap(batch["texts"])
        # text_enc_obj, text_clip_feature_obj = model.encode_text_objects(batch["texts_objects"])
        batch_size = len(text_enc)

        text_encodings[index_offset: index_offset + batch_size, :] = (
            text_enc.cpu().detach().numpy()
        )
        query_cell_ids[index_offset: index_offset + batch_size] = np.array(batch["cell_ids"])
        index_offset += batch_size
    # print(f"Encoded {len(text_encodings)} query texts in {time.time() - t0:0.2f}.")

    # get the known classes for sematic prediction evaluation
    if args.use_semantic_head:
        known_classes = dataloader.dataset.get_known_classes()
        known_classes = CLASS_TO_INDEX

        class_correct = {classname: 0 for classname in known_classes}
        print(f"known_classes: {known_classes}")
        print(f"class_correct: {class_correct}")
        class_total = {classname: 0 for classname in known_classes}

    # Encode the database side
    index_offset = 0
    for batch in cells_dataloader:
        if args.use_semantic_head:
            cell_enc, sem_pred = model.module.encode_objects(batch["objects"], batch["object_points"],
                                                             batch["cells_targets"])

            # 将预测转换为类别索引
            sem_pred_id = torch.argmax(sem_pred, dim=1)
            sem_pred_id = sem_pred_id.cpu().detach().numpy()

            # 为真实标签生成类别索引
            class_indices = []
            for i_batch, objects_sample in enumerate(batch["objects"]):
                for obj in objects_sample:
                    class_idx = known_classes.get(obj.label)
                    class_indices.append(class_idx)
            class_indices = np.array(class_indices)

            # 计算每个类别的准确率
            for true_label, predicted_label in zip(class_indices, sem_pred_id):
                true_classname = [name for name, idx in known_classes.items() if idx == true_label][0]
                predicted_classname = [name for name, idx in known_classes.items() if idx == predicted_label][0]
                class_total[true_classname] += 1
                if true_classname == predicted_classname:
                    class_correct[true_classname] += 1

        elif args.no_objects:
            # cell_enc, mask3d_output = model.module.encode_cell(batch["cells_coordinates"], batch["cells_features"], batch["cells_targets"], batch["cells_inverse_map"])
            # print(f"batch_cell_colors: {batch['cells_colors']}")
            cell_enc, mask3d_output = model.module.encode_cell(batch["cells_coordinates"], batch["cells_features"],
                                                               batch["cells_targets"], batch["cells_inverse_map"],
                                                               batch["cells_points"], batch["cells_colors"],
                                                               num_query=24,
                                                               freeze_cell_encoder=args.freeze_cell_encoder,
                                                               use_queries=args.use_queries)

        else:
            cell_enc = model.module.encode_objects(batch["objects"], batch["object_points"])
        batch_size = len(cell_enc)

        # cell_objs_2D_clip_features = batch["feature_2d"]

        cell_encodings[index_offset: index_offset + batch_size, :] = (
            cell_enc.cpu().detach().numpy()
        )
        db_cell_ids[index_offset: index_offset + batch_size] = np.array(batch["cell_ids"])
        index_offset += batch_size

    if args.use_semantic_head:
        # 输出每个类别的准确率
        for classname in known_classes:
            if class_total[classname] > 0:
                accuracy = 100 * class_correct[classname] / class_total[classname]
                print(f"Accuracy of {classname}: {accuracy:.2f}%")
            else:
                print(f"Accuracy of {classname}: N/A (No samples)")

    top_retrievals = {}  # {query_idx: top_cell_ids}
    for query_idx in range(len(text_encodings)):
        if args.ranking_loss != "triplet":  # Asserted above
            scores = cell_encodings[:] @ text_encodings[query_idx]
            assert len(scores) == len(dataloader.dataset.all_cells)  # TODO: remove
            sorted_indices = np.argsort(-1.0 * scores)  # High -> low

        sorted_indices = sorted_indices[0: np.max(args.top_k)]

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
    elif return_texts:
        return accuracies, accuracies_close, top_retrievals, texts_eval
    else:
        return accuracies, accuracies_close, top_retrievals


def load_checkpoint_with_missing_or_exsessive_keys(checkpoint, model):
    state_dict_ori = torch.load(checkpoint)["state_dict"]
    state_dict = {}
    for key in state_dict_ori.keys():
        if "model." in key:
            state_dict[key.replace("model.", "cell_encoder.")] = state_dict_ori[key]

    correct_dict = dict(model.state_dict())
    # print(state_dict.keys())
    # print(correct_dict.keys())
    # quit()

    # if parametrs have different shape, it will randomly initialize
    for key in correct_dict.keys():
        if key not in state_dict:
            # print(f"{key} not in loaded checkpoint")
            state_dict.update({key: correct_dict[key]})
        elif state_dict[key].shape != correct_dict[key].shape:
            # print(
            #     f"incorrect shape {key}:{state_dict[key].shape} vs {correct_dict[key].shape}"
            # )
            state_dict.update({key: correct_dict[key]})
        else:
            # print(f"correct shape {key}")
            pass

    model.load_state_dict(state_dict)
    return model


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
        from dataloading.kitti360pose.cells import Kitti360CoarseDatasetMulti, Kitti360CoarseDataset

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

    elif args.dataset == "K360_cell":
        from dataloading.kitti360pose.cells_sp import Kitti360CoarseDatasetMulti, Kitti360CoarseDataset

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
            collate_fn=Kitti360CoarseDataset.collate_fn_cell,
            shuffle=args.shuffle,
            pin_memory=True
        )

        dataset_val = Kitti360CoarseDatasetMulti(args.base_path, SCENE_NAMES_VAL, val_transform)
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            collate_fn=Kitti360CoarseDataset.collate_fn_cell,
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
        learning_rates = np.logspace(-2.5, -3.5, 3)[args.lr_idx: args.lr_idx + 1]
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

        if args.no_objects:
            # correct_dict = dict(model.state_dict())
            # for key in correct_dict.keys():
            #    print(key)
            # pass
            model = load_checkpoint_with_missing_or_exsessive_keys(args.mask3d_ckpt, model)
            # print(model)
            # quit()
            # load mask3d

            print("mask3d model loaded")
        # 检查是否有多个 GPU 可用

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # 封装模型以在多 GPU 上运行
            devices = [torch.device("cuda:{}".format(i)) for i in range(torch.cuda.device_count())]
            model = model.to("cuda")
            model = nn.DataParallel(model, device_ids=[0, 1])
        else:
            model = model.to(device)

        if args.two_optimizers:
            # cell_encoder_params = list(model.cell_encoder.parameters())
            # other_params = []
            # for name, param in model.named_parameters():
            #     if param not in cell_encoder_params:
            #         other_params.append(param)
            cell_encoder_params, other_params = [], []
            for name, param in model.named_parameters():
                # print(name, param.size())
                if "cell_encoder" in name:
                    # print(name, param.size(), "in cell_encoder params")
                    cell_encoder_params.append(param)
                else:
                    # print(name, param.size(), "in other params")
                    other_params.append(param)
            optimizer_cell_encoder = optim.Adam(cell_encoder_params, lr=lr / 1000)
            optimizer_other = optim.Adam(other_params, lr=lr)

        elif args.distill_query_embedding:  # only optimize the query embedding layer
            query_embedding_params = model.object_encoder.query_linear.parameters()
            optimizer = optim.Adam(query_embedding_params, lr=lr)

        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)

        if args.ranking_loss == "pairwise":
            criterion = PairwiseRankingLoss(margin=args.margin)
        if args.ranking_loss == "hardest":
            criterion = HardestRankingLoss(margin=args.margin)
        if args.ranking_loss == "triplet":
            criterion = nn.TripletMarginLoss(margin=args.margin)
        if args.ranking_loss == "CLIP":
            criterion = ClipLoss()

        criterion_class = SemanticLoss(CLASS_TO_INDEX)
        # print("criterion_class:", dataset_train.get_known_classes(), len(dataset_train.get_known_classes()))
        # quit()
        # criterion_color = nn.CrossEntropyLoss()
        matcher = HungarianMatcher(cost_class=2, cost_mask=5, cost_dice=2, num_points=-1)
        weight_dict = {
            "loss_ce": matcher.cost_class,
            "loss_mask": matcher.cost_mask,
            "loss_dice": matcher.cost_dice,
        }

        aux_weight_dict = {}
        for i in range(len([0, 1, 2, 3]) * 3):
            if i not in [255]:
                aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in weight_dict.items()}
                )
            else:
                aux_weight_dict.update(
                    {k + f"_{i}": 0.0 for k, v in weight_dict.items()}
                )
        weight_dict.update(aux_weight_dict)

        criterion_mask3d = SetCriterion(
            num_classes=22,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=0.1,
            losses=['labels', 'masks'],
            # losses=['masks'],
            # losses=['labels'],
            num_points=-1,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            class_weights=-1,
        )

        if args.two_optimizers:
            scheduler_cell_encoder = optim.lr_scheduler.ExponentialLR(optimizer_cell_encoder, args.lr_gamma)
            scheduler_other = optim.lr_scheduler.ExponentialLR(optimizer_other, args.lr_gamma)
        else:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)

        # test
        # val_acc, val_acc_close, val_retrievals = eval_epoch(model, dataloader_val, args)
        # test

        for epoch in range(1, args.epochs + 1):
            # dataset_train.reset_seed() #OPTION: re-setting seed leads to equal data at every epoch
            time_start_epoch = time.time()

            # test val
            # val_acc, val_acc_close, val_retrievals = eval_epoch(model, dataloader_val, args)
            # quit()

            loss, train_batches = train_epoch(model, dataloader_train, args)
            # train_acc, train_retrievals = eval_epoch(model, train_batches, args)
            time_end_epoch = time.time()
            print(f"Epoch {epoch} in {time_end_epoch - time_start_epoch:0.2f}.")

            if epoch == 1 or epoch >= args.epochs // 8:
                time_start_eval = time.time()
                # train_acc, train_acc_close, train_retrievals = eval_epoch(
                #     model, dataloader_train, args
                # )  # TODO/CARE: Is this ok? Send in batches again?
                val_acc, val_acc_close, val_retrievals = eval_epoch(model, dataloader_val, args)
                time_end_eval = time.time()
                print(f"Evaluation in {time_end_eval - time_start_eval:0.2f}.")

                key = lr
                dict_loss[key].append(loss)
                for k in args.top_k:
                    # dict_acc[k][key].append(train_acc[k])
                    dict_acc_val[k][key].append(val_acc[k])
                    dict_acc_val_close[k][key].append(val_acc_close[k])
                if args.two_optimizers:
                    scheduler_cell_encoder.step()
                    scheduler_other.step()
                else:
                    scheduler.step()
                print(f"\t lr {lr:0.4} loss {loss:0.2f} epoch {epoch} train-acc: ", end="")
                # for k, v in train_acc.items():
                #     print(f"{k}-{v:0.2f} ", end="")
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
                    if args.no_objects:
                        try:
                            model_path = model_path.replace(".pth", "_state_dict.pth")
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
                            print(f"Error saving model!", str(e))
                    else:
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
