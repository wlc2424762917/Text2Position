"""Module to prepare the original Kitti360 into Kitti360Pose
"""
import cv2
import matplotlib.pyplot as plt
import clip
from typing import List
from datapreparation.kitti360pose.utils import *
import cv2
import os
import os.path as osp
import numpy as np
import pickle
import sys
import time
from PIL import Image
from scipy.spatial.distance import cdist

import open3d
from datapreparation.kitti360pose.project_3D_2D import *
from datapreparation.kitti360pose.labels import id2label

try:
    import pptk
except:
    print("pptk not found")
from plyfile import PlyData, PlyElement

from datapreparation.kitti360pose.drawing import (
    show_pptk,
    show_objects,
    plot_cell,
    plot_pose_in_best_cell,
    plot_cells_and_poses,
)
from datapreparation.kitti360pose.utils import (
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    COLORS,
    COLOR_NAMES,
    SCENE_NAMES,
)
from datapreparation.kitti360pose.utils import CLASS_TO_MINPOINTS, CLASS_TO_VOXELSIZE, STUFF_CLASSES
from datapreparation.kitti360pose.imports import Object3d, Cell, Pose
from datapreparation.kitti360pose.descriptions import (
    create_cell,
    describe_pose_in_pose_cell,
    ground_pose_to_best_cell,
)
from datapreparation.args import parse_arguments


def show(img_or_name, img=None):
    if img is not None:
        cv2.imshow(img_or_name, img)
    else:
        cv2.imshow("", img_or_name)
    cv2.waitKey()


def load_points(filepath):
    plydata = PlyData.read(filepath)

    xyz = np.stack((plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"])).T
    rgb = np.stack(
        (plydata["vertex"]["red"], plydata["vertex"]["green"], plydata["vertex"]["blue"])
    ).T

    lbl = plydata["vertex"]["semantic"]
    iid = plydata["vertex"]["instance"]

    return xyz, rgb, lbl, iid


def downsample_points(points, voxel_size):
    # voxel_size = 0.25
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(points.copy())
    _, _, indices_list = point_cloud.voxel_down_sample_and_trace(
        voxel_size, point_cloud.get_min_bound(), point_cloud.get_max_bound()
    )
    # print(f'Downsampled from {len(points)} to {len(indices_list)} points')

    indices = np.array(
        [vec[0] for vec in indices_list]
    )  # Not vectorized but seems fast enough, CARE: first-index color sampling (not averaging)

    return indices


def extract_objects(xyz, rgb, lbl, iid):
    objects = []

    for label_name, label_idx in CLASS_TO_LABEL.items():
        mask = lbl == label_idx
        label_xyz, label_rgb, label_iid = xyz[mask], rgb[mask], iid[mask]

        for obj_iid in np.unique(label_iid):
            mask = label_iid == obj_iid
            obj_xyz, obj_rgb = label_xyz[mask], label_rgb[mask]

            obj_rgb = obj_rgb.astype(np.float32) / 255.0  # Scale colors [0,1]

            # objects.append(Object3d(obj_xyz, obj_rgb, label_name, obj_iid))
            objects.append(
                Object3d(obj_iid, obj_iid, obj_xyz, obj_rgb, label_name)
            )  # Initially also set id instance-id for later mergin. Re-set in create_cell()

    return objects


def project_3d_2d_feature(kitti360Path, sequence, objects, num_random_rounds=10, num_selected_points=5, num_levels=3, multi_level_expansion_ratio=0.1, save_crops=False, out_folder=None):
    # instances_clip_features = np.zeros((len(objects), 768))  # initialize mask clip
    # loop over objects
    print(f"Projecting 3D objects, {sequence}, to 2D features...")
    objects_num = len(objects)
    objects_with_2D_feature_num = 0
    for idx_obj, obj in enumerate(objects):
        # loop over frames
        # perspective
        image_crops = []
        for cam_id in range(2):
            if cam_id == 0 or cam_id == 1:
                camera = CameraPerspective(kitti360Path, sequence, cam_id)
            else:
                raise RuntimeError('Invalid Camera ID! fisheye not available.')
            for frame in camera.frames:
                # perspective
                if cam_id == 0 or cam_id == 1:
                    image_file = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, 'data_rect',
                                              '%010d.png' % frame)
                else:
                    raise RuntimeError('Invalid Camera ID!')
                if not os.path.isfile(image_file):
                    print('Missing %s ...' % image_file)
                    continue

                print(image_file)
                image2D = Image.open(image_file).convert("RGB")
                np_image2D = np.array(image2D)  # np_image2D is (y, x, 3) height, width, channels
                plt.imshow(np_image2D[:, :, ::-1])

                clip_model, clip_preprocess = clip.load("ViT-B/32", device='cuda')
                uv, d = camera.project_vertices(obj.xyz, frame)  # uv is (x, y) width and height
                mask = np.logical_and(np.logical_and(d > 0, uv[0] > 0), uv[1] > 0)
                mask = np.logical_and(np.logical_and(mask, uv[0] < np_image2D.shape[1]),
                                      uv[1] < np_image2D.shape[0])

                plt.plot(uv[0][mask], uv[1][mask], 'r.')
                plt.savefig(f"crop_{frame}_{sequence}_{obj.id}_{obj.label}_project.png")


                point_coords = np.array([uv[0][mask], uv[1][mask]]).T
                if (point_coords.shape[0] > 0):
                    predictor_sam = initialize_sam_model(device='cuda')
                    predictor_sam.set_image(np_image2D)
                    best_mask = run_sam(image_size=np_image2D,
                                        num_random_rounds=num_random_rounds,
                                        num_selected_points=num_selected_points,
                                        point_coords=point_coords,
                                        predictor_sam=predictor_sam, )

                    # MULTI LEVEL CROPS
                    for level in range(num_levels):
                        # get the bbox and corresponding crops
                        x1, y1, x2, y2 = mask2box_multi_level(torch.from_numpy(best_mask), level,
                                                              multi_level_expansion_ratio)
                        cropped_img =image2D.crop((x1, y1, x2, y2))

                        if (save_crops):
                            cropped_img.save(os.path.join(out_folder, f"crop{sequence}_{obj.id}_{obj.label}_{level}.png"))

                        # Currently compute the CLIP feature using the standard clip model
                        # TODO: using mask-adapted CLIP
                        cropped_img_processed = clip_preprocess(cropped_img)
                        image_crops.append(cropped_img_processed)

        if (len(image_crops) > 0):
            image_input = torch.tensor(np.stack(image_crops))
            with torch.no_grad():
                image_features = clip_model.encode_image(image_input.to('cuda')).float()
                image_features /= image_features.norm(dim=-1, keepdim=True)  # normalize

            # instances_clip_features[idx_obj] = image_features.mean(axis=0).cpu().numpy()
            obj.feature_2d = image_features.mean(axis=0).cpu().numpy()
            objects_with_2D_feature_num +=1

        return objects_with_2D_feature_num, objects_num


# def project_3d_2d_feature(kitti360Path, sequence, objects, cam_id, num_random_rounds=10, num_selected_points=5, num_levels=3, multi_level_expansion_ratio=0.1, save_crops=False, out_folder=None):
#     # loop over frames
#     # perspective
#     if cam_id == 0 or cam_id == 1:
#         camera = CameraPerspective(kitti360Path, sequence, cam_id)
#     else:
#         raise RuntimeError('Invalid Camera ID! fisheye not available.')
#     for frame in camera.frames:
#         # perspective
#         if cam_id == 0 or cam_id == 1:
#             image_file = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, 'data_rect',
#                                       '%010d.png' % frame)
#         # fisheye not implemented
#         elif cam_id == 2 or cam_id == 3:
#             image_file = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, 'data_rgb',
#                                       '%010d.png' % frame)
#         else:
#             raise RuntimeError('Invalid Camera ID!')
#         if not os.path.isfile(image_file):
#             print('Missing %s ...' % image_file)
#             continue
#
#         print(image_file)
#         image2D = Image.open(image_file)
#         np_image2D = np.array(image2D)
#
#         clip_preprocess = clip.load(clip_version="ViT-B/32", device='cuda')
#         # project 3D points to 2D
#         for obj in objects:
#             uv, depth = camera.project_vertices(obj.xyz, frame)
#             point_coords = np.array(uv).T
#             if (point_coords.shape[0] > 0):
#                 predictor_sam.set_image(np_image2D)
#                 predictor_sam = initialize_sam_model(device='cuda')
#                 best_mask = run_sam(image_size=np_image2D,
#                                     num_random_rounds=num_random_rounds,
#                                     num_selected_points=num_selected_points,
#                                     point_coords=point_coords,
#                                     predictor_sam=predictor_sam, )
#
#                 # MULTI LEVEL CROPS
#                 for level in range(num_levels):
#                     # get the bbox and corresponding crops
#                     x1, y1, x2, y2 = mask2box_multi_level(torch.from_numpy(best_mask), level,
#                                                           multi_level_expansion_ratio)
#                     cropped_img =image2D.crop((x1, y1, x2, y2))
#
#                     if (save_crops):
#                         cropped_img.save(os.path.join(out_folder, f"crop{mask}_{image_file}_{level}.png"))
#
#                     # Currently compute the CLIP feature using the standard clip model
#                     # TODO: using mask-adapted CLIP
#                     cropped_img_processed = clip_preprocess(cropped_img)
#
#     if (len(images_crops) > 0):
#         image_input = torch.tensor(np.stack(images_crops))
#         with torch.no_grad():
#             image_features = self.clip_model.encode_image(image_input.to(self.device)).float()
#             image_features /= image_features.norm(dim=-1, keepdim=True)  # normalize
#
#         mask_clip[mask] = image_features.mean(axis=0).cpu().numpy()
#
#     return mask_clip


def gather_objects(path_input, folder_name):
    """Gather the objects of a scene"""
    print(f"Loading objects for {folder_name}")

    path = osp.join(path_input, "data_3d_semantics", folder_name, "static")
    assert osp.isdir(path)
    file_names = [f for f in os.listdir(path) if not f.startswith("._")]

    scene_objects = {}

    for i_file_name, file_name in enumerate(file_names):
        print(f'\t loading file {file_name}, {i_file_name} / {len(file_names)}')
        xyz, rgb, lbl, iid = load_points(osp.join(path, file_name))
        file_objects = extract_objects(xyz, rgb, lbl, iid)  # xyz is in world coordinates
        # Add new object or merge to existing
        merges = 0
        for obj in file_objects:
            if obj.id in scene_objects:
                scene_objects[obj.id] = Object3d.merge(scene_objects[obj.id], obj)
                merges += 1
            else:
                scene_objects[obj.id] = obj

            # Downsample the new or merged object
            voxel_size = CLASS_TO_VOXELSIZE[obj.label]
            if voxel_size is not None:
                indices = downsample_points(scene_objects[obj.id].xyz, voxel_size)
                scene_objects[obj.id].apply_downsampling(indices)
        print(f'Merged {merges} / {len(file_objects)}')

    # Thresh objects by number of points
    objects = list(scene_objects.values())
    thresh_counts = {}
    objects_threshed = []
    for obj in objects:
        if len(obj.xyz) < CLASS_TO_MINPOINTS[obj.label]:
            if obj.label in thresh_counts:
                thresh_counts[obj.label] += 1
            else:
                thresh_counts[obj.label] = 1
        else:
            objects_threshed.append(obj)
    print(thresh_counts)

    # TODO Converting 3D objects to 2D CLIP features
    n_obj_2D_feature, n_obj = project_3d_2d_feature(path_input, folder_name, objects_threshed, num_random_rounds=10, num_selected_points=5, num_levels=3, multi_level_expansion_ratio=0.1, save_crops=True, out_folder='./tmp_crop/')
    print(f"Number of objects with 2D feature: {n_obj_2D_feature} / {n_obj}")
    return objects_threshed


def get_close_locations(
    locations: List[np.ndarray], scene_objects: List[Object3d], cell_size, location_objects=None
):
    """Retains all locations that are at most cell_size / 2 distant from an instance-object.
    至少有一个object在所选的location附近

    Args:
        locations (List[np.ndarray]): Pose locations
        scene_objects (List[Object3d]): All objects in the scene
        cell_size ([type]): Size of a cell
        location_objects (optional): Location objects to plot for debugging. Defaults to None.
    """
    instance_objects = [obj for obj in scene_objects if obj.label not in STUFF_CLASSES]
    close_locations, close_location_objects = [], []
    for i_location, location in enumerate(locations):
        for obj in instance_objects:
            closest_point = obj.get_closest_point(location)
            dist = np.linalg.norm(location - closest_point)
            obj.closest_point = None
            if dist < cell_size / 2:
                close_locations.append(location)
                close_location_objects.append(location_objects[i_location])
                break
    assert (
        len(close_locations) > len(locations) * 2 / 5
    ), f"Too few locations retained ({len(close_locations)} of {len(locations)}), are all objects loaded?"
    print(f"close locations: {len(close_locations)} of {len(locations)}")

    if location_objects:
        return close_locations, close_location_objects
    else:
        return close_locations


def create_locations(path_input, folder_name, location_distance, return_location_objects=False):
    """Sample locations along the original trajectories with at least location_distance between any two locations."""
    path = osp.join(path_input, "data_poses", folder_name, "poses.txt")
    poses = np.loadtxt(path)  # poses are in world coordinates
    poses = poses[:, 1:].reshape((-1, 3, 4))  # Convert to 3x4 matrices
    poses = poses[:, :, -1]  # Take last column

    sampled_poses = [
        poses[0],
    ]

    for pose in poses:
        dists = np.linalg.norm(pose - sampled_poses, axis=1)
        if np.min(dists) >= location_distance:
            sampled_poses.append(pose)

    if return_location_objects:
        pose_objects = []
        for pose in sampled_poses:
            pose_objects.append(
                Object3d(-1, -1, np.random.rand(50, 3) * 3 + pose, np.ones((50, 3)), "_pose")  # Random 50 points
            )
        print(f"{folder_name} sampled {len(sampled_poses)} locations")
        return sampled_poses, pose_objects
    else:
        return sampled_poses


def create_cells(objects, locations, scene_name, cell_size, args) -> List[Cell]:
    """Create the cells of the scene."""
    print("Creating cells...")
    cells = []
    none_indices = []
    locations = np.array(locations)  # poses

    assert len(scene_name.split("_")) == 6
    scene_name_short = scene_name.split("_")[-2]

    # Additionally shift each cell-location up, down, left and right by cell_dist / 2
    # TODO: Just create a grid
    if args.shift_cells:
        shifts = np.array(
            [
                [0, 0],
                [-args.cell_dist * 1.05, 0],
                [args.cell_dist * 1.05, 0],
                [0, -args.cell_dist * 1.05],
                [0, args.cell_dist * 1.05],
            ]
        )
        shifts = np.tile(shifts.T, len(locations)).T  # "trick" to tile along axis 0
        locations = np.repeat(locations, 5, axis=0)
        locations[:, 0:2] += shifts
        cell_locations = np.ones_like(locations) * np.inf
    elif args.grid_cells:
        # Define the grid
        x0, y0 = np.int0(np.min(locations[:, 0:2], axis=0))
        x1, y1 = np.int0(np.max(locations[:, 0:2], axis=0))
        x_centers = np.mgrid[x0 : x1 : int(args.cell_dist), y0 : y1 : int(args.cell_dist)][
            0
        ].flatten()
        y_centers = np.mgrid[x0 : x1 : int(args.cell_dist), y0 : y1 : int(args.cell_dist)][
            1
        ].flatten()
        centers = np.vstack((x_centers, y_centers)).T

        # Filter out far-away centers
        distances = cdist(centers, locations[:, 0:2])
        center_indices = np.min(distances, axis=1) <= cell_size
        closest_location_indices = np.argmin(distances, axis=1)

        centers = centers[center_indices]
        closest_location_indices = closest_location_indices[center_indices]

        # Add appropriate heights to the centers - we are not evaluating with heights, this is just a short-cut around a 3-dim grid
        centers = np.hstack((centers, locations[closest_location_indices, 2:3]))
        locations = centers

    for i_location, location in enumerate(locations):
        # Skip cell if it is too close
        if args.shift_cells:
            dists = np.linalg.norm(cell_locations - location, axis=1)
            if np.min(dists) < args.cell_dist:
                continue

        print(f'\r \t locations {i_location} / {len(locations)}', end='')
        bbox = np.hstack(
            (location - cell_size / 2, location + cell_size / 2)
        )  # [x0, y0, z0, x1, y1, z1]

        if args.all_cells:
            cell = create_cell(
                i_location,
                scene_name_short,
                bbox,
                objects,
                num_mentioned=args.num_mentioned,
                all_cells=True,
            )
        else:
            cell = create_cell(
                i_location, scene_name_short, bbox, objects, num_mentioned=args.num_mentioned
            )

        if cell is not None:
            assert len(cell.objects) >= 1
            cells.append(cell)
        else:
            none_indices.append(i_location)
            continue

        if args.shift_cells:
            cell_locations[i_location] = location

    print(f"None cells: {len(none_indices)} / {len(locations)}")
    if len(none_indices) > len(locations):
        return False, none_indices
    else:
        return True, cells


def create_poses(objects: List[Object3d], locations, cells: List[Cell], args) -> List[Pose]:
    """Create the poses of a scene.
    Create cells -> sample pose location -> describe with pose-cell -> convert description to best-cell for training

    Args:
        objects (List[Object3d]): Objects of the scene, needed to verify enough objects a close to a given pose.
        locations: Locations of the original trajectory around which to sample the poses.
        cells (List[Cell]): List of cells
        scene_name: Name of the scene
    """
    print("Creating poses...")
    poses = []
    none_indices = []

    cell_centers = np.array([cell.bbox_w for cell in cells])
    cell_centers = 1 / 2 * (cell_centers[:, 0:3] + cell_centers[:, 3:6])

    if args.pose_count > 1:
        locations = np.repeat(
            locations, args.pose_count, axis=0
        )  # Repeat the locations to increase the number of poses. (Poses are randomly shifted below.)
        assert (
            args.shift_poses == True
        ), "Pose-count greater than 1 but pose shifting is deactivated!"

    unmatched_counts = []
    num_duplicates = 0
    num_rejected = 0  # Pose that were rejected because the cell was None
    for i_location, location in enumerate(locations):
        # Shift the poses randomly to de-correlate database-side cells and query-side poses.
        if args.shift_poses:
            location[0:2] += np.int0(
                np.random.rand(2) * args.cell_size / 2.1
            )  # Shift less than 1 / 2 cell-size so that the pose has a corresponding cell # TODO: shift more?

        # Find closest cell. Discard poses too far from a database-cell so that all poses are retrievable.
        dists = np.linalg.norm(location - cell_centers, axis=1)
        best_cell = cells[np.argmin(dists)]

        if np.min(dists) > args.cell_size / 2:
            none_indices.append(i_location)
            continue

        # Create an extra cell on top of the pose to create the query-side description decoupled from the database-side cells.
        pose_cell_bbox = np.hstack(
            (location - args.cell_size / 2, location + args.cell_size / 2)
        )  # [x0, y0, z0, x1, y1, z1]
        pose_cell = create_cell(
            -1, "pose", pose_cell_bbox, objects, num_mentioned=args.num_mentioned
        )
        if pose_cell is None:  # Pose can be too far from objects to describe it
            none_indices.append(i_location)
            num_rejected += 1
            continue

        # Select description strategy / strategies
        if args.describe_by == "all":
            description_methods = ("closest", "class", "direction")
        else:
            description_methods = (args.describe_by,)

        mentioned_object_ids = []
        do_break = False
        for description_method in description_methods:
            if do_break:
                break

            if args.describe_best_cell:  # Ablation: use ground-truth best cell to describe the pose
                descriptions = describe_pose_in_pose_cell(
                    location, best_cell, description_method, args.num_mentioned
                )
            else:  # Obtain the descriptions based on the pose-cell
                descriptions = describe_pose_in_pose_cell(
                    location, pose_cell, description_method, args.num_mentioned
                )

            if descriptions is None or len(descriptions) < args.num_mentioned:
                none_indices.append(i_location)
                do_break = True  # Don't try again with other strategy
                continue

            assert len(descriptions) == args.num_mentioned

            # Convert the descriptions to the best-matching database cell for training. Some descriptions might not be matched anymore.
            if args.all_cells:
                descriptions, pose_in_cell, num_unmatched = ground_pose_to_best_cell(
                    location, descriptions, best_cell, all_cells=True
                )
            else:
                descriptions, pose_in_cell, num_unmatched = ground_pose_to_best_cell(
                    location, descriptions, best_cell
                )
            assert len(descriptions) == args.num_mentioned
            unmatched_counts.append(num_unmatched)

            if args.describe_best_cell:
                assert num_unmatched == 0, "Unmatched descriptors for best cell!"

            # Only append the new description if it actually comes out different than the ones before
            mentioned_ids = sorted([d.object_id for d in descriptions if d.is_matched])
            if mentioned_ids in mentioned_object_ids:
                num_duplicates += 1
            else:
                pose = Pose(
                    pose_in_cell,
                    location,
                    best_cell.id,
                    best_cell.scene_name,
                    descriptions,
                    described_by=description_method,
                )
                poses.append(pose)
                mentioned_object_ids.append(mentioned_ids)

    print(f"Num duplicates: {num_duplicates} / {len(poses)}")
    print(
        f"None poses: {len(none_indices)} / {len(locations)}, avg. unmatched: {np.mean(unmatched_counts):0.1f}, num_rejected: {num_rejected}"
    )
    if len(none_indices) > len(locations):
        return False, none_indices
    else:
        return True, poses


if __name__ == "__main__":
    np.random.seed(4096)  # Set seed to re-produce results

    # 2013_05_28_drive_0003_sync
    args = parse_arguments()
    print(str(args).replace(",", "\n"))
    print()

    cell_locations, cell_location_objects = create_locations(
        args.path_in,
        args.scene_name,
        location_distance=args.cell_dist,
        return_location_objects=True,
    )

    pose_locations, pose_location_objects = create_locations(
        args.path_in,
        args.scene_name,
        location_distance=args.pose_dist,
        return_location_objects=True,
    )

    path_objects = osp.join(args.path_in, "objects", f"{args.scene_name}.pkl")
    path_cells = osp.join(args.path_out, "cells", f"{args.scene_name}.pkl")
    path_poses = osp.join(args.path_out, "poses", f"{args.scene_name}.pkl")

    t_start = time.time()

    # Load or gather objects of the scene
    if not osp.isfile(path_objects):  # Build if not cached
        objects = gather_objects(args.path_in, args.scene_name)
        pickle.dump(objects, open(path_objects, "wb"))
        print(f"Saved objects to {path_objects}")
    else:
        print(f"Loaded objects from {path_objects}")
        objects = pickle.load(open(path_objects, "rb"))

    t_object_loaded = time.time()

    # Filter out locations that are too far from objects
    cell_locations, cell_location_objects = get_close_locations(
        cell_locations, objects, args.cell_size, cell_location_objects
    )

    # Filter out locations that are too far from objects
    pose_locations, pose_location_objects = get_close_locations(
        pose_locations, objects, args.cell_size, pose_location_objects
    )

    t_close_locations = time.time()

    t0 = time.time()
    res, cells = create_cells(objects, cell_locations, args.scene_name, args.cell_size, args)
    assert res is True, "Too many cell nones, quitting."
    t1 = time.time()
    print("Ela cells:", t1 - t0, "Num cells", len(cells))

    t_cells_created = time.time()

    res, poses = create_poses(objects, pose_locations, cells, args)
    assert res is True, "Too many pose nones, quitting."

    t_poses_created = time.time()

    print(
        f"Ela: objects {t_object_loaded - t_start:0.2f} close {t_close_locations - t_object_loaded:0.2f} cells {t_cells_created - t_close_locations:0.2f} poses {t_poses_created - t_cells_created:0.2f}"
    )
    print()

    pickle.dump(cells, open(path_cells, "wb"))
    print(f"Saved {len(cells)} cells to {path_cells}")

    pickle.dump(poses, open(path_poses, "wb"))
    print(f"Saved {len(poses)} poses to {path_poses}")
    print()

    # Debugging
    idx = np.random.randint(len(poses))
    pose = poses[idx]
    cells_dict = {cell.id: cell for cell in cells}
    cell = cells_dict[pose.cell_id]
    print("IDX:", idx)
    print(pose.get_text())

    img = plot_pose_in_best_cell(cell, pose)
    cv2.imwrite(f"cell_{args.describe_by}_idx{idx}.png", img)
