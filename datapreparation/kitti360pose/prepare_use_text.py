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
from PIL import Image

CLASS_TO_LABEL = {
    "building": 11,
    "pole": 17,
    "traffic light": 19,
    "traffic sign": 20,
    "garage": 34,
    "stop": 36,
    "smallpole": 37,
    "lamp": 38,
    "trash bin": 39,
    "vending machine": 40,
    "box": 41,
    "road": 7,
    "sidewalk": 8,
    "parking": 9,
    "wall": 12,
    "fence": 13,
    "guard rail": 14,
    "bridge": 15,
    "tunnel": 16,
    "vegetation": 21,
    "terrain": 22,
}

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

    clip_model, clip_preprocess = clip.load("ViT-B/32", device='cuda')

    for cam_id in range(2):
        if cam_id == 0 or cam_id == 1:
            camera = CameraPerspective(kitti360Path, sequence, cam_id)
        else:
            raise RuntimeError('Invalid Camera ID! fisheye not available.')
        # loop over frames
        for frame in camera.frames:
            # if frame > 10:
            #     break
            if cam_id == 0 or cam_id == 1:
                # Check if the semantic mask file exists
                if os.path.exists(os.path.join(kitti360Path, 'data_2d_semantics', sequence, 'image_%02d' % cam_id,
                                               'semantic',
                                               '%010d.png' % frame)):
                    image_file = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, 'data_rect',
                                              '%010d.png' % frame)
                    sem_mask_file = os.path.join(kitti360Path, 'data_2d_semantics', sequence, 'image_%02d' % cam_id,
                                                 'semantic',
                                                 '%010d.png' % frame)
                else:
                    print('Missing %s in semantics' % frame)
                    continue

            else:
                raise RuntimeError('Invalid Camera ID!')
            if not os.path.isfile(image_file):
                print('Missing %s ...' % image_file)
                continue
            print("processing frame:", image_file)

            image2D = Image.open(image_file).convert("RGB")
            np_image2D = np.array(image2D)  # np_image2D is (y, x, 3) height, width, channels

            sem_mask2D = Image.open(sem_mask_file)
            np_sem_mask2D = np.array(sem_mask2D)

            depth_buffer = np.full(np_image2D.shape[0:2], np.inf)
            label3DSemanticsPaths = os.path.join(kitti360Path, 'data_3d_semantics', sequence, "static")
            plys = glob.glob(os.path.join(label3DSemanticsPaths, "*.ply"))
            for ply in plys:
                # GET DEPTH BUFFER
                xyz, rgb, lbl, iid = load_points(ply)
                uv, d = camera.project_vertices(xyz, frame)
                mask = np.logical_and(np.logical_and(d > 0, uv[0] > 0), uv[1] > 0)
                mask = np.logical_and(np.logical_and(mask, uv[0] < np_image2D.shape[1]), uv[1] < np_image2D.shape[0])
                uv, d = (uv[0][mask], uv[1][mask]), d[mask]
                for i in range(len(d)):
                    if d[i] > 0 and d[i] < depth_buffer[int(uv[1][i]), int(uv[0][i])]:
                        depth_buffer[int(uv[1][i]), int(uv[0][i])] = d[i]

            # loop over objects
            for idx_obj, obj in enumerate(objects):
                image_crops = []
                # print(obj.label, len(obj.xyz))
                uv, d = camera.project_vertices(obj.xyz, frame)
                mask = np.logical_and(np.logical_and(d > 0, uv[0] > 0), uv[1] > 0)
                mask = np.logical_and(np.logical_and(mask, uv[0] < np_image2D.shape[1]), uv[1] < np_image2D.shape[0])
                uv, d = (uv[0][mask], uv[1][mask]), d[mask]
                mask = [d[i] <= depth_buffer[int(uv[1][i]), int(uv[0][i])] for i in range(len(d))]

                # Combine the UV coordinates and color data for clustering
                combined_data = np.column_stack((uv[0][mask], uv[1][mask]))
                if combined_data.shape[0] > 10:
                    # print(obj.label, len(uv[0][mask]))
                    # Normalize the combined data
                    # Apply DBSCAN
                    dbscan = DBSCAN(eps=8, min_samples=10)
                    clusters = dbscan.fit_predict(combined_data)

                    # Determine the densest cluster
                    cluster_counts = Counter(clusters)
                    # Exclude noise points which are labeled as -1
                    if -1 in cluster_counts:
                        del cluster_counts[-1]
                    # 检查是否有有效的聚类
                    if len(cluster_counts) <= 0:
                        # print("No valid clusters found")
                        densest_cluster_id = None
                    else:
                        # 获取数量最多的聚类的 ID
                        densest_cluster_id = cluster_counts.most_common(1)[0][0]

                        print("Number of points in the densest cluster:", cluster_counts[densest_cluster_id])

                        # Get the indices of the points in the densest cluster
                        densest_cluster_indices = np.where(clusters == densest_cluster_id)[0]
                        densest_cluster_points_uv = (
                            uv[0][mask][densest_cluster_indices], uv[1][mask][densest_cluster_indices])

                        # 计算densest_cluster_points_uv在semantic mask中的label和instance label有多少一致
                        obj_sem_id = CLASS_TO_LABEL[obj.label]
                        sem_matched = np_sem_mask2D[
                                          densest_cluster_points_uv[1], densest_cluster_points_uv[0]] == obj_sem_id
                        num_sem_matched = np.sum(sem_matched)
                        num_total = len(sem_matched)

                        if num_sem_matched / num_total < 0.8:
                            # print(
                            #     "The semantic labels of the densest cluster points are not consistent with the object label")
                            continue
                        else:
                            # print(
                            #     "The semantic labels of the densest cluster points are mostly consistent with the object label")

                            # plt.figure()  # 创建一个新的图形
                            # plt.imshow(np_image2D)  # Adjust if necessary for color channels
                            # plt.plot(densest_cluster_points_uv[0], densest_cluster_points_uv[1], 'r.', markersize=0.8)
                            # plt.savefig(os.path.join(out_folder, f"cluster{sequence}_{obj.id}_{obj.label}.png"))
                            # plt.close()

                            predictor_sam = initialize_sam_model(device='cuda')
                            predictor_sam.set_image(np_image2D)
                            best_mask = run_sam(image_size=np_image2D,
                                                num_random_rounds=num_random_rounds,
                                                num_selected_points=num_selected_points,
                                                point_coords=np.array([densest_cluster_points_uv[0], densest_cluster_points_uv[1]]).T,
                                                predictor_sam=predictor_sam, )

                            if sum(best_mask.flatten()) > 0:
                                # MULTI LEVEL CROPS
                                for level in range(num_levels):
                                    # mask out bg
                                    bg_rgb_color = (0.48145466 * 255.0, 0.4578275 * 255.0, 0.40821073 * 255.0)
                                    best_mask_boolean = best_mask.astype(bool)
                                    np_image2D_mask = np_image2D.copy()
                                    np_image2D_mask[~best_mask_boolean] = bg_rgb_color
                                    image_2D_masked = Image.fromarray(np_image2D_mask.astype('uint8'))

                                    # get the bbox and corresponding crops
                                    x1, y1, x2, y2 = mask2box_multi_level(torch.from_numpy(best_mask), level,
                                                                          multi_level_expansion_ratio)
                                    # cropped_img =image_2D_masked.crop((x1, y1, x2, y2))
                                    cropped_img = image2D.crop((x1, y1, x2, y2))
                                    if cropped_img.size[0] < 60 and cropped_img.size[1] < 60:
                                        # print("crop too small")
                                        break

                                    if (save_crops):
                                        cropped_img.save(os.path.join(out_folder, f"crop{sequence}_{obj.id}_{obj.label}_{level}.png"))

                                    # Currently compute the CLIP feature using the standard clip model
                                    # TODO: using mask-adapted CLIP
                                    cropped_img_processed = clip_preprocess(cropped_img)
                                    image_crops.append(cropped_img_processed)

                if (len(image_crops) > 0):
                    image_input = torch.tensor(np.stack(image_crops))  # (num_crops, 3, 224, 224)
                    with torch.no_grad():
                        image_features = clip_model.encode_image(image_input.to('cuda')).float()  # (num_crops, 512)
                        image_features /= image_features.norm(dim=-1, keepdim=True)  # normalize, (num_crops, 512)

                    # instances_clip_features[idx_obj] = image_features.mean(axis=0).cpu().numpy()
                    obj.image_2d.append(image_crops)
                    obj.feature_2d.append(image_features.mean(axis=0).cpu().numpy())
                    # obj.feature_2d_text = clip.tokenize([obj.label]).to('cuda')


    # torch.cuda.empty_cache()
    # clip_model, clip_preprocess = clip.load("ViT-B/32", device='cuda')
    for idx_obj, obj in enumerate(objects):
        # feature_2d_text = clip.tokenize([obj.label]).to('cuda')
        # obj.feature_2d_text = clip_model.encode_text(feature_2d_text).float()
        # obj.feature_2d_text /= obj.feature_2d_text.norm(dim=-1, keepdim=True)  # normalize

        if len(obj.feature_2d) == 0:
            # obj.feature_2d = obj.feature_2d_text.cpu().detach().numpy()
            obj.feature_2d = None
        else:
            obj.feature_2d = np.stack(obj.feature_2d)
            obj.feature_2d = obj.feature_2d.mean(axis=0)
            objects_with_2D_feature_num += 1

    return objects_with_2D_feature_num, objects_num


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

    # TODO Converting 3D objects to 2D CLIP features
    model, preprocess = clip.load("ViT-B/32", device='cuda')
    n = 0
    n_obj = len(objects_threshed)
    for obj in objects_threshed:
        with torch.no_grad():
            print(f'processing object {n} / {n_obj}, {obj.label}')
            text = clip.tokenize(obj.label).to('cuda')
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            obj.feature_2d = text_features.cpu().numpy()
            n+=1
    # n_obj_2D_feature, n_obj = project_3d_2d_feature(path_input, folder_name, objects_threshed, num_random_rounds=10, num_selected_points=5, num_levels=3, multi_level_expansion_ratio=0.1, save_crops=True, out_folder='./tmp_crop/')
    print(f"Number of objects with 2D feature: {n} / {n_obj}")
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
        x0, y0 = np.intp(np.min(locations[:, 0:2], axis=0))
        x1, y1 = np.intp(np.max(locations[:, 0:2], axis=0))
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
            location[0:2] += np.intp(
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
