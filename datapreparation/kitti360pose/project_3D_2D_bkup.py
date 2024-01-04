import glob
import os
import numpy as np
import re
from sklearn.cluster import DBSCAN
from collections import Counter
import open3d
import yaml
import sys
from plyfile import PlyData, PlyElement
from sklearn.preprocessing import StandardScaler

from datapreparation.kitti360pose.utils import *
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
from datapreparation.kitti360pose.utils import *
from PIL import Image


def checkfile(filename):
    if not os.path.isfile(filename):
        raise RuntimeError('%s does not exist!' % filename)


def readVariable(fid, name, M, N):
    # rewind
    fid.seek(0, 0)

    # search for variable identifier
    line = 1
    success = 0
    while line:
        line = fid.readline()
        if line.startswith(name):
            success = 1
            break

    # return if variable identifier not found
    if success == 0:
        return None

    # fill matrix
    line = line.replace('%s:' % name, '')
    line = line.split()
    assert (len(line) == M * N)
    line = [float(x) for x in line]
    mat = np.array(line).reshape(M, N)

    return mat


def loadCalibrationCameraToPose(filename):
    # check file
    checkfile(filename)

    # open file
    fid = open(filename, 'r');

    # read variables
    Tr = {}
    cameras = ['image_00', 'image_01', 'image_02', 'image_03']
    lastrow = np.array([0, 0, 0, 1]).reshape(1, 4)
    for camera in cameras:
        Tr[camera] = np.concatenate((readVariable(fid, camera, 3, 4), lastrow))

    # close file
    fid.close()
    return Tr


def readYAMLFile(fileName):
    '''make OpenCV YAML file compatible with python'''
    ret = {}
    skip_lines = 1  # Skip the first line which says "%YAML:1.0". Or replace it with "%YAML 1.0"
    with open(fileName) as fin:
        for i in range(skip_lines):
            fin.readline()
        yamlFileOut = fin.read()
        myRe = re.compile(r":([^ ])")  # Add space after ":", if it doesn't exist. Python yaml requirement
        yamlFileOut = myRe.sub(r': \1', yamlFileOut)
        ret = yaml.load(yamlFileOut)
    return ret


class Camera:
    def __init__(self):

        # load intrinsics
        self.load_intrinsics(self.intrinsic_file)

        # load poses
        poses = np.loadtxt(self.pose_file)
        frames = poses[:, 0]
        poses = np.reshape(poses[:, 1:], [-1, 3, 4])
        self.cam2world = {}
        self.frames = frames
        for frame, pose in zip(frames, poses):
            pose = np.concatenate((pose, np.array([0., 0., 0., 1.]).reshape(1, 4)))
            # consider the rectification for perspective cameras
            if self.cam_id == 0 or self.cam_id == 1:
                self.cam2world[frame] = np.matmul(np.matmul(pose, self.camToPose),
                                                  np.linalg.inv(self.R_rect))
            # fisheye cameras
            elif self.cam_id == 2 or self.cam_id == 3:
                self.cam2world[frame] = np.matmul(pose, self.camToPose)
            else:
                raise RuntimeError('Unknown Camera ID!')

    def world2cam(self, points, R, T, inverse=False):
        assert (points.ndim == R.ndim)
        assert (T.ndim == R.ndim or T.ndim == (R.ndim - 1))
        ndim = R.ndim
        if ndim == 2:
            R = np.expand_dims(R, 0)
            T = np.reshape(T, [1, -1, 3])
            points = np.expand_dims(points, 0)
        if not inverse:
            points = np.matmul(R, points.transpose(0, 2, 1)).transpose(0, 2, 1) + T
        else:
            points = np.matmul(R.transpose(0, 2, 1), (points - T).transpose(0, 2, 1))

        if ndim == 2:
            points = points[0]

        return points

    def cam2image(self, points):
        raise NotImplementedError

    def load_intrinsics(self, intrinsic_file):
        raise NotImplementedError

    def project_vertices(self, vertices, frameId, inverse=True):

        # current camera pose
        curr_pose = self.cam2world[frameId]
        T = curr_pose[:3, 3]
        R = curr_pose[:3, :3]

        # convert points from world coordinate to local coordinate
        points_local = self.world2cam(vertices, R, T, inverse)

        # perspective projection
        u, v, depth = self.cam2image(points_local)

        return (u, v), depth

    def __call__(self, obj3d, frameId):

        vertices = obj3d.vertices

        uv, depth = self.project_vertices(vertices, frameId)

        obj3d.vertices_proj = uv
        obj3d.vertices_depth = depth
        obj3d.generateMeshes()


class CameraPerspective(Camera):

    def __init__(self, root_dir, seq='2013_05_28_drive_0009_sync', cam_id=0):
        # perspective camera ids: {0,1}, fisheye camera ids: {2,3}
        assert (cam_id == 0 or cam_id == 1)

        pose_dir = os.path.join(root_dir, 'data_poses', seq)
        print(pose_dir)
        calib_dir = os.path.join(root_dir, 'calibration')
        self.pose_file = os.path.join(pose_dir, "poses.txt")
        self.intrinsic_file = os.path.join(calib_dir, 'perspective.txt')
        fileCameraToPose = os.path.join(calib_dir, 'calib_cam_to_pose.txt')
        self.camToPose = loadCalibrationCameraToPose(fileCameraToPose)['image_%02d' % cam_id]
        self.cam_id = cam_id
        super(CameraPerspective, self).__init__()

    def load_intrinsics(self, intrinsic_file):
        ''' load perspective intrinsics '''

        intrinsic_loaded = False
        width = -1
        height = -1
        with open(intrinsic_file) as f:
            intrinsics = f.read().splitlines()
        for line in intrinsics:
            line = line.split(' ')
            if line[0] == 'P_rect_%02d:' % self.cam_id:
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3, 4])
                intrinsic_loaded = True
            elif line[0] == 'R_rect_%02d:' % self.cam_id:
                R_rect = np.eye(4)
                R_rect[:3, :3] = np.array([float(x) for x in line[1:]]).reshape(3, 3)
            elif line[0] == "S_rect_%02d:" % self.cam_id:
                width = int(float(line[1]))
                height = int(float(line[2]))
        assert (intrinsic_loaded == True)
        assert (width > 0 and height > 0)

        self.K = K
        self.width, self.height = width, height
        self.R_rect = R_rect

    def cam2image(self, points):
        ndim = points.ndim
        if ndim == 2:
            points = np.expand_dims(points, 0)
        points_proj = np.matmul(self.K[:3, :3].reshape([1, 3, 3]), points)
        depth = points_proj[:, 2, :]
        depth[depth == 0] = -1e-6
        u = np.round(points_proj[:, 0, :] / np.abs(depth)).astype(np.int32)
        v = np.round(points_proj[:, 1, :] / np.abs(depth)).astype(np.int32)

        if ndim == 2:
            u = u[0];
            v = v[0];
            depth = depth[0]
        return u, v, depth


class CameraFisheye(Camera):
    def __init__(self, root_dir, seq='2013_05_28_drive_0009_sync', cam_id=2):
        # perspective camera ids: {0,1}, fisheye camera ids: {2,3}
        assert (cam_id == 2 or cam_id == 3)

        pose_dir = os.path.join(root_dir, 'data_poses', seq)
        calib_dir = os.path.join(root_dir, 'calibration')
        self.pose_file = os.path.join(pose_dir, "poses.txt")
        self.intrinsic_file = os.path.join(calib_dir, 'image_%02d.yaml' % cam_id)
        fileCameraToPose = os.path.join(calib_dir, 'calib_cam_to_pose.txt')
        self.camToPose = loadCalibrationCameraToPose(fileCameraToPose)['image_%02d' % cam_id]
        self.cam_id = cam_id
        super(CameraFisheye, self).__init__()

    def load_intrinsics(self, intrinsic_file):
        ''' load fisheye intrinsics '''

        intrinsics = readYAMLFile(intrinsic_file)

        self.width, self.height = intrinsics['image_width'], intrinsics['image_height']
        self.fi = intrinsics

    def cam2image(self, points):
        ''' camera coordinate to image plane '''
        points = points.T
        norm = np.linalg.norm(points, axis=1)

        x = points[:, 0] / norm
        y = points[:, 1] / norm
        z = points[:, 2] / norm

        x /= z + self.fi['mirror_parameters']['xi']
        y /= z + self.fi['mirror_parameters']['xi']

        k1 = self.fi['distortion_parameters']['k1']
        k2 = self.fi['distortion_parameters']['k2']
        gamma1 = self.fi['projection_parameters']['gamma1']
        gamma2 = self.fi['projection_parameters']['gamma2']
        u0 = self.fi['projection_parameters']['u0']
        v0 = self.fi['projection_parameters']['v0']

        ro2 = x * x + y * y
        x *= 1 + k1 * ro2 + k2 * ro2 * ro2
        y *= 1 + k1 * ro2 + k2 * ro2 * ro2

        x = gamma1 * x + u0
        y = gamma2 * y + v0

        return x, y, norm * points[:, 2] / np.abs(points[:, 2])


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    from labels import id2label

    semantics = 1
    os.environ["KITTI360_DATASET"] = '/home/wanglichao/KITTI-360'

    if 'KITTI360_DATASET' in os.environ:
        kitti360Path = os.environ['KITTI360_DATASET']
    else:
        kitti360Path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), '..', '..')

    seq = 0
    cam_id = 0
    sequence = '2013_05_28_drive_%04d_sync' % seq
    # perspective
    if cam_id == 0 or cam_id == 1:
        camera = CameraPerspective(kitti360Path, sequence, cam_id)
    # fisheye
    elif cam_id == 2 or cam_id == 3:
        camera = CameraFisheye(kitti360Path, sequence, cam_id)
        print(camera.fi)
    else:
        raise RuntimeError('Invalid Camera ID!')

    # loop over frames
    for frame in camera.frames:
        # perspective
        if cam_id == 0 or cam_id == 1:
            # Check if the semantic mask file exists
            if os.path.exists(os.path.join(kitti360Path, 'data_2d_semantics', sequence, 'image_%02d' % cam_id,
                                         'data_rect',
                                         '%010d.png' % frame)):

                image_file = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, 'data_rect',
                                          '%010d.png' % frame)

                sem_mask_file = os.path.join(kitti360Path, 'data_2d_semantics', sequence, 'image_%02d' % cam_id, 'data_rect',
                                          '%010d.png' % frame)
        # fisheye
        elif cam_id == 2 or cam_id == 3:
            image_file = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, 'data_rgb',
                                      '%010d.png' % frame)
        else:
            raise RuntimeError('Invalid Camera ID!')
        if not os.path.isfile(image_file):
            print('Missing %s ...' % image_file)
            continue

        print(image_file)
        image = cv2.imread(image_file)
        # plt.imshow(image[:, :, ::-1])
        np_image2D = np.array(image[:, :, ::-1])

        sem_mask2D = Image.open(sem_mask_file)
        np_sem_mask2D = np.array(sem_mask2D)

        depth_buffer = np.full(image.shape[0:2], np.inf)
        # print(depth_buffer.shape)
        # 3D bbox
        if semantics == 0:
            from annotation import Annotation3D

            label3DBboxPath = os.path.join(kitti360Path, 'data_3d_bboxes')
            annotation3D = Annotation3D(label3DBboxPath, sequence)

            for k, v in annotation3D.objects.items():
                if len(v.keys()) == 1 and (-1 in v.keys()):  # show static only
                    obj3d = v[-1]
                    if not id2label[obj3d.semanticId].name == 'building':  # show buildings only
                        continue
                    camera(obj3d, frame)
                    vertices = np.asarray(obj3d.vertices_proj).T  # 3xN  # 顶点坐标
                    for line in obj3d.lines:
                        v = [obj3d.vertices[line[0]] * x + obj3d.vertices[line[1]] * (1 - x) for x in np.arange(0, 1, 0.01)]
                        uv, d = camera.project_vertices(np.asarray(v), frame)
                        mask = np.logical_and(np.logical_and(d > 0, uv[0] > 0), uv[1] > 0)  # 去除投影到图像平面后的线段坐标中的异常值
                        mask = np.logical_and(np.logical_and(mask, uv[0] < image.shape[1]), uv[1] < image.shape[0])  # 去除投影到图像平面后的线段坐标中的异常值
                        plt.plot(uv[0][mask], uv[1][mask], 'r.')

        elif semantics == 1:
            def load_points(filepath):
                plydata = PlyData.read(filepath)

                xyz = np.stack((plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"])).T
                rgb = np.stack(
                    (plydata["vertex"]["red"], plydata["vertex"]["green"], plydata["vertex"]["blue"])
                ).T

                lbl = plydata["vertex"]["semantic"]
                iid = plydata["vertex"]["instance"]

                return xyz, rgb, lbl, iid

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

            label3DSemanticsPaths = os.path.join(kitti360Path, 'data_3d_semantics', sequence, "static")
            plys = glob.glob(os.path.join(label3DSemanticsPaths, "*.ply"))
            scene_objects = {}
            for ply in plys:
                xyz, rgb, lbl, iid = load_points(ply)
                file_objects = extract_objects(xyz, rgb, lbl, iid)  # xyz is in world coordinates

                # GET DEPTH BUFFER
                uv, d = camera.project_vertices(xyz, frame)
                mask = np.logical_and(np.logical_and(d > 0, uv[0] > 0), uv[1] > 0)
                mask = np.logical_and(np.logical_and(mask, uv[0] < image.shape[1]), uv[1] < image.shape[0])
                uv, d = (uv[0][mask], uv[1][mask]), d[mask]
                for i in range(len(d)):
                    if d[i] > 0 and d[i] < depth_buffer[int(uv[1][i]), int(uv[0][i])]:
                        depth_buffer[int(uv[1][i]), int(uv[0][i])] = d[i]

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
            # print(len(objects_threshed))
            for obj in objects_threshed:
                # print(obj.label, len(obj.xyz))
                if obj.label != 'building':
                    continue
                print(obj.label, len(obj.xyz))
                uv, d = camera.project_vertices(obj.xyz, frame)
                # print(len(uv))
                # print(uv[0].shape)
                # uv = np.array(uv)
                # print(uv.shape)
                mask = np.logical_and(np.logical_and(d > 0, uv[0] > 0), uv[1] > 0)
                mask = np.logical_and(np.logical_and(mask, uv[0] < image.shape[1]), uv[1] < image.shape[0])
                uv, d = (uv[0][mask], uv[1][mask]), d[mask]
                mask = [d[i] <= depth_buffer[int(uv[1][i]), int(uv[0][i])] for i in range(len(d))]
                # print(uv[0][mask].shape, uv[1][mask].shape)
                # Combine the UV coordinates and color data for clustering
                combined_data = np.column_stack((uv[0][mask], uv[1][mask]))
                # print(uv[0][mask].shape, uv[1][mask].shape)
                # print(np.array([uv[0][mask], uv[1][mask]]).shape)
                # plt.plot(uv[0][mask], uv[1][mask], 'r.', markersize=0.3)
                if combined_data.shape[0] > 180:
                    print(combined_data.shape)
                    # Normalize the combined data
                    # It's important to scale the color information to be on a similar numerical range as the UV coordinates
                    # Apply DBSCAN
                    # The 'eps' and 'min_samples' parameters are crucial and should be tuned based on the specifics of the dataset
                    dbscan = DBSCAN(eps=10, min_samples=120)
                    clusters = dbscan.fit_predict(combined_data)

                    # Determine the densest cluster
                    cluster_counts = Counter(clusters)
                    # Exclude noise points which are labeled as -1
                    if -1 in cluster_counts:
                        del cluster_counts[-1]
                    # 检查是否有有效的聚类
                    if len(cluster_counts) > 0:
                        # 获取数量最多的聚类的 ID
                        densest_cluster_id = cluster_counts.most_common(1)[0][0]

                        print("Number of points in the densest cluster:", cluster_counts[densest_cluster_id])

                        # Get the indices of the points in the densest cluster
                        densest_cluster_indices = np.where(clusters == densest_cluster_id)[0]
                        densest_cluster_points_uv = (
                        uv[0][mask][densest_cluster_indices], uv[1][mask][densest_cluster_indices])

                        # Visualize the clustering results (optional)
                        # for i, point in enumerate(densest_cluster_points_uv):
                        #     plt.plot(point[0], point[1], '.', markersize=1)

                        plt.figure()  # 创建一个新的图形
                        plt.imshow(image[:, :, ::-1])  # Adjust if necessary for color channels
                        plt.plot(uv[0][mask], uv[1][mask], 'r.', markersize=0.8)
                        plt.show()


                        plt.figure()  # 创建一个新的图形
                        plt.imshow(image[:, :, ::-1])  # Adjust if necessary for color channels
                        plt.plot(densest_cluster_points_uv[0], densest_cluster_points_uv[1], 'r.', markersize=0.8)
                        plt.show()

                        # If you want to sample points from the densest cluster, you can use:
                        num_points_to_sample = 5  # or 10, as needed
                        if len(densest_cluster_indices) > num_points_to_sample:
                            sampled_indices = np.random.choice(densest_cluster_indices, size=num_points_to_sample,
                                                               replace=False)
                        else:
                            sampled_indices = densest_cluster_indices
                            sampled_indices = densest_cluster_indices
                    else:
                        # 没有找到有效聚类
                        print("No valid clusters found")
                        densest_cluster_id = None


                # print(uv[0][mask].shape, uv[1][mask].shape)
                # print(np.array([uv[0][mask], uv[1][mask]]).shape)
                # plt.plot(uv[0][mask], uv[1][mask], 'r.', markersize=0.3)
                if (sum(mask) > 0):
                    predictor_sam = initialize_sam_model(device='cuda')
                    predictor_sam.set_image(np_image2D)
                    best_mask = run_sam(image_size=np_image2D,
                                        num_random_rounds=5,
                                        num_selected_points=5,
                                        point_coords=np.array([uv[0][mask], uv[1][mask]]).T,
                                        predictor_sam=predictor_sam, )
                   #  plt.imshow(best_mask, alpha=0.5)

        else:
            def load_points(filepath):
                plydata = PlyData.read(filepath)

                xyz = np.stack((plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"])).T
                rgb = np.stack(
                    (plydata["vertex"]["red"], plydata["vertex"]["green"], plydata["vertex"]["blue"])
                ).T

                lbl = plydata["vertex"]["semantic"]
                iid = plydata["vertex"]["instance"]

                return xyz, rgb, lbl, iid

            label3DSemanticsPaths = os.path.join(kitti360Path, 'data_3d_semantics', sequence, "static")
            plys = glob.glob(os.path.join(label3DSemanticsPaths, "*.ply"))
            for ply in plys:
                xyz, rgb, lbl, iid = load_points(ply)
                # print(np.unique(lbl))
                uv, d = camera.project_vertices(xyz, frame)
                mask = np.logical_and(np.logical_and(d > 0, uv[0] > 0), uv[1] > 0)
                mask = np.logical_and(np.logical_and(mask, uv[0] < image.shape[1]), uv[1] < image.shape[0])
                uv, d = (uv[0][mask], uv[1][mask]), d[mask]
                for i in range(len(d)):
                    if d[i] > 0 and d[i] < depth_buffer[uv[1][i], uv[0][i]]:
                        depth_buffer[uv[1][i], uv[0][i]] = d[i]

            for ply in plys:
                xyz, rgb, lbl, iid = load_points(ply)
                # mask out lbl!= "building"
                mask = lbl == 11
                xyz = xyz[mask]
                uv, d = camera.project_vertices(xyz, frame)
                mask = np.logical_and(np.logical_and(d > 0, uv[0] > 0), uv[1] > 0)
                mask = np.logical_and(np.logical_and(mask, uv[0] < image.shape[1]), uv[1] < image.shape[0])
                uv, d = (uv[0][mask], uv[1][mask]), d[mask]
                mask = [d[i] <= depth_buffer[int(uv[1][i]), int(uv[0][i])] for i in range(len(d))]
                # print(uv[0][mask].shape, uv[1][mask].shape)
                # print(np.array([uv[0][mask], uv[1][mask]]).shape)

                combined_data = np.column_stack((uv[0][mask], uv[1][mask]))
                print(combined_data.shape)
                if combined_data.shape[0] > 0:
                    # Normalize the combined data
                    # It's important to scale the color information to be on a similar numerical range as the UV coordinates
                    # Apply DBSCAN
                    # The 'eps' and 'min_samples' parameters are crucial and should be tuned based on the specifics of the dataset
                    dbscan = DBSCAN(eps=5, min_samples=10)
                    clusters = dbscan.fit_predict(combined_data)

                    # Determine the densest cluster
                    cluster_counts = Counter(clusters)
                    # Exclude noise points which are labeled as -1
                    if -1 in cluster_counts:
                        del cluster_counts[-1]
                    densest_cluster_id = cluster_counts.most_common(1)[0][0]

                    # Get the indices of the points in the densest cluster
                    densest_cluster_indices = np.where(clusters == densest_cluster_id)[0]
                    densest_cluster_points_uv = uv[:, mask][:, densest_cluster_indices]

                    # Visualize the clustering results (optional)
                    for i, point in enumerate(densest_cluster_points_uv.T):
                        plt.plot(point[0], point[1], '.', markersize=1)

                    plt.imshow(np_image2D)
                    plt.show()

                plt.plot(uv[0][mask], uv[1][mask], 'r.')
                # break


        plt.pause(0.5)
        plt.clf()
