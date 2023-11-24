""" GraspNet dataset processing.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image

import torch
# from torch._six import container_abcs
import collections.abc as container_abcs
from torch.utils.data import Dataset
from tqdm import tqdm
from multiprocessing import Pool

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from utils.data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image,\
                            get_workspace_mask, remove_invisible_grasp_points

class GraspNetDataset(Dataset):
    def __init__(self, root, valid_obj_idxs, grasp_labels, camera='kinect', split='train', num_points=20000,
                 remove_outlier=False, remove_invisible=True, augment=False, load_label=True):
        assert(num_points<=50000)
        self.root = root
        self.split = split
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.remove_invisible = remove_invisible
        self.valid_obj_idxs = valid_obj_idxs
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.collision_labels = {} #每个场景下的物体在自身坐标系下的所有抓取姿态是否碰撞的flag

        if split == 'train':
            self.sceneIds = list( range(100) )
        elif split == 'test':
            self.sceneIds = list( range(100,190) )
        elif split == 'test_seen':
            self.sceneIds = list( range(100,130) )
        elif split == 'test_similar':
            self.sceneIds = list( range(130,160) )
        elif split == 'test_novel':
            self.sceneIds = list( range(160,190) )
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]
        
        self.colorpath = [] #所有sceneid对应scene下的所有图片路径
        self.depthpath = []
        self.labelpath = [] #物体分割标签，每个像素点的值是物体的id
        self.metapath = []
        self.scenename = []
        self.frameid = [] #每个图片在scene下的id。0-255的循环
        for x in tqdm(self.sceneIds, desc = 'Loading data path and collision labels...'):
            for img_num in range(256):
                self.colorpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4)+'.png'))
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4)+'.png'))
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4)+'.png'))
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4)+'.mat'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)
            if self.load_label:
                collision_labels = np.load(os.path.join(root, 'collision_label', x.strip(),  'collision_labels.npz'))
                self.collision_labels[x.strip()] = {}
                for i in range(len(collision_labels)):
                    self.collision_labels[x.strip()][i] = collision_labels['arr_{}'.format(i)]

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.depthpath)

    def augment_data(self, point_clouds, object_poses_list):
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c,-s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)

        return point_clouds, object_poses_list

    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        else:
            return self.get_data(index)

    def get_data(self, index, return_raw_cloud=False):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        seg_mask = (seg > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]
        if return_raw_cloud:
            return cloud_masked, color_masked

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        
        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)

        return ret_dict

    def get_data_label(self, index):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0 #shape 720*1280*3
        depth = np.array(Image.open(self.depthpath[index])) #shape 720*1280
        seg = np.array(Image.open(self.labelpath[index])) #shape 720*1280
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses'] # shape 3*4*9
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True) #shape 720*1280*3

        # get valid points
        depth_mask = (depth > 0)
        seg_mask = (seg > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy')) #每一帧相机相对于第一帧的位姿变换矩阵4*4
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy')) #第一帧相对于桌面的位姿变换矩阵4*4
            trans = np.dot(align_mat, camera_poses[self.frameid[index]]) #当前帧相对于桌面的位姿变换矩阵4*4
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs] #shape self.num_points*3 xyz
        color_sampled = color_masked[idxs] #shape self.num_points*3 rgb
        seg_sampled = seg_masked[idxs] #shape self.num_points 0-88 object id
        objectness_label = seg_sampled.copy() #让objectness_label支持多个物体类别标签
        objectness_label[objectness_label>1] = 1 #shape self.num_points 0-1 1表示有物体
        
        object_poses_list = [] #每个元素是一个list，每个list里是一个物体的位姿变换矩阵4*4
        grasp_points_list = [] #每一个元素是一个list，每个list里是一个物体的所有抓取姿态的抓取点的坐标
        grasp_offsets_list = []
        grasp_scores_list = []
        grasp_tolerance_list = []
        for i, obj_idx in enumerate(obj_idxs):# 遍历该场景中的每个物体
            if obj_idx not in self.valid_obj_idxs:
                continue
            if (seg_sampled == obj_idx).sum() < 50: #场景中物体的点数太少，不考虑
                continue
            object_poses_list.append(poses[:, :, i])
            points, offsets, scores, tolerance = self.grasp_labels[obj_idx]
            collision = self.collision_labels[scene][i] #(Np, V, A, D)

            # remove invisible grasp points
            if self.remove_invisible:
                visible_mask = remove_invisible_grasp_points(cloud_sampled[seg_sampled==obj_idx], points, poses[:,:,i], th=0.01)
                points = points[visible_mask]
                offsets = offsets[visible_mask]
                scores = scores[visible_mask]
                tolerance = tolerance[visible_mask]
                collision = collision[visible_mask]

            idxs = np.random.choice(len(points), min(max(int(len(points)/4),300),len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_offsets_list.append(offsets[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)
            tolerance = tolerance[idxs].copy()
            tolerance[collision] = 0
            grasp_tolerance_list.append(tolerance)
        
        if self.augment:
            cloud_sampled, object_poses_list = self.augment_data(cloud_sampled, object_poses_list)
        
        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32) #shape self.num_points*3 xyz
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32) #shape self.num_points*3 rgb
        ret_dict['objectness_label'] = objectness_label.astype(np.int64) #shape self.num_points 0-1 1表示有物体
        ret_dict['object_poses_list'] = object_poses_list #每一个元素表示该场景中一个物体的所有抓取姿态的位姿变换矩阵4*4 #shape num_ojbects*3*4
        ret_dict['grasp_points_list'] = grasp_points_list #shape num_objects*300*3 #每一个元素表示该场景中一个物体的所有抓取姿态的抓取点的坐标 #每个物体有300个抓取姿态
        ret_dict['grasp_offsets_list'] = grasp_offsets_list #shape num_objects*300*300*12*4*3 #每一个元素表示该场景中一个物体的所有抓取姿态的抓取点的偏移量。 最后4个维度：视角，旋转，深度，宽度
        ret_dict['grasp_labels_list'] = grasp_scores_list #shape num_objects*300*300*12*4 #抓取分数
        ret_dict['grasp_tolerance_list'] = grasp_tolerance_list #shape num_objects*300*300*12*4 #抓取容差

        return ret_dict

def load_grasp_labels(root):
    '''
    加载88个物体，在每个物体坐标系下的每个抓取
    '''
    obj_names = list(range(88))
    valid_obj_idxs = []
    grasp_labels = {}
    for i, obj_name in enumerate(tqdm(obj_names, desc='Loading grasping labels...')):
        if i == 18: continue
        valid_obj_idxs.append(i + 1) #here align with label png
        label = np.load(os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3))))
        tolerance = np.load(os.path.join(root, 'tolerance', '{}_tolerance.npy'.format(str(i).zfill(3))))
        grasp_labels[i + 1] = (label['points'].astype(np.float32), label['offsets'].astype(np.float32),
                                label['scores'].astype(np.float32), tolerance)

    return valid_obj_idxs, grasp_labels

# def load_label(i, root):
#     if i == 18:
#         return None
#     label = np.load(os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3))))
#     tolerance = np.load(os.path.join(root, 'tolerance', '{}_tolerance.npy'.format(str(i).zfill(3))))
#     return i + 1, (label['points'].astype(np.float32), label['offsets'].astype(np.float32),
#                    label['scores'].astype(np.float32), tolerance)

# def load_grasp_labels(root):
#     '''
#     加载88个物体，在每个物体坐标系下的每个抓取
#     '''
#     obj_names = list(range(88))
#     with Pool() as p:
#         results = p.starmap(load_label, [(i, root) for i in obj_names])
#     valid_obj_idxs = [result[0] for result in results if result is not None]
#     grasp_labels = {result[0]: result[1] for result in results if result is not None}
#     return valid_obj_idxs, grasp_labels

def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key:collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))

if __name__ == "__main__":
    root = '/data/Benchmark/graspnet'
    valid_obj_idxs, grasp_labels = load_grasp_labels(root)
    train_dataset = GraspNetDataset(root, valid_obj_idxs, grasp_labels, split='train', remove_outlier=True, remove_invisible=True, num_points=20000)
    print(len(train_dataset))

    end_points = train_dataset[233]
    cloud = end_points['point_clouds']
    seg = end_points['objectness_label']
    print(cloud.shape)
    print(cloud.dtype)
    print(cloud[:,0].min(), cloud[:,0].max())
    print(cloud[:,1].min(), cloud[:,1].max())
    print(cloud[:,2].min(), cloud[:,2].max())
    print(seg.shape)
    print((seg>0).sum())
    print(seg.dtype)
    print(np.unique(seg))
