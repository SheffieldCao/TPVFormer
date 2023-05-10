import os
import numpy as np
import torch
import numba as nb
from torch.utils import data
from dataloader.transform_3d import PadMultiViewImage, NormalizeMultiviewImage, \
    PhotoMetricDistortionMultiViewImage, RandomScaleImageMultiViewImage

from mmcv.image import imresize

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)


def load_occ_from_file(occ_gt_path):
    results = {}
    occ_gt_path = os.path.join(occ_gt_path, "labels.npz")

    occ_labels = np.load(occ_gt_path)
    semantics = occ_labels['semantics']
    mask_lidar = occ_labels['mask_lidar'].astype(bool)
    mask_camera = occ_labels['mask_camera'].astype(bool)

    return semantics, mask_lidar, mask_camera

class DatasetWrapper_NuScenes(data.Dataset):
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    AttrMapping = {
        'cycle.with_rider': 0,
        'cycle.without_rider': 1,
        'pedestrian.moving': 2,
        'pedestrian.standing': 3,
        'pedestrian.sitting_lying_down': 4,
        'vehicle.moving': 5,
        'vehicle.parked': 6,
        'vehicle.stopped': 7,
    }
    AttrMapping_rev = [
        'cycle.with_rider',
        'cycle.without_rider',
        'pedestrian.moving',
        'pedestrian.standing',
        'pedestrian.sitting_lying_down',
        'vehicle.moving',
        'vehicle.parked',
        'vehicle.stopped',
    ]
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')

    def __init__(self, in_dataset, grid_size, fill_label=0,
                 fixed_volume_space=False, max_volume_space=[51.2, 51.2, 3], 
                 min_volume_space=[-51.2, -51.2, -5], phase='train', scale_rate=1,
                 input_size=(432, 768), src_size=(900, 1600)):
        'Initialization'
        self.imagepoint_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.fill_label = fill_label
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.input_h, self.input_w = input_size
        self.src_h, self.src_w = src_size

        if scale_rate != 1:
            if phase == 'train':
                transforms = [
                    PhotoMetricDistortionMultiViewImage(),
                    NormalizeMultiviewImage(**img_norm_cfg),
                    RandomScaleImageMultiViewImage([scale_rate]),
                    PadMultiViewImage(size_divisor=32)
                ]
            else:
                transforms = [
                    NormalizeMultiviewImage(**img_norm_cfg),
                    RandomScaleImageMultiViewImage([scale_rate]),
                    PadMultiViewImage(size_divisor=32)
                ]
        else:
            if phase == 'train':
                transforms = [
                    PhotoMetricDistortionMultiViewImage(),
                    NormalizeMultiviewImage(**img_norm_cfg),
                    PadMultiViewImage(size_divisor=32)
                ]
            else:
                transforms = [
                    NormalizeMultiviewImage(**img_norm_cfg),
                    PadMultiViewImage(size_divisor=32)
                ]
        self.transforms = transforms

    def __len__(self):
        return len(self.imagepoint_dataset)

    def __getitem__(self, index):
        data = self.imagepoint_dataset[index]

        # only occ 20230509
        # imgs, img_metas, xyz, labels = data
        imgs, img_metas = data
        
        # sample and resize to input size 20230509
        imgs = [imresize(img, (self.input_w, self.input_h)) for img in imgs]

        # deal with img augmentations
        imgs_dict = {'img': imgs, 'lidar2img': img_metas['lidar2img']}
        for t in self.transforms:
            imgs_dict = t(imgs_dict)
        imgs = imgs_dict['img']
        imgs = [img.transpose(2, 0, 1) for img in imgs]

        img_metas['img_shape'] = imgs_dict['img_shape']
        img_metas['lidar2img'] = imgs_dict['lidar2img']

        # assert self.fixed_volume_space
        # max_bound = np.asarray(self.max_volume_space)  # 51.2 51.2 3
        # min_bound = np.asarray(self.min_volume_space)  # -51.2 -51.2 -5
        # # get grid index
        # crop_range = max_bound - min_bound
        # cur_grid_size = self.grid_size                 # 200, 200, 16
        # # TODO: intervals should not minus one.
        # intervals = crop_range / (cur_grid_size - 1)   # voxel size

        # if (intervals == 0).any(): 
        #     print("Zero interval!")
        # # TODO: grid_ind_float should actually be returned.
        # # grid_ind_float = (np.clip(xyz, min_bound, max_bound - 1e-3) - min_bound) / intervals
        # # point cloud coords in voxel scale
        # grid_ind_float = (np.clip(xyz, min_bound, max_bound) - min_bound) / intervals
        # # point cloud coords (int) in voxel scale with origin as (-50,-50,-1)
        # grid_ind = np.floor(grid_ind_float).astype(np.int)   

        # # process labels
        # processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.fill_label
        # label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        # label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        # # sorted voxel label pair with priority: xyz
        # processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)         # n, 3+1
        occ_path = self.imagepoint_dataset.nusc_infos[index]['occ_path']
        processed_label = load_occ_from_file(occ_path)
        semantics, mask_lidar, mask_cam = processed_label
        # data_tuple = (imgs, img_metas, semantics, mask_lidar, mask_cam, grid_ind, labels)
        data_tuple = (imgs, img_metas, semantics, mask_cam)

        # data_tuple += (grid_ind, labels)

        return data_tuple


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


def custom_collate_fn(data):
    img2stack = np.stack([d[0] for d in data]).astype(np.float32)
    meta2stack = [d[1] for d in data]
    label2stack = np.stack([d[2] for d in data]).astype(np.int)
    # because we use a batch size of 1, so we can stack these tensor together.
    # grid_ind_stack = np.stack([d[3] for d in data]).astype(np.float)
    mask_stack = np.stack([d[3] for d in data]).astype(np.bool)
    # point_label = np.stack([d[4] for d in data]).astype(np.int)
    return torch.from_numpy(img2stack), \
        meta2stack, \
        torch.from_numpy(label2stack), \
        torch.from_numpy(mask_stack)
        # torch.from_numpy(point_label)

class EvalDataset_NuScenes(DatasetWrapper_NuScenes):
    '''Wrapper to get images and corresponding tokens for evaluation.
    '''
    def __init__(self, *args, **kwargs):
        super(EvalDataset_NuScenes, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        data = self.imagepoint_dataset[index]

        items = super(EvalDataset_NuScenes, self).__getitem__(index)

        return NotImplementedError