import copy
import os
import numpy as np
from tqdm import tqdm
from os import path as osp
import argparse
import torch
from torch.nn.functional import interpolate
import torch.distributed as dist

import mmcv
from mmcv import Config, DictAction
from mmcv.runner import (build_optimizer, get_dist_info, init_dist)
from mmcv.parallel import DataContainer as DC
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.utils import collect_env, get_root_logger
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

from evaluation.nus_occ_metrics import Metric_mIoU, Metric_FScore
from builder import data_builder, loss_builder
from builder.dist_model_builder import custom_load_model2gpu
from utils.load_save_util import revise_ckpt, revise_ckpt_2


def load_annotations(ann_file, load_interval=1):
    """Load annotations from ann_file.

    Args:
        ann_file (str): Path of the annotation file.

    Returns:
        list[dict]: List of annotations sorted by timestamps.
    """
    data = mmcv.load(ann_file)
    data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
    data_infos = data_infos[::load_interval]
    print(f"Loaded val annotation data: {data['metadata']}")
    return data_infos


def evaluate_miou(data_root, occ_results, eval_fscore=True, show_dir=None, **eval_kwargs):
    data_infos = load_annotations(osp.join(data_root, 'occ_infos_temporal_val.pkl'))
    if show_dir is not None:
        mmcv.mkdir_or_exist(show_dir)
        if not os.path.exists(show_dir):
            os.mkdir(show_dir)
        print('Saving output and gt in {} for visualization.'.format(show_dir))
        begin=eval_kwargs.get('begin',None)
        end=eval_kwargs.get('end',None)
    occ_eval_metrics = Metric_mIoU(
        num_classes=18,
        use_lidar_mask=False,
        use_image_mask=True)
    if eval_fscore:
        fscore_eval_metrics = Metric_FScore(
            leaf_size=10,
            threshold_acc=0.4,
            threshold_complete=0.4,
            voxel_size=[0.4, 0.4, 0.4],
            range=[-40, -40, -1, 40, 40, 5.4],
            void=[17, 255],
            use_lidar_mask=False,
            use_image_mask=True,
        )
    
    print('Starting Occ3d Evaluation...')
    for index, occ_pred in enumerate(tqdm(occ_results)):
        info = data_infos[index]

        occ_gt = np.load(os.path.join(data_root, info['occ_gt_path']))
        if show_dir is not None:
            if begin is not None and end is not None:
                if index>= begin and index<end:
                    sample_token = info['token']
                    save_path = os.path.join(show_dir,str(index).zfill(4))
                    np.savez_compressed(save_path, pred=occ_pred, gt=occ_gt, sample_token=sample_token)
            else:
                sample_token=info['token']
                save_path=os.path.join(show_dir,str(index).zfill(4))
                np.savez_compressed(save_path,pred=occ_pred,gt=occ_gt,sample_token=sample_token)


        # gt_semantics = occ_gt['semantics']
        # mask_lidar = occ_gt['mask_lidar'].astype(bool)
        # mask_camera = occ_gt['mask_camera'].astype(bool)
        # interpolate
        gt_semantics = interpolate(torch.from_numpy(occ_gt['semantics'].astype(np.float)).unsqueeze(0).unsqueeze(0), size=(100,100,8), mode='trilinear').squeeze(0).squeeze(0).numpy()
        mask_lidar = interpolate(torch.from_numpy(occ_gt['mask_lidar'].astype(np.float)).unsqueeze(0).unsqueeze(0), size=(100,100,8), mode='trilinear').squeeze(0).squeeze(0).numpy().astype(bool)
        mask_camera = interpolate(torch.from_numpy(occ_gt['mask_camera'].astype(np.float)).unsqueeze(0).unsqueeze(0), size=(100,100,8), mode='trilinear').squeeze(0).squeeze(0).numpy().astype(bool)
        # occ_pred = occ_pred
        occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)
        if eval_fscore:
            fscore_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)

    occ_eval_metrics.count_miou()
    if eval_fscore:
        fscore_eval_metrics.count_fscore()

def main():
    args = parse_args()
    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # build model
    if cfg.get('occupancy', False):
        from builder import tpv_occupancy_builder as model_builder
    else:
        from builder import tpv_lidarseg_builder as model_builder
    
    my_model = model_builder.build(cfg.model)
    my_model.init_weights()
    my_model = custom_load_model2gpu(my_model, cfg, distributed)
    # load model
    name = args.ckpt_path.split('/')[-1]
    ckpt_path = osp.join(cfg.work_dir, name)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    state_dict = revise_ckpt(state_dict)
    try:
        print(my_model.load_state_dict(state_dict, strict=False))
    except:
        state_dict = revise_ckpt_2(state_dict)
        print(my_model.load_state_dict(state_dict, strict=False))

    # build eval dataset
    dataset_config = cfg.dataset_params
    ignore_label = dataset_config['ignore_label']
    version = dataset_config['version']
    val_dataloader_config = cfg.val_data_loader
    grid_size = cfg.grid_size
    val_dataset_loader = \
        data_builder.build_only_val_dataloader(
            dataset_config,
            val_dataloader_config,
            grid_size=grid_size,
            version=version,
            dist=distributed,
            scale_rate=cfg.get('scale_rate', 1)
        )

    # reference loss function
    loss_func, lovasz_softmax = \
        loss_builder.build(ignore_label=ignore_label)
    # eval
    my_model.eval()
    val_loss_list = []
    val_vox_pred_list = []

    with torch.no_grad():
        for i_iter_val, (imgs, img_metas, val_vox_label, val_grid, val_pt_labs) in enumerate(val_dataset_loader):
            
            imgs = imgs.cuda()
            val_grid_float = val_grid.to(torch.float32).cuda()
            vox_label = val_vox_label.cuda()
            val_pt_labs = val_pt_labs.cuda()

            predict_labels_vox, predict_labels_pts = my_model(img=imgs, img_metas=img_metas, points=val_grid_float)

            # size: B, Cls, H, W, V = [1, 18, 100, 100, 8]
            # val_vox_pred_list.append(predict_labels_vox)

            if cfg.lovasz_input == 'voxel':
                lovasz_input = predict_labels_vox
                lovasz_label = vox_label
            else:
                lovasz_input = predict_labels_pts
                lovasz_label = val_pt_labs
                
            if cfg.ce_input == 'voxel':
                ce_input = predict_labels_vox
                ce_label = vox_label
            else:
                ce_input = predict_labels_pts.squeeze(-1).squeeze(-1)
                ce_label = val_pt_labs.squeeze(-1)
            
            loss = lovasz_softmax(
                torch.nn.functional.softmax(lovasz_input, dim=1).detach(), 
                lovasz_label, ignore=ignore_label
            ) + loss_func(ce_input.detach(), ce_label)
            
            # size: Cls, H, W, V = [2, 18, 100, 100, 8]
            predict_labels_vox = interpolate(predict_labels_vox, scale_factor=(2,2,2), mode='trilinear')
            predict_labels_vox = torch.argmax(predict_labels_vox, dim=1)
            # size: B, H, W, V = [2, 200, 200, 16]
            predict_labels_vox = predict_labels_vox.detach().cpu().numpy()[0]
            val_vox_pred_list.append(predict_labels_vox)

            val_loss_list.append(loss.detach().cpu().numpy())
            if i_iter_val % 100 == 0 and dist.get_rank() == 0:
                print('[EVAL] Iter %5d: Loss: %.3f (%.3f)'%(
                    i_iter_val, loss.item(), np.mean(val_loss_list)))

        evaluate_miou("data/nuscenes", val_vox_pred_list, osp.join(cfg.work_dir, f"val_{name.split('.')[0]}_show"))


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path', default='config/tpv04_occupancy.py')
    parser.add_argument('--work-dir', help='the dir to save logs and models', default=None)
    parser.add_argument('--ckpt-path', help='the dir to save logs and models', default=None)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

if __name__ == '__main__':
    main()