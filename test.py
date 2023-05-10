import os, time, argparse, os.path as osp, numpy as np
from tqdm import tqdm
import cv2
import torch
from torch.nn.functional import interpolate
import torch.distributed as dist

from utils.metric_util import MeanIoU, Metric_FScore, Metric_mIoU
from utils.load_save_util import revise_ckpt, revise_ckpt_2
from dataloader.dataset import get_nuScenes_label_name
from builder import loss_builder
from builder.dist_model_builder import custom_load_model2gpu

import mmcv
from mmcv import Config, DictAction
# from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (build_optimizer, get_dist_info, init_dist)
# from mmseg.utils import get_root_logger
from mmseg import __version__ as mmseg_version
from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from timm.scheduler import CosineLRScheduler

import warnings
warnings.filterwarnings("ignore")



colors_map = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
        [0, 0, 0, 0],  # 16 undefined
        [0, 0, 0, 0],  # 16 undefined
    ])

def pass_print(*args, **kwargs):
    pass

def vis_occ(semantics):
    # simple visualization of result in BEV
    semantics_valid = np.logical_not(semantics == 17)
    d = np.arange(16).reshape(1, 1, 16)
    d = np.repeat(d, 200, axis=0)
    d = np.repeat(d, 200, axis=1).astype(np.float32)
    d = d * semantics_valid
    selected = np.argmax(d, axis=2)

    selected_torch = torch.from_numpy(selected)
    semantics_torch = torch.from_numpy(semantics)

    occ_bev_torch = torch.gather(semantics_torch, dim=2,
                                    index=selected_torch.unsqueeze(-1))
    occ_bev = occ_bev_torch.numpy()

    occ_bev = occ_bev.flatten().astype(np.int32)
    occ_bev_vis = colors_map[occ_bev].astype(np.uint8)
    occ_bev_vis = occ_bev_vis.reshape(200, 200, 4)[::-1, ::-1, :3]
    occ_bev_vis = cv2.resize(occ_bev_vis,(400,400))
    return occ_bev_vis

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path', default='config/tpv04_occupancy.py')
    parser.add_argument('--work-dir', help='the dir to save logs and models', default=None)
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
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

def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

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
    
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    dataset_config = cfg.dataset_params
    ignore_label = dataset_config['ignore_label']
    version = dataset_config['version']
    train_dataloader_config = cfg.train_data_loader
    val_dataloader_config = cfg.val_data_loader

    max_num_epochs = cfg.max_epochs
    grid_size = cfg.grid_size

    if dist.get_rank() != 0:
        import builtins
        builtins.print = pass_print

    # configure logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed

    # build model
    if cfg.get('occupancy', False):
        from builder import tpv_occupancy_builder as model_builder
    else:
        from builder import tpv_lidarseg_builder as model_builder
    
    my_model = model_builder.build(cfg.model)
    my_model.init_weights()
    my_model = custom_load_model2gpu(my_model, cfg, distributed)

    # # generate datasets
    # SemKITTI_label_name = get_nuScenes_label_name(dataset_config["label_mapping"])
    # unique_label = np.asarray(cfg.unique_label)
    # unique_label_str = [SemKITTI_label_name[x] for x in unique_label]

    # from builder import data_builder
    # _, val_dataset_loader = \
    #     data_builder.build(
    #         dataset_config,
    #         train_dataloader_config,
    #         val_dataloader_config,
    #         grid_size=grid_size,
    #         version=version,
    #         dist=distributed,
    #         scale_rate=cfg.get('scale_rate', 1)
    #     )

    # if cfg.checkpoint_config is not None:
    #     # save mmdet version, config file content and class names in
    #     # checkpoints as meta data
    #     cfg.checkpoint_config.meta = dict(
    #         mmdet_version=mmdet_version,
    #         mmseg_version=mmseg_version,
    #         mmdet3d_version=mmdet3d_version,
    #         config=cfg.pretty_text)
        
    # # get optimizer, loss, scheduler
    # loss_func, lovasz_softmax = \
    #     loss_builder.build(ignore_label=ignore_label)
    
    # CalMeanIou_vox = MeanIoU(unique_label, ignore_label, unique_label_str, 'vox')
    # CalMeanIou_pts = MeanIoU(unique_label, ignore_label, unique_label_str, 'pts')
    
    # OccChallengeIoU = Metric_mIoU(num_classes=18, use_lidar_mask=False, use_image_mask=True)
    # # OccChallengeFScore = Metric_FScore(num_classes=18, use_lidar_mask=False, use_image_mask=True)
    
    # # resume and load       
    # epoch = 0
    # best_val_miou_pts, best_val_miou_vox = 0, 0

    cfg.resume_from = ''
    if osp.exists(osp.join(cfg.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(cfg.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    print('resume from: ', cfg.resume_from)
    print('work dir: ', cfg.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(my_model.load_state_dict(revise_ckpt(ckpt['state_dict']), strict=False))
        epoch = ckpt['epoch']
        if 'best_val_miou_pts' in ckpt:
            best_val_miou_pts = ckpt['best_val_miou_pts']
        if 'best_val_miou_vox' in ckpt:
            best_val_miou_vox = ckpt['best_val_miou_vox']
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
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
    return None

    # training
    print_freq = cfg.print_freq
    show_dir = osp.join(cfg.work_dir, "show_dir")
    mmcv.mkdir_or_exist(show_dir)

    # eval
    my_model.eval()
    val_loss_list = []
    CalMeanIou_pts.reset()
    CalMeanIou_vox.reset()

    with torch.no_grad():
        for i_iter_val, (imgs, img_metas, semantics, val_grid, mask_cam) in enumerate(val_dataset_loader):
            # torch.Size([1, 200, 200, 16])
            # torch.Size([1, 34720, 3])
            # torch.Size([1, 200, 200, 16])
            # print(semantics.size())
            # print(val_grid.size())
            # print(mask_cam.size())
            imgs = imgs.cuda()
            val_grid_float = val_grid.to(torch.float32).cuda()
            val_grid_int = val_grid.to(torch.long).cuda()
            # vox_label = semantics.cuda()
            # val_pt_labs = val_pt_labs.cuda()

            predict_labels_vox, predict_labels_pts = my_model(img=imgs, img_metas=img_metas, points=val_grid_float)
            # if cfg.lovasz_input == 'voxel':
            #     lovasz_input = predict_labels_vox
            #     lovasz_label = vox_label
            # else:
            #     lovasz_input = predict_labels_pts
            #     lovasz_label = val_pt_labs
                
            # if cfg.ce_input == 'voxel':
            #     ce_input = predict_labels_vox
            #     ce_label = vox_label
            # else:
            #     ce_input = predict_labels_pts.squeeze(-1).squeeze(-1)
            #     ce_label = val_pt_labs.squeeze(-1)
            
            # loss = lovasz_softmax(
            #     torch.nn.functional.softmax(lovasz_input, dim=1).detach(), 
            #     lovasz_label, ignore=ignore_label
            # ) + loss_func(ce_input.detach(), ce_label)
            
            # predict_labels_pts = predict_labels_pts.squeeze(-1).squeeze(-1)
            # predict_labels_pts = torch.argmax(predict_labels_pts, dim=1) # bs, n
            # predict_labels_pts = predict_labels_pts.detach().cpu()
            # val_pt_labs = val_pt_labs.squeeze(-1).cpu()
            predict_labels_vox = interpolate(predict_labels_vox, scale_factor=(2,2,2), mode='trilinear')
            predict_labels_vox = torch.argmax(predict_labels_vox, dim=1)
            predict_labels_vox = predict_labels_vox.detach().cpu()[0]  # W, H, Z

            # rotate 90 deg
            predict_labels_vox = torch.rot90(predict_labels_vox, -1, (0, 1)).numpy()

            semantics = semantics.detach().cpu().numpy()[0]
            mask_camera = mask_cam.detach().cpu().numpy()[0].astype(bool)
            # mask_lidar = mask_lidar.detach().cpu().numpy()[0].astype(bool)
            # print(mask_camera.dtype, mask_camera.shape, predict_labels_vox.shape) bool (200, 200, 16) (200, 200, 16)

            OccChallengeIoU.add_batch(predict_labels_vox, semantics_gt=semantics, mask_lidar=None, mask_camera=mask_camera)

            # for count in range(len(val_grid_int)):
            #     CalMeanIou_pts._after_step(predict_labels_pts[count], val_pt_labs[count])
            #     CalMeanIou_vox._after_step(
            #         predict_labels_vox[
            #         count, 
            #         val_grid_int[count][:, 0], 
            #         val_grid_int[count][:, 1], 
            #         val_grid_int[count][:, 2]].flatten(),
            #         val_pt_labs[count])
            # val_loss_list.append(loss.detach().cpu().numpy())
            if i_iter_val % print_freq == 0 and dist.get_rank() == 0:
                logger.info('[EVAL] Epoch %d Iter %5d'%(
                    epoch, i_iter_val))
                gt_vis = vis_occ(semantics)
                pred_vis = vis_occ(predict_labels_vox)
                mmcv.imwrite(np.concatenate([gt_vis, pred_vis], axis=1),
                            os.path.join(show_dir, "%d.jpg"%i_iter_val))
                
    OccChallengeIoU.count_miou()
    # val_miou_pts = CalMeanIou_pts._after_epoch()
    # val_miou_vox = CalMeanIou_vox._after_epoch()

    # if best_val_miou_pts < val_miou_pts:
    #     best_val_miou_pts = val_miou_pts
    # if best_val_miou_vox < val_miou_vox:
    #     best_val_miou_vox = val_miou_vox

    # logger.info('Current val miou pts is %.3f while the best val miou pts is %.3f' %
    #         (val_miou_pts, best_val_miou_pts))
    # logger.info('Current val miou vox is %.3f while the best val miou vox is %.3f' %
    #         (val_miou_vox, best_val_miou_vox))
    # logger.info('Current val loss is %.3f' %
    #         (np.mean(val_loss_list)))
        

if __name__ == '__main__':

    main()
