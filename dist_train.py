import os, time, argparse, os.path as osp, numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist

from utils.metric_util import MeanIoU, Metric_mIoU, Metric_FScore
from utils.load_save_util import revise_ckpt, revise_ckpt_2
from utils.lovasz_losses import loss_occ_single
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
from mmdet.models.builder import build_loss
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from timm.scheduler import CosineLRScheduler

import warnings
warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass

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

    # generate datasets
    SemKITTI_label_name = get_nuScenes_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(cfg.unique_label)
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label]

    from builder import data_builder
    train_dataset_loader, val_dataset_loader = \
        data_builder.build(
            dataset_config,
            train_dataloader_config,
            val_dataloader_config,
            grid_size=grid_size,
            version=version,
            dist=distributed,
            scale_rate=cfg.get('scale_rate', 1)
        )

    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text)
        
    # get optimizer, loss, scheduler
    optimizer = build_optimizer(my_model, cfg.optimizer)

    loss_func = build_loss(cfg.model.loss_cfg)

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=len(train_dataset_loader)*max_num_epochs,
        lr_min=1e-6,
        warmup_t=500,
        warmup_lr_init=1e-5,
        t_in_epochs=False
    )
    
    CalMeanIou_vox = Metric_mIoU(num_classes=cfg.nbr_class, use_image_mask=True)
    # CalFScore_vox = Metric_FScore(num_classes=cfg.nbr_class, use_image_mask=True)
    
    # resume and load       
    epoch = 0
    best_val_miou_pts, best_val_miou_vox = 0, 0
    global_iter = 0

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
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        if 'best_val_miou_pts' in ckpt:
            best_val_miou_pts = ckpt['best_val_miou_pts']
        if 'best_val_miou_vox' in ckpt:
            best_val_miou_vox = ckpt['best_val_miou_vox']
        global_iter = ckpt['global_iter']
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
        

    # training
    print_freq = cfg.print_freq

    while epoch < max_num_epochs:
        my_model.train()
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)
        loss_list = []
        time.sleep(10)
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, (imgs, img_metas, train_vox_label, mask_camera) in enumerate(train_dataset_loader):

            imgs = imgs.cuda()
            train_vox_label = train_vox_label.type(torch.LongTensor).cuda()
            mask_camera = mask_camera.cuda()
            # forward + backward + optimize
            data_time_e = time.time()
            outputs_vox = my_model(img=imgs, img_metas=img_metas)

            loss = loss_occ_single(loss_func, train_vox_label, outputs_vox, mask_camera=mask_camera, num_classes=cfg.nbr_class)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
            optimizer.step()
            loss_list.append(loss.item())
            scheduler.step_update(global_iter)
            time_e = time.time()

            global_iter += 1
            if i_iter % print_freq == 0 and dist.get_rank() == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info('[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f), grad_norm: %.1f, lr: %.7f, time: %.3f (%.3f)'%(
                    epoch, i_iter, len(train_dataset_loader), 
                    loss.item(), np.mean(loss_list), grad_norm, lr,
                    time_e - time_s, data_time_e - data_time_s
                ))
            data_time_s = time.time()
            time_s = time.time()
        
        # save checkpoint
        if dist.get_rank() == 0:
            dict_to_save = {
                'state_dict': my_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'global_iter': global_iter,
                'best_val_miou_pts': best_val_miou_pts,
                'best_val_miou_vox': best_val_miou_vox
            }
            save_file_name = os.path.join(os.path.abspath(cfg.work_dir), f'epoch_{epoch+1}.pth')
            torch.save(dict_to_save, save_file_name)
            dst_file = osp.join(cfg.work_dir, 'latest.pth')
            mmcv.symlink(save_file_name, dst_file)

        epoch += 1
        
        # eval
        my_model.eval()
        val_loss_list = []
        CalMeanIou_vox.reset()

        with torch.no_grad():
            for i_iter_val, (imgs, img_metas, val_vox_label, mask_camera) in enumerate(val_dataset_loader):
                
                imgs = imgs.cuda()
                val_vox_label = val_vox_label.cuda()
                mask_camera = mask_camera.cuda()

                predict_labels_vox = my_model(img=imgs, img_metas=img_metas)
                
                loss = loss_occ_single(loss_func, val_vox_label, predict_labels_vox, mask_camera=mask_camera, num_classes=cfg.nbr_class)
                
                predict_labels_vox = torch.argmax(predict_labels_vox, dim=1)
                predict_labels_vox = predict_labels_vox.detach().cpu().numpy()[0]

                # # rotate 90 deg
                # predict_labels_vox = torch.rot90(predict_labels_vox, -1, (0, 1)).numpy()

                val_vox_label = val_vox_label.detach().cpu().numpy()[0]
                mask_camera = mask_camera.detach().cpu().numpy()[0].astype(bool)

                val_loss_list.append(loss.detach().cpu().numpy())
                CalMeanIou_vox.add_batch(predict_labels_vox, semantics_gt=val_vox_label, mask_lidar=None, mask_camera=mask_camera)

                if i_iter_val % print_freq == 0 and dist.get_rank() == 0:
                    logger.info('[EVAL] Epoch %d Iter %5d: Loss: %.3f (%.3f)'%(
                        epoch, i_iter_val, loss.item(), np.mean(val_loss_list)))
        
        cls_names, mIoU, _ = CalMeanIou_vox.count_miou()
        val_miou_vox = round(np.nanmean(mIoU[:len(cls_names)-1]) * 100, 2)
        if best_val_miou_vox < val_miou_vox:
            best_val_miou_vox = val_miou_vox
            # save best checkpoint
            if dist.get_rank() == 0:
                mmcv.check_file_exist(save_file_name)
                best_file = osp.join(cfg.work_dir, 'best.pth')
                mmcv.symlink(save_file_name, best_file)

        logger.info('Current val miou vox is %.3f while the best val miou vox is %.3f' %
                (val_miou_vox, best_val_miou_vox))
        logger.info('Current val loss is %.3f' %
                (np.mean(val_loss_list)))
        

if __name__ == '__main__':    
    # ngpus = torch.cuda.device_count()
    # args.gpus = ngpus
    # torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)

    main()
