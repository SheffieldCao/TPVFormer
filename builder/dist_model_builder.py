# import random
# import warnings

import numpy as np
import torch
# import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
# from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
#                          Fp16OptimizerHook, OptimizerHook, build_optimizer,
#                          build_runner, get_dist_info)
# from mmcv.utils import build_from_cfg

# from mmdet.core import EvalHook

# from mmdet.datasets import (build_dataset,
#                             replace_ImageToTensor)
# from mmdet.utils import get_root_logger
# import time
# import os.path as osp


def custom_load_model2gpu(model,
                            cfg,
                            distributed=False):
    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    
    return model