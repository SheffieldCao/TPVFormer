_base_ = [
    './_base_/dataset.py',
    './_base_/optimizer.py',
    './_base_/schedule.py',
    './_base_/default_runtime.py',
]

# (4) runtime variables
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01
)

max_epochs = 24

# (2-1) TODO: PC range
dataset_is_occ = True
dataset_params = dict(
    version = "v1.0-trainval",
    ignore_label = 0,
    fill_label = 17,
    fixed_volume_space = True,
    label_mapping = "./config/label_mapping/nuscenes-noIgnore.yaml",
    max_volume_space = [50, 50, 5.4],
    min_volume_space = [-50, -50, -1],
    input_size = (512, 1408),
    src_size = (900, 1600),
)

train_data_loader = dict(
    data_path = "data/nuscenes/",
    imageset = "data/tpvocc-nuscenes_infos_train.pkl",
    batch_size = 1,
    shuffle = True,
    num_workers = 2,
)

val_data_loader = dict(
    data_path = "data/nuscenes/",
    imageset = "data/tpvocc-nuscenes_infos_val.pkl",
    batch_size = 1,
    shuffle = False,
    num_workers = 2,
)

dist_params = dict(backend='nccl')

occupancy = True

# (2-2) TODO: PC range
point_cloud_range = [-50, -50, -1, 50, 50, 5.4]

# (1) TODO: TPVFormer feature dim
_dim_ = 128

_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
_num_cams_ = 6

# (2-3) TODO: voxel range -> 200. 200. 16
tpv_h_ = 100
tpv_w_ = 100
tpv_z_ = 8
# TODOl: limited memory
upsample_ratio = 2
scale_h = upsample_ratio
scale_w = upsample_ratio
scale_z = upsample_ratio
grid_size = [tpv_h_*scale_h, tpv_w_*scale_w, tpv_z_*scale_z]

num_points_in_pillar = [4, 32, 32]
num_points = [8, 64, 64]
nbr_class = 18


# (5) change model aggregator
model = dict(
    type='TPVMaskFormer',
    use_grid_mask=True,
    loss_cfg=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0),
    mask_head=dict(
        type='MaskHead',
        in_dims=3*_dim_,
        hidden_dims=256,
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_),
    tpv_aggregator = dict(
        type='TPVAggregatorUpSample',
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_,
        nbr_classes=nbr_class,
        in_dims=_dim_,
        hidden_dims=2*_dim_,
        out_dims=_dim_,
        scale_h=scale_h,
        scale_w=scale_w,
        scale_z=scale_z
    ),
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), 
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    tpv_head=dict(
        type='TPVFormerMaskHead',
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_,
        pc_range=point_cloud_range,
        num_feature_levels=_num_levels_,
        num_cams=_num_cams_,
        embed_dims=_dim_,
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=tpv_h_,
            col_num_embed=tpv_w_),
        encoder=dict(
            type='TPVFormerEncoder',
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            num_layers=3,
            pc_range=point_cloud_range,
            num_points_in_pillar=num_points_in_pillar,
            return_intermediate=True,
            transformerlayers=dict(
                type='TPVFormerLayer',
                attn_cfgs=[
                    dict(
                        type='TPVCrossViewHybridAttention',
                        embed_dims=_dim_,
                        num_levels=1),
                    dict(
                        type='TPVImageCrossAttention',
                        pc_range=point_cloud_range,
                        deformable_attention=dict(
                            type='TPVMSDeformableAttention3D',
                            embed_dims=_dim_,
                            num_points=num_points,
                            num_z_anchors=num_points_in_pillar,
                            num_levels=_num_levels_,
                            floor_sampling_offset=False,
                            tpv_h=tpv_h_,
                            tpv_w=tpv_w_,
                            tpv_z=tpv_z_,
                        ),
                        embed_dims=_dim_,
                        tpv_h=tpv_h_,
                        tpv_w=tpv_w_,
                        tpv_z=tpv_z_,
                    )
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm')))),)
