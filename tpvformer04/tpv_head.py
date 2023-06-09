
import torch
import torch.nn as nn
from torch.nn.init import normal_

from mmcv.runner import BaseModule
from mmseg.models import HEADS, builder
from mmcv.cnn.bricks.transformer import build_positional_encoding, \
    build_transformer_layer_sequence, build_transformer_layer
from mmcv.runner import force_fp32, auto_fp16

from .modules.cross_view_hybrid_attention import TPVCrossViewHybridAttention
from .modules.image_cross_attention import TPVMSDeformableAttention3D


@HEADS.register_module()
class TPVFormerHead(BaseModule):

    def __init__(self,
                 positional_encoding=None,
                 tpv_h=30,
                 tpv_w=30,
                 tpv_z=30,
                 pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
                 num_feature_levels=4,
                 num_cams=6,
                 encoder=None,
                 embed_dims=256,
                 **kwargs):
        super().__init__()

        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.real_z = self.pc_range[5] - self.pc_range[2]
        self.fp16_enabled = False

        # positional encoding
        self.positional_encoding = build_positional_encoding(positional_encoding)
        tpv_mask_hw = torch.zeros(1, tpv_h, tpv_w)
        self.register_buffer('tpv_mask_hw', tpv_mask_hw)

        # transformer layers
        self.encoder = build_transformer_layer_sequence(encoder)
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.tpv_embedding_hw = nn.Embedding(self.tpv_h * self.tpv_w, self.embed_dims)
        self.tpv_embedding_zh = nn.Embedding(self.tpv_z * self.tpv_h, self.embed_dims)
        self.tpv_embedding_wz = nn.Embedding(self.tpv_w * self.tpv_z, self.embed_dims)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, TPVMSDeformableAttention3D) or isinstance(m, TPVCrossViewHybridAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        """
        bs = mlvl_feats[0].shape[0]
        dtype = mlvl_feats[0].dtype
        device = mlvl_feats[0].device

        # tpv queries and pos embeds
        tpv_queries_hw = self.tpv_embedding_hw.weight.to(dtype)
        tpv_queries_zh = self.tpv_embedding_zh.weight.to(dtype)
        tpv_queries_wz = self.tpv_embedding_wz.weight.to(dtype)
        tpv_queries_hw = tpv_queries_hw.unsqueeze(0).repeat(bs, 1, 1)
        tpv_queries_zh = tpv_queries_zh.unsqueeze(0).repeat(bs, 1, 1)
        tpv_queries_wz = tpv_queries_wz.unsqueeze(0).repeat(bs, 1, 1)
        tpv_mask_hw = self.tpv_mask_hw.expand(bs, -1, -1)
        tpv_pos_hw = self.positional_encoding(tpv_mask_hw).to(dtype)
        tpv_pos_hw = tpv_pos_hw.flatten(2).transpose(1, 2)
        
        # flatten image features of different scales
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2) # num_cam, bs, hw, c
            feat = feat + self.cams_embeds[:, None, None, :].to(dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl+1, :].to(dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2) # num_cam, bs, hw++, c
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        tpv_embed = self.encoder(
            [tpv_queries_hw, tpv_queries_zh, tpv_queries_wz],
            feat_flatten,
            feat_flatten,
            tpv_h=self.tpv_h,
            tpv_w=self.tpv_w,
            tpv_z=self.tpv_z,
            tpv_pos=[tpv_pos_hw, None, None],
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            img_metas=img_metas,
        )
        
        return tpv_embed


@HEADS.register_module()
class MaskHead(BaseModule):
    def __init__(self,
                 in_dims,
                 hidden_dims=256,
                 tpv_h=30,
                 tpv_w=30,
                 tpv_z=30,
                 upsample_ratio=2):
        super(MaskHead, self).__init__()
        self.in_dims = in_dims
        self.upsample_ratio = upsample_ratio
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z

        self.hw_mask_head = nn.Sequential(
            nn.Conv2d(in_dims, hidden_dims, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dims, 9*upsample_ratio**2, 1, padding=0))

        self.zh_mask_head = nn.Sequential(
            nn.Conv2d(in_dims, hidden_dims, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dims, 9*upsample_ratio**2, 1, padding=0))

        self.wz_mask_head = nn.Sequential(
            nn.Conv2d(in_dims, hidden_dims, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dims, 9*upsample_ratio**2, 1, padding=0))

    def forward(self, intermediate_tpv_embed):
        tpv_hws, tpv_zhs, tpv_wzs = [], [], []
        for tpv_inter in intermediate_tpv_embed:
            tpv_inter_hw, tpv_inter_zh, tpv_inter_wz = tpv_inter
            bs, _, c = tpv_inter_hw.shape
            tpv_hws.append(tpv_inter_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w))
            tpv_zhs.append(tpv_inter_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h))
            tpv_wzs.append(tpv_inter_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z))
        tpv_hw = torch.cat(tpv_hws, dim=1)
        tpv_zh = torch.cat(tpv_zhs, dim=1)
        tpv_wz = torch.cat(tpv_wzs, dim=1)

        mask_hw = self.hw_mask_head(tpv_hw)
        mask_zh = self.zh_mask_head(tpv_zh)
        mask_wz = self.wz_mask_head(tpv_wz)

        return mask_hw, mask_zh, mask_wz

@HEADS.register_module()
class TPVFormerMaskHead(BaseModule):

    def __init__(self,
                 positional_encoding=None,
                 tpv_h=30,
                 tpv_w=30,
                 tpv_z=30,
                 pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
                 num_feature_levels=4,
                 num_cams=6,
                 encoder=None,
                 embed_dims=256,
                 **kwargs):
        super().__init__()

        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.real_z = self.pc_range[5] - self.pc_range[2]
        self.fp16_enabled = False

        # positional encoding
        self.positional_encoding = build_positional_encoding(positional_encoding)
        tpv_mask_hw = torch.zeros(1, tpv_h, tpv_w)
        self.register_buffer('tpv_mask_hw', tpv_mask_hw)

        # transformer layers
        self.encoder = build_transformer_layer_sequence(encoder)
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.tpv_embedding_hw = nn.Embedding(self.tpv_h * self.tpv_w, self.embed_dims)
        self.tpv_embedding_zh = nn.Embedding(self.tpv_z * self.tpv_h, self.embed_dims)
        self.tpv_embedding_wz = nn.Embedding(self.tpv_w * self.tpv_z, self.embed_dims)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, TPVMSDeformableAttention3D) or isinstance(m, TPVCrossViewHybridAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        """
        bs = mlvl_feats[0].shape[0]
        dtype = mlvl_feats[0].dtype
        device = mlvl_feats[0].device

        # tpv queries and pos embeds
        tpv_queries_hw = self.tpv_embedding_hw.weight.to(dtype)
        tpv_queries_zh = self.tpv_embedding_zh.weight.to(dtype)
        tpv_queries_wz = self.tpv_embedding_wz.weight.to(dtype)
        tpv_queries_hw = tpv_queries_hw.unsqueeze(0).repeat(bs, 1, 1)
        tpv_queries_zh = tpv_queries_zh.unsqueeze(0).repeat(bs, 1, 1)
        tpv_queries_wz = tpv_queries_wz.unsqueeze(0).repeat(bs, 1, 1)
        tpv_mask_hw = self.tpv_mask_hw.expand(bs, -1, -1)
        tpv_pos_hw = self.positional_encoding(tpv_mask_hw).to(dtype)
        tpv_pos_hw = tpv_pos_hw.flatten(2).transpose(1, 2)
        
        # flatten image features of different scales
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2) # num_cam, bs, hw, c
            feat = feat + self.cams_embeds[:, None, None, :].to(dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl+1, :].to(dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2) # num_cam, bs, hw++, c
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        tpv_embed = self.encoder(
            [tpv_queries_hw, tpv_queries_zh, tpv_queries_wz],
            feat_flatten,
            feat_flatten,
            tpv_h=self.tpv_h,
            tpv_w=self.tpv_w,
            tpv_z=self.tpv_z,
            tpv_pos=[tpv_pos_hw, None, None],
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            img_metas=img_metas,
        )
    
        return tpv_embed