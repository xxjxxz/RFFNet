from torchvision.ops import DeformConv2d
import torch
from torch import nn
from torch.nn import functional as F
import logging
import torch


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, norm_cfg=None, act_cfg=None):
        """
        Initialize the ConvModule.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            padding (int): Padding to apply.
            norm_cfg (dict or None): Normalization configuration (e.g., {'type': 'BN'} for BatchNorm).
            act_cfg (dict or None): Activation configuration (e.g., {'type': 'ReLU', 'inplace': True}).
        """
        super(ConvModule, self).__init__()

        # Convolution layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=norm_cfg is None  # Bias is unnecessary if normalization follows
        )

        # Normalization layer
        if norm_cfg is not None:
            norm_type = norm_cfg.get('type', 'BN')
            if norm_type == 'BN':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm_type == 'GN':
                num_groups = norm_cfg.get('num_groups', 32)
                self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
            else:
                raise ValueError(f"Unsupported normalization type: {norm_type}")
        else:
            self.norm = None

        # Activation layer
        if act_cfg is not None:
            act_type = act_cfg.get('type', 'ReLU')
            if act_type == 'ReLU':
                self.activation = nn.ReLU(inplace=act_cfg.get('inplace', True))
            elif act_type == 'LeakyReLU':
                self.activation = nn.LeakyReLU(negative_slope=act_cfg.get('negative_slope', 0.01),
                                               inplace=act_cfg.get('inplace', True))
            elif act_type == 'Sigmoid':
                self.activation = nn.Sigmoid()
            else:
                raise ValueError(f"Unsupported activation type: {act_type}")
        else:
            self.activation = None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class FeatureAggregator(nn.Module):
    def __init__(self,
                 num_convs=1,
                 channels=256,
                 kernel_size=3,
                 norm_cfg=None,
                 activation_cfg=dict(type='ReLU')):
        super(FeatureAggregator, self).__init__()
        assert num_convs > 0, 'The number of convs must be greater than 0.'

        self.embedding_convs = nn.ModuleList()
        for i in range(num_convs):
            if i == num_convs - 1:
                norm_cfg_final = None
                activation_cfg_final = None
            else:
                norm_cfg_final = norm_cfg
                activation_cfg_final = activation_cfg
            self.embedding_convs.append(
                ConvModule(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                    norm_cfg=norm_cfg_final,
                    act_cfg=activation_cfg_final))

        self.feature_fusions = nn.ModuleList()
        self.feature_fusions.append(
             ConvModule(
                 in_channels=channels * 8,
                 out_channels=channels * 4,
                 kernel_size=1,
                 padding=0,
                 norm_cfg=norm_cfg,
                 act_cfg=activation_cfg))
        self.feature_fusions.append(
             ConvModule(
                 in_channels=channels * 4,
                 out_channels=channels * 2,
                 kernel_size=3,
                 padding=1,
                 norm_cfg=norm_cfg,
                 act_cfg=activation_cfg))
        self.feature_fusions.append(
             ConvModule(
                 in_channels=channels * 2,
                 out_channels=channels,
                 kernel_size=1,
                 padding=0,
                 norm_cfg=None,
                 act_cfg=None))

    def forward(self, x, ref_x):
        """Aggregate reference feature maps `ref_x`.

        The aggregation mainly contains two steps:
        1. Computing the cosine similarity between `x` and `ref_x`.
        2. Use the normalized (i.e. softmax) cosine similarity to weightedly sum `ref_x`.

        Args:
            x (Tensor): of shape [1, C, H, W]
            ref_x (Tensor): of shape [N, C, H, W]. N is the number of reference feature maps.

        Returns:
            Tensor: The aggregated feature map with shape [1, C, H, W].
        """
        assert len(x.shape) == 4 and x.shape[0] == 1, "Only support 'batch_size == 1' for x"

        x_embedded = x
        for embed_conv in self.embedding_convs:
            x_embedded = embed_conv(x_embedded)
        x_embedded = x_embedded / x_embedded.norm(p=2, dim=1, keepdim=True)

        ref_x_embedded = ref_x
        for embed_conv in self.embedding_convs:
            ref_x_embedded = embed_conv(ref_x_embedded)
        ref_x_embedded = ref_x_embedded / ref_x_embedded.norm(p=2, dim=1, keepdim=True)

        fusion_input = torch.cat((x_embedded.repeat(ref_x_embedded.shape[0], 1, 1, 1),
                                  ref_x_embedded,
                                  x_embedded.repeat(ref_x_embedded.shape[0], 1, 1, 1) - ref_x_embedded,
                                  x.repeat(ref_x_embedded.shape[0], 1, 1, 1),
                                  ref_x,
                                  x.repeat(ref_x_embedded.shape[0], 1, 1, 1) - ref_x,
                                  -x_embedded.repeat(ref_x_embedded.shape[0], 1, 1, 1) + ref_x_embedded,
                                  -x.repeat(ref_x_embedded.shape[0], 1, 1, 1) + ref_x),
                                 dim=1)

        for feature_fusion in self.feature_fusions:
            fusion_input = feature_fusion(fusion_input)

        adaptive_weights = fusion_input
        adaptive_weights = adaptive_weights.softmax(dim=0)
        aggregated_feature_map = torch.sum(ref_x * adaptive_weights, dim=0, keepdim=True)

        return aggregated_feature_map


class GetAlignedFeature(nn.Module):
    def __init__(self, in_dim):
        super(GetAlignedFeature, self).__init__()

        inter_dim = 2  # --------------------------- 1

        self.al_conv_one = nn.Conv2d(in_dim, inter_dim, kernel_size=3, stride=1, padding=1)
        # nn.init.kaiming_uniform_(self.conv1.weight, a=1)
        self.defConv1 = DeformConv2d(384, 384, kernel_size=1, stride=1, padding=0, groups=1, bias=False)

        # self.al_conv_two = nn.Conv2d(in_dim, inter_dim, kernel_size=3, stride=1, padding=1)
        # nn.init.kaiming_uniform_(self.conv1.weight, a=1)
        # self.defConv2 = DeformConv2d(384, 384, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        # self.al_conv_two = nn.Conv2d(2048, inter_dim, kernel_size=3, stride=1, padding=1)
        # self.defConv2 = DeformConv2d(2048, 2048, kernel_size=3, stride = 1, padding = 1, groups = 1, bias = False)

        # self.al_conv_three = nn.Conv2d(2048, inter_dim, kernel_size=3, stride=1, padding=1)
        # self.defConv3 = DeformConv2d(2048, 2048, kernel_size=3, stride = 1, padding = 1, groups = 1, bias = False)

        # self.al_conv_four = nn.Conv2d(2048, inter_dim, kernel_size=1, stride=1, padding=0)
        # self.defConv4 = DeformConv2d(1024, 2048, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)

    def forward(self, f_m, f_p):
        concat_featmp = torch.cat([f_m, f_p], dim=1)
        # logger.info("concat_feart: {}".format(concat_feat.shape))

        aligned_feat_1_offset = self.al_conv_one(concat_featmp)  # [1,18,38,50]
        # aligned_feat_1 = self.defConv1(concat_feat, aligned_feat_1_offset)
        aligned_fp = self.defConv1(f_p, aligned_feat_1_offset)

        # aligned_feat_2_offset = self.al_conv_two(aligned_feat_1)
        # aligned_feat_2 = self.defConv2(aligned_feat_1, aligned_feat_2_offset)

        # aligned_feat_3_offset = self.al_conv_three(aligned_feat_2)
        # aligned_feat_3 = self.defConv3(aligned_feat_2, aligned_feat_3_offset)

        # aligned_feat_4_offset = self.al_conv_four(aligned_feat_1)
        # aligned_feat = self.defConv4(sup_feat, aligned_feat_4_offset)   # 注意是sup_feat, 利用得到的偏移，对支持支持帧进行可变性卷积，以配准特征

        return aligned_fp


if __name__ == '__main__':
    f1 = GetAlignedFeature(384*2)
    x1 = torch.randn(4, 384, 16, 16)
    x2 = torch.randn(4, 384, 16, 16)
    feat_local_res = f1(x1, x2)
    feat_local_res_list = torch.chunk(feat_local_res, 2, dim=1)

    x0 = torch.randn(1, 384, 16, 16)

    f2 = FeatureAggregator(num_convs=2, channels=384)
    feat_cur = f2(x0, torch.cat([feat_local_res_list[0], feat_local_res_list[1]], dim=0))
    feat_local_new = (feat_cur, feat_local_res_list[0], feat_local_res_list[1])
    feats_l_list = feat_local_new





