import random

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights, mobilenet_v3_small
import torch
import torch.nn as nn
import torch.nn.functional as F
from Networks.FFNet.ODConv2d import ODConv2d
from Networks.FFNet.deform_conv import GetAlignedFeature

__all__ = ['FFNet']
from Networks.FFNet.mam import MAM, RGBDiffFusion, PyramidFusion

# from VMamba.classification.models.vmamba import VSSBlock


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.Tensor.permute(x, self.dims)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


def conv(in_ch, out_ch, ks, stride):
    pad = (ks - 1) // 2
    stage = nn.Sequential(nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch, kernel_size=ks, stride=stride,
                                    padding=pad, bias=False),
                          LayerNorm2d((out_ch), eps=1e-06, elementwise_affine=True),
                          nn.GELU(approximate='none'))
    return stage


class ChannelAttention(nn.Module):
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        avgout = self.shared_MLP(x)
        return self.sigmoid(avgout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.conv1(x)
        return self.sigmoid(x)


class Backbone(nn.Module):
    def __init__(self, pretrained_weights_path=None):  # 添加参数以支持自定义权重路径
        super(Backbone, self).__init__()

        # 初始化模型的特征
        feats = list(convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).features.children())
        # feats = list(convnext_tiny().features.children())


        self.stem = nn.Sequential(*feats[0])
        self.stage1 = nn.Sequential(*feats[1])
        self.stage2 = nn.Sequential(*feats[2:4])
        self.stage3 = nn.Sequential(*feats[4:6])
        self.stage4 = nn.Sequential(*feats[6:])

        # 如果提供了权重路径，则加载权重
        if pretrained_weights_path is not None:
            self.load_pretrained_weights(pretrained_weights_path)

    def load_pretrained_weights(self, path):
        # 加载权重文件
        state_dict = torch.load(path)['model']

        # 检查并移除多余的键（如果有）
        state_dict1 = {k.replace('module.', ''): v for k, v in state_dict.items()}  # 处理分布式训练的权重

        # 将权重加载到模型
        self.load_state_dict(state_dict1)

    def forward(self, x):
        x = x.float()
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        feature1 = x
        x = self.stage3(x)
        feature2 = x
        x = self.stage4(x)

        return feature1, feature2, x


class ccsm(nn.Module):
    def __init__(self, channel, channel2, num_filters):
        super(ccsm, self).__init__()
        self.ch_att_s = ChannelAttention(channel)
        self.sa_s = SpatialAttention(7)
        self.conv1 = nn.Sequential(
            ODConv2d(channel, channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=channel))
        self.conv2 = nn.Sequential(
            ODConv2d(channel, channel2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=channel2))

        self.conv3 = nn.Sequential(
            ODConv2d(channel2, channel2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=channel2))
        self.conv4 = nn.Sequential(
            ODConv2d(channel2, num_filters, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=num_filters))

    def forward(self, x):
        x = self.ch_att_s(x) * x
        pool1 = x
        x = self.conv1(x)
        x = x + pool1
        x = self.conv2(x)
        pool2 = x
        x = self.conv3(x)
        x = x + pool2
        x = self.conv4(x)

        x = self.sa_s(x) * x

        return x


class CSM(nn.Module):
    def __init__(self, channel, channel2, num_filters):
        super(CSM, self).__init__()
        self.ch_att_s = ChannelAttention(channel)
        self.sa_s = SpatialAttention(7)
        self.conv1 = nn.Sequential(
            ODConv2d(channel, channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=channel))
        self.conv2 = nn.Sequential(
            ODConv2d(channel, channel2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=channel2))

        self.conv3 = nn.Sequential(
            ODConv2d(channel2, channel2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=channel2))
        self.conv4 = nn.Sequential(
            ODConv2d(channel2, num_filters, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=num_filters))

    def forward(self, x, x1):
        # 通道注意力模块
        x = self.ch_att_s(x) * x
        pool1 = x

        # 提取主要特征
        x = self.conv1(x)
        x = x + pool1
        x = self.conv2(x)

        # 使用帧差信息生成注意力权重
        x1_down = F.interpolate(x1, scale_factor=1 / 16, mode='bilinear', align_corners=False)

        attention = torch.sigmoid(x1_down)  # 假设 x1 是位置信息的表示
        x = x * attention

        pool2 = x
        x = self.conv3(x)
        x = x + pool2
        x = self.conv4(x)

        # 融合后的特征通过空间注意力模块
        x = self.sa_s(x) * x

        return x


class Fusion(nn.Module):
    def __init__(self, num_filters1, num_filters2, num_filters3):
        super(Fusion, self).__init__()
        self.upsample_1 = nn.ConvTranspose2d(in_channels=num_filters2, out_channels=num_filters2, kernel_size=4,
                                             padding=1, stride=2)
        self.upsample_2 = nn.ConvTranspose2d(in_channels=num_filters3, out_channels=num_filters3, kernel_size=4,
                                             padding=0, stride=4)
        self.final = nn.Sequential(
            nn.Conv2d(num_filters1 + num_filters2 + num_filters3, 1, kernel_size=1, padding=0),
            nn.ReLU(),
        )

    def forward(self, x1, x2, x3):
        x2 = self.upsample_1(x2)
        x3 = self.upsample_2(x3)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.final(x)

        return x

class TSCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_size = 384
        self.img_encoder = Backbone()
        self.fd_encoder = Backbone()

        self.cnv1_1 = nn.Conv2d(192, 64, 1, 1, 0)
        self.cnv1_2 = nn.Conv2d(192, 64, 1, 1, 0)

        self.cnv2_1 = nn.Conv2d(384, 64, 1, 1, 0)
        self.cnv2_2 = nn.Conv2d(384, 64, 1, 1, 0)

        self.cnv3_1 = nn.Conv2d(768, 64, 1, 1, 0)
        self.cnv3_2 = nn.Conv2d(768, 64, 1, 1, 0)

        self.mam1 = MAM(64)
        self.mam2 = MAM(64)
        self.mam3 = MAM(64)

        self.fuse1 = RGBDiffFusion()
        self.fuse2 = RGBDiffFusion()
        self.fuse3 = RGBDiffFusion()

        self.fuse_final = PyramidFusion()

        self.upSample = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4,
                                             padding=1, stride=2)

        self.proj_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128)
        )
        self.catf = nn.Conv2d(128, 64, 1)

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.ReLU()
        )

    def forward(self,  img_cur, diff, points=None):
        # Backbone 提取多层特征
        f_r1, f_r2, f_r3 = self.img_encoder(img_cur)
        f_d1, f_d2, f_d3 = self.fd_encoder(diff.repeat(1, 3, 1, 1))

        f_r1 = self.cnv1_1(f_r1)
        f_d1 = self.cnv1_2(f_d1)
        #
        f_r2 = self.cnv2_1(f_r2)
        f_d2 = self.cnv2_2(f_d2)

        f_r3 = self.cnv3_1(f_r3)
        f_d3 = self.cnv3_2(f_d3)

        cl_loss = 0
        fl = []
        if points is not None:  # 表示训练阶段
            fl_1 = (f_r1, f_d1)
            fl_2 = (f_r2, f_d2)
            fl_3 = (f_r3, f_d3)
            fl.append(fl_1)
            fl.append(fl_2)
            fl.append(fl_3)
            cl_loss = self.contrastive_loss(fl, points)
        # #
        f_R1, f_D1 = self.mam1(f_r1, f_d1)
        f_R2, f_D2 = self.mam2(f_r2, f_d2)
        f_R3, f_D3 = self.mam3(f_r3, f_d3)

        # 例如 mam1（代表 f_r1 和 f_d1）

        #
        # torch.save(f_r3.detach().cpu(), "./atten_map/scale-3/rgb_input_scale3.pt")
        # torch.save(f_d3.detach().cpu(), "./atten_map/scale-3/diff_input_scale3.pt")
        # torch.save(f_R3.detach().cpu(), "./atten_map/scale-3/rgb_output_scale3.pt")
        # torch.save(f_D3.detach().cpu(), "./atten_map/scale-3/diff_output_scale3.pt")

        f1 = self.fuse1(f_R1, f_D1)
        f2 = self.fuse2(f_R2, f_D2)
        f3 = self.fuse3(f_R3, f_D3)
        # f1 = f_r1 + f_d1
        # f2 = f_r2 + f_d2
        # f3 = f_r3 + f_d3
        # torch.save(f1.detach().cpu(), "./atten_map/scale-1/rgb_output_safm.pt")

        # torch.save(f_r1.detach().cpu(), "./atten_map/attMatrix_only_rgb/rgb_input_scale1.pt")
        # torch.save(f_d1.detach().cpu(), "./atten_map/attMatrix_only_rgb/diff_input_scale1.pt")
        # torch.save(f1.detach().cpu(), "./atten_map/scale-1/rgb_output_safm.pt")
        # torch.save(f_D1.detach().cpu(), "./atten_map/attMatrix_only_rgb/diff_output_scale1.pt")

        f = self.fuse_final(f1, f2, f3)[0]
        y = self.decoder(f)

        y_norm = self.norm(y)

        return y, y_norm, cl_loss

    def norm(self, dmap):
        # 密度图归一化
        B, C, H, W = dmap.size()
        density_sum = dmap.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        density_normed = dmap / (density_sum + 1e-6)
        return density_normed

    def contrastive_loss(self, f_l, points):
        loss_diff_cl = 0
        loss_rgb_cl = 0
        loss_rgb_diff_cl = 0
        for f_layer in f_l:
            fr = f_layer[0]
            fd = f_layer[1]
            pos_rgb_sample, neg_rgb_sample = self.make_sample(fr, points)
            pos_diff_sample, neg_diff_sample = self.make_sample(fd, points)
            pos_rgb_sample, pos_diff_sample = self.sample_paired_feats(pos_rgb_sample, pos_diff_sample)

            loss_rgb_cl += self.compute_intra_modal_loss(pos_rgb_sample, neg_rgb_sample)
            # loss_diff_cl += self.compute_intra_modal_loss(pos_diff_sample, neg_diff_sample)
            loss_rgb_diff_cl += self.compute_cross_modal_loss(pos_rgb_sample, pos_diff_sample)

        return loss_rgb_cl + loss_rgb_diff_cl

    def sample_paired_feats(self, feat_rgb, feat_diff, max_num=50):

        assert feat_rgb.shape[0] == feat_diff.shape[0], "两个模态的正样本数量必须一致"

        N = feat_rgb.shape[0]
        if N <= max_num:
            return feat_rgb, feat_diff
        else:
            idx = torch.randperm(N)[:max_num]
            return feat_rgb[idx], feat_diff[idx]

    def compute_cross_modal_loss(self, pos_rgb_sample, pos_diff_sample):
        pos_rgb = self.proj_head(pos_rgb_sample)
        pos_diff = self.proj_head(pos_diff_sample)

        # 归一化处理
        pos_rgb = F.normalize(pos_rgb, dim=1)
        pos_diff = F.normalize(pos_diff, dim=1)

        # 计算每个对应正样本的 cosine 相似度
        sim = F.cosine_similarity(pos_rgb, pos_diff, dim=1)  # [N_total]
        # 目标是相似度接近1，所以损失为 1 - sim
        loss = (1 - sim).mean()
        return loss

    def compute_intra_modal_loss(self, pos_feats, neg_feats, temperature=0.1):
        device = pos_feats.device
        N_pos, C = pos_feats.shape

        pos_feats = self.proj_head(pos_feats)
        neg_feats = self.proj_head(neg_feats)

        # Step 1: 归一化（用于计算 cosine 相似度）
        pos_feats = F.normalize(pos_feats, dim=1)
        neg_feats = F.normalize(neg_feats, dim=1)

        pos_sim = torch.ones(N_pos, 1).to(device)  # exp(1) = e 作为正对的分子

        neg_sim = torch.matmul(pos_feats, neg_feats.T)  # [N_pos, N_neg]

        logits = torch.cat([pos_sim, torch.exp(neg_sim / temperature)], dim=1)  # [N_pos, 1 + N_neg]

        denom = logits.sum(dim=1)  # softmax 分母
        loss = -torch.log(pos_sim.squeeze() / denom)  # 只保留正对的 log prob
        return loss.mean()

    def make_sample(self, f_map, points_list):
        B, C, H, W = f_map.shape
        stride = self.image_size // H
        pos_mask = self.get_postive_coords(points_list, stride, H, W)
        neg_mask = self.get_negative_coords(pos_mask, H, W)
        pos_feats = self.get_feature_by_coords(f_map, pos_mask)  # list of [N, C]
        neg_feats = self.get_feature_by_coords(f_map, neg_mask)  # list of [M, C]
        return torch.cat(pos_feats, dim=0), torch.cat(neg_feats, dim=0)

    def get_feature_by_coords(self, feature_map, coords):
        """
        feature_map: [B, C, H, W]
        coords: List of List of (py, px) per image in batch
        return: List of [N, C] tensors, N为正样本数
        """
        B, C, H, W = feature_map.shape
        features = []
        for b in range(B):
            f = feature_map[b]  # [C, H, W]
            this_coords = coords[b]
            selected = []
            for (py, px) in this_coords:
                if 0 <= py < H and 0 <= px < W:
                    feat = f[:, py, px]
                    selected.append(feat)
            if selected:
                features.append(torch.stack(selected))  # [N, C]
            else:
                features.append(torch.empty(0, C, device=feature_map.device))
        return features  # list of N x C

    def get_postive_coords(self, points_list, stride, H, W):
        """
        points_list: list of list of (x, y) in original image space
        stride: scaling ratio from image to feature map
        H, W: feature map height and width
        """
        pos_coords = []
        for pts in points_list:
            coord_set = set()
            for (x, y) in pts:
                px = min(W - 1, int(x // stride))
                py = min(H - 1, int(y // stride))
                coord_set.add((py, px))
            pos_coords.append(list(coord_set))
        return pos_coords

    def get_negative_coords(self, pos_coords, H, W, num_neg=30):
        """
        从特征图中随机采样负样本（不与正样本重合）
        """
        neg_coords = []
        for pos in pos_coords:
            pos_set = set(pos)
            candidates = [(py, px) for py in range(H) for px in range(W)
                          if (py, px) not in pos_set]
            selected = random.sample(candidates, min(num_neg, len(candidates)))
            neg_coords.append(selected)
        return neg_coords




class compare_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_encoder = Backbone()

        self.upSample = nn.ConvTranspose2d(in_channels=384, out_channels=64, kernel_size=4,
                                             padding=1, stride=2)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.ReLU()
        )

    def forward(self,  img_cur, diff):
        # Backbone 提取多层特征
        f_r1, f_r2, f_r3 = self.img_encoder(img_cur)

        y = self.decoder(self.upSample(f_r2))

        y_norm = self.norm(y)

        return y, y_norm

    def norm(self, dmap):
        # 密度图归一化
        B, C, H, W = dmap.size()
        density_sum = dmap.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        density_normed = dmap / (density_sum + 1e-6)
        return density_normed

class vss(nn.Module):
    def __init__(self, dim):
        super(vss, self).__init__()
        self.vssblock = VSSBlock(hidden_dim=int(dim/2))
        self.spatial_attention = sa(kernel_size=7)  # 空间注意力
        self.dim = dim

    def forward(self, x):
        # 将 x 分为两个张量，沿通道维度 (dim=1) 分割
        x1, x2 = torch.split(x, int(self.dim / 2), dim=1)

        # 调整 x1 的维度顺序
        x1 = x1.permute(0, 2, 3, 1)  # 转换为 (batch, height, width, channel)

        # 处理 x1 和 x2
        x1 = self.vssblock(x1)
        # 对 x1 应用 VSSBlock
        x1 = x1.permute(0, 3, 1, 2)

        # 对 x2 应用空间注意力机制
        x2 = x2 * self.spatial_attention(x2)

        # 将处理后的 x1 和 x2 合并
        out = torch.cat([x1, x2], dim=1)  # 在通道维度重新拼接

        return out


class sa(nn.Module):
    def __init__(self, kernel_size=7):
        super(sa, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 按通道维度进行平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 按通道维度进行最大池化
        # 将两个池化结果拼接
        x = torch.cat([avg_out, max_out], dim=1)
        # 使用卷积生成空间注意力图
        x = self.conv1(x)
        return self.sigmoid(x)  # 返回每个空间位置的权重


class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        # 上采样模块
        self.upsample_3 = nn.ConvTranspose2d(in_channels=768, out_channels=768, kernel_size=4,
                                             padding=1, stride=2)
        self.upsample_2 = nn.ConvTranspose2d(in_channels=384, out_channels=384, kernel_size=4,
                                             padding=1, stride=2)
        # 通道压缩模块
        self.squeeze3 = ODConv2d(768, 384, kernel_size=1, stride=1, padding=0)
        self.squeeze2 = ODConv2d(384, 192, kernel_size=1, stride=1, padding=0)
        self.squeeze1 = ODConv2d(192, 192, kernel_size=1, stride=1, padding=0)

        self.squeeze4 = ODConv2d(768, 192, kernel_size=1, stride=1, padding=0)
        self.squeeze5 = ODConv2d(384, 192, kernel_size=1, stride=1, padding=0)



    def forward(self, p1, p2, p3):
        """
        Args:
            p1: 最底层特征图 (高分辨率，低语义)，通常分辨率最高。
            p2: 中间层特征图。
            p3: 顶层特征图 (低分辨率，高语义)，通常分辨率最低。
        Returns:
            out: 融合后的多尺度特征图列表 [P1, P2, P3]
        """
        p3_out = self.squeeze4(p3)

        # 对 p3 上采样并与 p2 融合
        up3 = self.upsample_3(p3)
        s3 = self.squeeze3(up3)
        # 上采样 p3
        p2_fused = s3 + p2        # 融合上采样的 p3 和 p2
        p2_out = self.squeeze5(p2_fused)
        # 对 p2_fused 上采样并与 p1 融合
        up2 = self.upsample_2(p2_fused)  # 上采样 p2_fused
        p2_ = self.squeeze2(up2)  # 压缩通道数

        p1_fused = p2_ + p1              # 融合上采样的 p2 和 p1
        p1_out = self.squeeze1(p1_fused)  # 压缩通道数

        # 压缩 p3 的通道数

        return [p1_out, p2_out, p3_out]  # 返回融合后的多尺度特征


# class MMVCC(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.backbone = Backbone()
#         self.GLA1 = vss(192)
#         self.GLA2 = vss(384)
#         self.GLA3 = vss(768)
#         self.fpn = FPN()
#         self.final = nn.Sequential(
#             nn.Conv2d(192, 192, kernel_size=3, padding=1),
#             nn.BatchNorm2d(192),
#             nn.ReLU(),
#             nn.Conv2d(192, 96, 3, 1, 1),
#             nn.BatchNorm2d(96),
#             nn.ReLU(),
#             nn.Conv2d(96, 1, 1, 1, 0),
#             nn.ReLU()
#         )
#
#     def forward(self, x, x1):
#         # Backbone 提取多层特征
#         pool1, pool2, pool3 = self.backbone(x)
#         p1 = self.GLA1(pool1)
#         p2 = self.GLA2(pool2)
#         p3 = self.GLA3(pool3)
#
#         p1_fuse, _, _ = self.fpn(p1, p2, p3)
#         x = self.final(p1_fuse)
#
#         return x, self.norm(x)
#
#     def norm(self, dmap):
#         # 密度图归一化
#         B, C, H, W = dmap.size()
#         density_sum = dmap.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
#         density_normed = dmap / (density_sum + 1e-6)
#         return density_normed


from thop import profile


if __name__ == '__main__':
    inputA = torch.randn(1,3,224,224).cuda()
    inputB = torch.randn(1,1,224,224).cuda()

    model = TSCNet().cuda()
    flops, params = profile(model, inputs=(inputA, inputB))
    print(f"FLOPs: {flops / 1e6:.2f} M, Params: {params / 1e6:.2f} M")

