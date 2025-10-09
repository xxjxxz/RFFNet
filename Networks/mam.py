import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class NonLocalBlock(nn.Module):
    """ NonLocalBlock Module"""

    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()

        conv_nd = nn.Conv2d
        self.in_channels = in_channels
        self.inter_channels = self.in_channels // 2

        self.catconv = BasicConv2d(in_planes=self.in_channels * 2, out_planes=self.in_channels, kernel_size=3,
                                   padding=1, stride=1)

        self.main_bnRelu = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.auxiliary_bnRelu = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.R_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.R_W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)

        self.F_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.F_W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)
        self.F_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.F_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, main_fea, auxiliary_fea):
        mainNonLocal_fea = self.main_bnRelu(main_fea)
        auxiliaryNonLocal_fea = self.auxiliary_bnRelu(auxiliary_fea)

        batch_size = mainNonLocal_fea.size(0)

        g_x = self.R_g(mainNonLocal_fea).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        l_x = self.F_g(auxiliaryNonLocal_fea).view(batch_size, self.inter_channels, -1)
        l_x = l_x.permute(0, 2, 1)

        catNonLocal_fea = self.catconv(torch.cat([mainNonLocal_fea, auxiliaryNonLocal_fea], dim=1))

        theta_x = self.F_theta(catNonLocal_fea).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.F_phi(catNonLocal_fea).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)

        # add self_f and mutual f
        f_div_C = F.softmax(f, dim=-1)
        # torch.save(f_div_C, f"atten_map/attention_softmax.pt")
        f_div_C_1 = torch.ones_like(f) / f.size(-1)  # 每行和为1

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *mainNonLocal_fea.size()[2:])
        W_y = self.R_W(y)
        z = W_y + main_fea

        m = torch.matmul(f_div_C, l_x)
        m = m.permute(0, 2, 1).contiguous()
        m = m.view(batch_size, self.inter_channels, *auxiliaryNonLocal_fea.size()[2:])
        W_m = self.F_W(m)
        p = W_m + auxiliary_fea

        return z, p


class MAM(nn.Module):
    def __init__(self, inchannels):
        super(MAM, self).__init__()
        self.Nonlocal_RGB_Diff = NonLocalBlock(inchannels)
        # self.a = 0.01

    def forward(self, cur, diff):
        cur_mix, diff_mix = self.Nonlocal_RGB_Diff(cur, diff)
        cur = cur + cur_mix
        diff = diff + diff_mix
        return cur, diff


class RGBDiffFusion(nn.Module):
    def __init__(self):
        super(RGBDiffFusion, self).__init__()
        self.fuse1 = nn.Conv2d(in_channels=128, out_channels=32, stride=1, kernel_size=3, padding=1)
        self.fuse2 = nn.Conv2d(in_channels=128, out_channels=32, stride=1, kernel_size=3, padding=1)
        self.fuse3 = nn.Conv2d(in_channels=128, out_channels=32, stride=1, kernel_size=3, padding=1)

        self.gate = nn.Conv2d(in_channels=32, out_channels=2, stride=1, kernel_size=3, padding=1)

        self.mlp_rgb = nn.Linear(32, 64)
        self.mlp_diff = nn.Linear(32, 64)

        self.conv_rgb = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv_diff = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, rgb_fea, diff_fea):

        rgb_diff_feat = self.fuse1(torch.cat((rgb_fea, diff_fea), dim=1))
        N, C, h, w = rgb_diff_feat.shape

        attention_feature_max = F.adaptive_max_pool2d(rgb_diff_feat, (1, 1)).view(N, C)

        channel_attention_rgb = (F.softmax(self.mlp_rgb(attention_feature_max), dim=1).unsqueeze(2).unsqueeze(3)) * rgb_fea
        channel_attention_diff = (F.softmax(self.mlp_diff(attention_feature_max), dim=1).unsqueeze(2).unsqueeze(3)) * diff_fea

        ch_attn_rgb_diff_feat = self.fuse2(torch.cat((channel_attention_rgb, channel_attention_diff), dim=1))

        ch_spatial_attention_rgb = (torch.sigmoid(self.conv_rgb(ch_attn_rgb_diff_feat))) * channel_attention_rgb
        ch_spatial_attention_diff = (torch.sigmoid(self.conv_diff(ch_attn_rgb_diff_feat))) * channel_attention_diff

        gt_ch_attn_rgb_diff_feat = self.fuse3(torch.cat((ch_spatial_attention_rgb, ch_spatial_attention_diff), dim=1))

        gate = F.adaptive_avg_pool2d(torch.sigmoid(self.gate(gt_ch_attn_rgb_diff_feat)), (1, 1))
        gate_rgb = gate[:, :1, :, :]
        gate_diff = gate[:, 1:, :, :]

        rgb_final = gate_rgb * ch_spatial_attention_rgb
        diff_final = gate_diff * ch_spatial_attention_diff

        final = rgb_final + diff_final
        return final


class PyramidFusion(nn.Module):
    def __init__(self):
        super(PyramidFusion, self).__init__()
        self.upSample1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, padding=1, stride=2)
        self.upSample2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, padding=1, stride=2)
        self.conv1 = nn.Conv2d(64*2, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64*2, 64, 3, 1, 1)

    def forward(self, rgb_diff_m1, rgb_diff_m2, rgb_diff_m3):
        rgb_diff_m3x2 = self.upSample1(rgb_diff_m3)
        fuse23 = self.conv1(torch.cat((rgb_diff_m2, rgb_diff_m3x2), dim=1))
        fuse23x2 = self.upSample2(fuse23)
        fuse123 = self.conv2(torch.cat((rgb_diff_m1, fuse23x2), dim=1))
        return fuse123, fuse23, rgb_diff_m3

if __name__ == '__main__':
    f = MAM(64)
    x1 = torch.randn(1, 64, 32, 32)
    x2 = torch.randn(1, 64, 32, 32)
    y = f(x1, x2)