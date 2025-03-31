import torch
import torch.nn as nn
import torch.nn.functional as F


def Normalize(in_channels, num_groups=16):
    """GroupNorm模块,"""
    # 如果不能整除
    if in_channels % num_groups != 0:
        num_groups = in_channels
    return nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.SiLU(),
            Normalize(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.SiLU(),
            Normalize(out_channels),
        )
        if self.in_channels != self.out_channels:
            # 如果输入输出通道不等，加一个卷积使残差可以连接
            self.conv_shortcut = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        x0 = self.layer(x)
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)
        return x + x0


class DownSample(nn.Module):
    """平均池化下采样两倍或卷积缩小两倍"""

    def __init__(self, in_chs):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            #    Transformer(in_chs, in_chs)
        )

    def forward(self, x):
        return self.down(x)


class UpSample(nn.Module):
    """最临近插值二倍上采样或上采样后再进行一次卷积"""

    def __init__(self, in_channels, with_conv=False):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.up(x)
        if self.with_conv:
            x = self.conv(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self, in_channels, out_channels, embed_dim=None, num_heads=2, dropout=0.2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if embed_dim is None:
            self.embed_dim = in_channels
        else:
            self.embed_dim = embed_dim
        self.att = nn.MultiheadAttention(
            self.embed_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)
        x = self.norm1(x)
        attn_output, _ = self.att(x, x, x)
        x = x + self.dropout(attn_output)  # 残差连接
        x = self.norm2(x)
        x = x.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        x = self.conv(x)
        return x


class AvgMaxAttention(nn.Module):

    def __init__(self, in_chs, out_chs):
        super().__init__()
        self.avg = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=0),
            nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            Normalize(out_chs),
        )
        self.max = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0),
            nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            Normalize(out_chs),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(2 * out_chs, in_chs, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x_avg = self.avg(x)
        x_max = self.max(x)
        score = self.mix(torch.cat((x_avg, x_max), dim=1))
        return x * F.sigmoid(score)


class ENetConv(nn.Module):
    """同UNet定义连续的俩次卷积"""

    def __init__(
        self,
        in_channels,
        out_channels,
        att=True,
    ):
        super(ENetConv, self).__init__()
        self.key = att
        self.conv = ResnetBlock(in_channels, out_channels)
        if att:
            self.att = AvgMaxAttention(out_channels, out_channels)
            # self.att = DotAttention(out_channels, out_channels)
            # self.att = Transformer(out_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.key:
            att = self.att(x)
            return x + att
        return x
