import torch
import torch.nn as nn


def norm(n_channel, n_group=32):
    return nn.GroupNorm(n_group, n_channel)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel=None, dropout=0.2):
        super(ResNetBlock, self).__init__()
        out_channel = in_channel if out_channel is None else out_channel
        self.norm1 = norm(in_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.norm2 = norm(out_channel)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.activation = Swish()

        self.skip = nn.Conv2d(in_channel, out_channel, 3, 1, 1)

    def forward(self, x):
        h_ = self.norm1(x)
        h_ = self.activation(h_)
        h_ = self.conv1(h_)
        h_ = self.norm2(h_)
        h_ = self.activation(h_)
        h_ = self.conv2(h_)

        x = self.skip(x)
        return x + h_


class Downsample(nn.Module):
    def __init__(self, channel):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(channel, channel, 3, 2, 0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channel):
        super(Upsample, self).__init__()
        self.conv = nn.ConvTranspose2d(channel, channel, 2, 2, 0)

    def forward(self, x):
        return self.conv(x)


class Attention(nn.Module):
    def __init__(self, channel):
        super(Attention, self).__init__()
        self.norm = norm(channel)
        self.q = nn.Conv2d(channel, channel, 3, 1, 1)
        self.k = nn.Conv2d(channel, channel, 3, 1, 1)
        self.v = nn.Conv2d(channel, channel, 3, 1, 1)
        self.out = nn.Conv2d(channel, channel, 3, 1, 1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        bs, c, h, w = x.shape
        q = q.reshape(bs, c, h * w)
        k = k.reshape(bs, c, h * w)
        v = v.reshape(bs, c, h * w)

        q = q.permute(0, 2, 1)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** 0.5)
        w_ = torch.nn.functional.softmax(w_, dim=2)

        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(bs, c, h, w)

        return self.out(h_) + x
