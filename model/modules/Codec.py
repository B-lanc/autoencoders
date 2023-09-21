import torch
import torch.nn as nn

from .Blocks import ResNetBlock, Upsample, Downsample, Attention, norm


"stolen from compvis"


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_channel=128,
        ch_mult=[1, 1, 2, 2, 4],
        z_channels=32,
        resolution=256,
        depth=1,
        attention_res=[16],
        dropout=0.2,
    ):
        super(Encoder, self).__init__()
        level = len(ch_mult)
        ch_mult = (1,) + tuple(ch_mult)
        res = resolution

        self.conv_in = nn.Conv2d(in_channels, base_channel, 3, 1, 1)
        self.down = []
        for i in range(level):
            block = []
            block_in = base_channel * ch_mult[i]
            block_out = base_channel * ch_mult[i + 1]

            for _ in range(depth):
                block.append(ResNetBlock(block_in, block_out, dropout))
                block_in = block_out
                if res in attention_res:
                    block.append(Attention(block_in))
            if i != level - 1:
                block.append(Downsample(block_in))
                res = res // 2

            block = nn.Sequential(*block)
            self.down.append(block)
        self.down = nn.Sequential(*self.down)

        self.mid = nn.Sequential(
            ResNetBlock(block_in, block_in, dropout),
            Attention(block_in),
            ResNetBlock(block_in, block_in, dropout),
        )
        self.norm = norm(block_in)
        self.conv_out = nn.Conv2d(block_in, z_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.down(x)
        x = self.mid(x)
        x = self.norm(x)
        x = self.conv_out(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels=3,
        base_channel=128,
        ch_mult=[1, 1, 2, 2, 4],
        z_channels=32,
        resolution=256,
        depth=1,
        attention_res=[16],
        dropout=0.2,
    ):
        super(Decoder, self).__init__()
        level = len(ch_mult)
        ch_mult = (1,) + tuple(ch_mult)
        channels = [base_channel * 2 ** i for i in reversed(ch_mult)]
        res = resolution // 2 * (level - 1)

        self.conv_in = nn.Conv2d(z_channels, channels[0], 3, 1, 1)
        self.mid = nn.Sequential(
            ResNetBlock(channels[0], channels[0], dropout),
            Attention(channels[0]),
            ResNetBlock(channels[0], channels[0], dropout),
        )

        self.up = []
        for i in range(level):
            block = []
            block_in = channels[i]
            block_out = channels[i + 1]

            for _ in range(depth):
                block.append(ResNetBlock(block_in, block_out, dropout))
                block_in = block_out
                if res in attention_res:
                    block.append(Attention(block_in))
                if i != level - 1:
                    block.append(Upsample(block_in))
                    res = res * 2

            block = nn.Sequential(*block)
            self.up.append(block)
        self.up = nn.Sequential(*self.up)

        self.norm = norm(block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid(x)
        x = self.up(x)
        x = self.norm(x)
        x = self.conv_out(x)
        return x
