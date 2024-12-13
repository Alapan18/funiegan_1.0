"""
 > Network architecture of FUnIE-GAN model
   * Paper: arxiv.org/pdf/1903.09766.pdf
 > Maintainer: https://github.com/xahidbuffon
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if bn: layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x
    
class GeneratorFunieGAN(nn.Module):
    """ A 5-layer UNet-based generator with Axial Attention layers """
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorFunieGAN, self).__init__()
        # Encoding layers
        self.down1 = UNetDown(in_channels, 32, bn=False)
        self.down2 = UNetDown(32, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 256)
        self.down5 = UNetDown(256, 256, bn=False)

        # Axial Attention layers after down blocks
        self.axial1 = AxialAttention(256)
        self.axial2 = AxialAttention(256)

        # Decoding layers
        self.up1 = UNetUp(256, 256)
        self.up2 = UNetUp(512, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        # Apply Axial Attention
        d5 = self.axial1(d5)
        d5 = self.axial2(d5)

        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u45 = self.up4(u3, d1)
        return self.final(u45)


class DiscriminatorFunieGAN(nn.Module):
    """ A 4-layer Markovian discriminator as described in the paper
    """
    def __init__(self, in_channels=3):
        super(DiscriminatorFunieGAN, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            #Returns downsampling layers of each discriminator block
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if bn: layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels*2, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

# Attention block definition
class AxialAttention(nn.Module):
    def __init__(self, dim, dim_head=32, num_heads=4):
        super(AxialAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head

        self.to_qkv = nn.Conv1d(dim, dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape  # Batch, Channels, Height, Width

        # Process along height axis
        x_height = x.permute(0, 3, 1, 2).reshape(b * w, c, h)  # Treat width as batch
        qkv = self.to_qkv(x_height).chunk(3, dim=1)  # Query, Key, Value
        q, k, v = map(lambda t: t.reshape(b * w, self.num_heads, self.dim_head, -1), qkv)
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        x_height = (attn @ v).reshape(b * w, -1, h).permute(0, 2, 1).reshape(b, c, h, w)

        # Process along width axis
        x_width = x.permute(0, 2, 1, 3).reshape(b * h, c, w)
        qkv = self.to_qkv(x_width).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b * h, self.num_heads, self.dim_head, -1), qkv)
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        x_width = (attn @ v).reshape(b * h, -1, w).permute(0, 2, 1).reshape(b, c, h, w)

        return x_height + x_width
