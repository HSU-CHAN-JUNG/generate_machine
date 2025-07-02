import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as dset
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

class DoubleConv(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.GroupNorm(4, out_c), #equivalent with LayerNorm
        nn.SiLU()
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.GroupNorm(4, out_c), #equivalent with LayerNorm
        nn.SiLU()
    )

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    return x

class Down(nn.Module):
  def __init__(self, in_c, out_c, emb_dim=64):
    super().__init__()
    self.down = nn.Sequential(
        nn.AvgPool2d(2),
        DoubleConv(in_c,out_c),
    )

    self.emb_layer = nn.Sequential(
        nn.ReLU(),
        nn.Linear(emb_dim, out_c),
    )

  def forward(self, x, t):
    x = self.down(x)
    t_emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
    return x + t_emb

class Up(nn.Module):
    def __init__(self, in_c, out_c, emb_dim=64):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_c,out_c)
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_c),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

class Unet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=64, resize=64):
        super().__init__()
        self.time_dim = time_dim

        self.inc = DoubleConv(c_in, 128) 

        self.down1 = Down(128, 256) 
        self.sa1 = SelfAttention(256, int(resize/2)) 
        self.down2 = Down(256, 256) 
        self.sa2 = SelfAttention(256, int(resize/4)) 
        self.down3 = Down(256, 256) 
        self.sa3 = SelfAttention(256, int(resize/8)) 
        self.down4 = Down(256, 256) 
        self.sa4 = SelfAttention(256, int(resize/16)) 

        self.bot1 = DoubleConv(256, 512) 
        self.bot2 = DoubleConv(512, 512) 
        self.bot3 = DoubleConv(512, 256) 

        self.up1 = Up(512, 256) 
        self.sa5 = SelfAttention(256, int(resize/8))
        self.up2 = Up(512, 256)
        self.sa6 = SelfAttention(256, int(resize/4)) 
        self.up3 = Up(512, 128) 
        self.sa7 = SelfAttention(128, int(resize/2)) 
        self.up4 = Up(256, 128) 
        self.sa8 = SelfAttention(128, int(resize))

        self.outc = nn.Conv2d(128, c_out, kernel_size=1) 

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
         10000
         ** (torch.arange(0, channels, 2, device=t.device).type(t.dtype) / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        # (bs,) -> (bs, time_dim)
        t = t.unsqueeze(-1).type(x.dtype)
        t = self.pos_encoding(t, self.time_dim)
        #initial conv
        x1 = self.inc(x)

        #Down
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        x5 = self.down4(x4, t)
        x5 = self.sa4(x5)
        #Bottle neck
        x5 = self.bot1(x5)
        x5 = self.bot2(x5)
        x5 = self.bot3(x5)

        #Up
        x = self.up1(x5, x4, t)
        x = self.sa5(x)
        # print(x.shape, x2.shape)
        x = self.up2(x, x3, t)
        x = self.sa6(x)
        x = self.up3(x, x2, t)
        x = self.sa7(x)
        x = self.up4(x, x1, t)
        x = self.sa8(x)

        #Output
        output = self.outc(x)
        return output

if __name__ =="__main__":
    model = Unet()
    y = model(torch.randn(2,3,64,64), torch.randn(2))
    print(y.shape)