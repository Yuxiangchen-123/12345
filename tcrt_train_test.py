# tcrt_train_test.py
# Simplified version with model definitions (part 1 of full script)
# (Full training + testing loop can be added here as needed)

import os, math, random, argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ECA Layer
class ECALayer(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x.transpose(1,2)).transpose(1,2)
        y = self.conv(y.transpose(1,2))
        y = self.sigmoid(y).transpose(1,2)
        return x * y.expand_as(x)

# Causal Conv
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=0, dilation=dilation)
    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))

# TCN Residual Block
class TCNResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.proj1 = nn.utils.weight_norm(nn.Conv1d(channels, channels, 1))
        self.proj2 = nn.utils.weight_norm(nn.Conv1d(channels, channels, 1))
    def forward(self, x):
        y = self.conv1(x); y = self.gelu(y); y = self.drop(y)
        y = self.conv2(y)
        y = self.proj1(y) + x
        y = self.gelu(y)
        y = self.proj2(y)
        return y

# DI-TCN
class DI_TCN(nn.Module):
    def __init__(self, in_ch, base_ch=100, max_kernel=9, scales=3, tcn_blocks=3, dropout=0.1):
        super().__init__()
        self.stem = nn.Conv1d(in_ch, base_ch, kernel_size=1)
        ks = [max_kernel // (2**i) for i in range(scales)]
        ks = [k if k % 2 == 1 else k+1 for k in ks]
        self.branches = nn.ModuleList([nn.Conv1d(base_ch, base_ch, k, padding=k//2) for k in ks])
        self.eca = ECALayer(base_ch)
        self.tcn = nn.ModuleList([TCNResidualBlock(base_ch, 3, dilation=2**i, dropout=dropout)
                                  for i in range(tcn_blocks)])
        self.bn = nn.BatchNorm1d(base_ch)
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.stem(x)
        outs = [b(x) for b in self.branches]
        y = torch.mean(torch.stack(outs, dim=0), dim=0)
        y = F.gelu(y); y = self.eca(y)
        for blk in self.tcn: y = blk(y)
        y = self.bn(y); y = F.gelu(y)
        return y.transpose(1,2)
