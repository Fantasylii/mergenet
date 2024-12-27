import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

class LowRankLinear(nn.Module, LoRALayer):
    def __inti__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            bias: bool = True
    ):
        super(LowRankLinear, self).__init__()
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        self.A = nn.Parameter(torch.randn(r, in_features))
        self.B = nn.Parameter(torch.randn(out_features, r))
        self.use_bias = bias
        if self.use_bias:
            self.bias = nn.Parameter(out_features)
        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor):
        y = (self.lora_dropout(x) @ self.A.transpose(0, 1) @ self.B.transpose(0, 1)) * self.scaling
        if self.use_bias:
            y = y + self.bias
        return y
    
class LowRankConv(nn.Module, LoRALayer):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            kernel_size: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
    ):
        super(LowRankConv, self).__init__()
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        self.conv_size = [out_channel, in_channel, kernel_size, kernel_size]
        self.A = nn.Parameter(torch.randn(r * kernel_size, in_channel * kernel_size))
        self.B = nn.Parameter(torch.randn(out_channel * kernel_size, r * kernel_size))
        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        weight = (self.B @ self.A).view(self.conv_size) * self.scaling
        return F.conv2d(
            x,
            weight
        )