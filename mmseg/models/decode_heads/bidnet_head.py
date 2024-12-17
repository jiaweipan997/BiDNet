# Copyright (c) OpenMMLab. All rights reserved.
# Originally from https://github.com/visual-attention-network/segnext
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.device import get_device

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead

from ..utils import make_divisible
from ..utils import InvertedResidualV3 as InvertedResidual


class Matrix_Decomposition_2D_Base(nn.Module):
    """Base class of 2D Matrix Decomposition.

    Args:
        MD_S (int): The number of spatial coefficient in
            Matrix Decomposition, it may be used for calculation
            of the number of latent dimension D in Matrix
            Decomposition. Defaults: 1.
        MD_R (int): The number of latent dimension R in
            Matrix Decomposition. Defaults: 64.
        train_steps (int): The number of iteration steps in
            Multiplicative Update (MU) rule to solve Non-negative
            Matrix Factorization (NMF) in training. Defaults: 6.
        eval_steps (int): The number of iteration steps in
            Multiplicative Update (MU) rule to solve Non-negative
            Matrix Factorization (NMF) in evaluation. Defaults: 7.
        inv_t (int): Inverted multiple number to make coefficient
            smaller in softmax. Defaults: 100.
        rand_init (bool): Whether to initialize randomly.
            Defaults: True.
    """

    def __init__(self,
                 MD_S=1,
                 MD_R=64,
                 train_steps=6,
                 eval_steps=7,
                 inv_t=100,
                 rand_init=True):
        super().__init__()

        self.S = MD_S
        self.R = MD_R

        self.train_steps = train_steps
        self.eval_steps = eval_steps

        self.inv_t = inv_t

        self.rand_init = rand_init

    def _build_bases(self, B, S, D, R, device=None):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        """Forward Function."""
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        D = C // self.S
        N = H * W
        x = x.view(B * self.S, D, N)
        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, device=x.device)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, device=x.device)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        x = x.view(B, C, H, W)

        return x


class NMF2D(Matrix_Decomposition_2D_Base):
    """Non-negative Matrix Factorization (NMF) module.

    It is inherited from ``Matrix_Decomposition_2D_Base`` module.
    """

    def __init__(self, args=dict()):
        super().__init__(**args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, device=None):
        """Build bases in initialization."""
        if device is None:
            device = get_device()
        bases = torch.rand((B * S, D, R)).to(device)
        bases = F.normalize(bases, dim=1)

        return bases

    def local_step(self, x, bases, coef):
        """Local step in iteration to renew bases and coefficient."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        """Compute coefficient."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class Hamburger(nn.Module):
    """Hamburger Module. It consists of one slice of "ham" (matrix
    decomposition) and two slices of "bread" (linear transformation).

    Args:
        ham_channels (int): Input and output channels of feature.
        ham_kwargs (dict): Config of matrix decomposition module.
        norm_cfg (dict | None): Config of norm layers.
    """

    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=None,
                 **kwargs):
        super().__init__()

        self.ham_in = ConvModule(
            ham_channels, ham_channels, 1, norm_cfg=None, act_cfg=None)

        self.ham = NMF2D(ham_kwargs)

        self.ham_out = ConvModule(
            ham_channels, ham_channels, 1, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)

        return ham


class MFCA(nn.Module):
    """MFCA Module.

    Args:
        channels (int): The input (and output) channels of the MFCA Module.
        ratio (int): Squeeze ratio in MFCA Module, the intermediate channel will be
            ``int(channels/ratio)``. Default: 1.
        conv_cfg (None or dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configured
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configured by the first dict and the
            second activation layer will be configured by the second dict.
            Default: (dict(type='ReLU'), dict(type='HSigmoid', bias=3.0,
            divisor=6.0)).
    """

    def __init__(self,
                 channels,
                 ratio=1,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'),
                          dict(type='HSigmoid', bias=3.0, divisor=6.0))):
        super().__init__()

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=make_divisible(channels // ratio, 8),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=make_divisible(channels // ratio, 8),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):

        x = [
           resize(
               level,
               size=x[0].shape[2:],
               mode='bilinear',
               align_corners=False) for level in x
       ]  
        C = torch.cat(x, dim=1)
        out = self.global_avgpool(C)
        out = self.conv1(out)
        out = self.conv2(out)
        A1 = C * out

        return A1

class MFSA(nn.Module):
    """MFSA Module.

    Args:
        channels (int): The input (and output) channels of the MFSA Module.
        norm_cfg (None or dict): Config dict for norm layer.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
    
    """

    def __init__(self,channels,norm_cfg,act_cfg):
        super().__init__()

        self.DWConv = ConvModule(
          channels,
          channels,
          1,
          padding=0,
          groups=channels,
          norm_cfg=norm_cfg,
          act_cfg=act_cfg,
          )

    def forward(self, S, D):

       P = S + resize(D,size=S.shape[2:],mode='bilinear',align_corners=False)
       out = self.DWConv(P)
       A2 = P * out

       return A2

@MODELS.register_module()
class BiDNetHead(BaseDecodeHead):
   """BiDNet decode head.

   This decode head is the implementation of `BiDNet: A Real-Time Semantic Segmentation 
   Network with Anti-Feature Interference and Detail Recovery for Industrial Defects`.

   Args:
       attention_channels (int): input channels for attention module.
           Defaults: 512.
      attention_kwargs (int): kwagrs for attention module. Defaults: dict().
   """

   def __init__(self, attention_channels=512, attention_kwargs=dict(), **kwargs):
       super().__init__(input_transform='multiple_select', **kwargs)

       self.attention_channels = attention_channels
       self.Shallow_branch_dim = self.in_channels[0]//2
       
       # Shallow branch
       self.Shallow_branch = InvertedResidual(
               in_channels=self.Shallow_branch_dim,
               out_channels=self.channels,
               mid_channels=self.Shallow_branch_dim*6,
               kernel_size=3,
               stride=1,
               se_cfg=dict(channels=self.Shallow_branch_dim*6, ratio=1, act_cfg=(dict(type='ReLU'), dict(type='HSigmoid', bias=3.0, divisor=6.0))),
               with_expand_conv=True,
               conv_cfg=self.conv_cfg,
               norm_cfg=self.norm_cfg,
               act_cfg=dict(type='HSwish'),
               with_cp=False) 
       
       # Deep branch
       self.MFCA = MFCA(**dict(channels=sum(self.in_channels), ratio=1, act_cfg=(dict(type='ReLU'),dict(type='HSigmoid', bias=3.0, divisor=6.0))))

       self.conv1 = ConvModule(
           sum(self.in_channels),
           self.attention_channels,
           1,
           conv_cfg=self.conv_cfg,
           norm_cfg=self.norm_cfg,
           act_cfg=self.act_cfg)

       self.attention = Hamburger(self.attention_channels, attention_kwargs, **kwargs)

       self.conv2 = ConvModule(
           self.attention_channels,
           self.channels,
           1,
           conv_cfg=self.conv_cfg,
           norm_cfg=self.norm_cfg,
           act_cfg=self.act_cfg)
       
       # Fuse
       self.MFSA = MFSA(self.channels, self.norm_cfg, self.act_cfg)

   
   def forward(self, inputs):
       """Forward function."""

       #Shallow branch
       S1 = inputs[0]
       S = self.Shallow_branch(S1)
       
       #Deep branch
       S2_3_4 = self._transform_inputs(inputs)    
       A1 = self.MFCA(S2_3_4)
       A1 = self.conv1(A1)
       A1 = self.attention(A1)
       D = self.conv2(A1)

       #Fuse
       A2 = self.MFSA(S,D)
       O = self.cls_seg(A2)
       
       return O
