import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from fdbench.models.latent.modules.autoencoder2d import SimpleAutoencoder
from fdbench.models.latent.modules.basics import GroupNorm


class DilatedResidualBlock(nn.Module):
    def __init__(self, dim, dilation=1, padding_mode='circular'):
        super(DilatedResidualBlock, self).__init__()
        self.dim = dim
        self.dilation = dilation
        self.padding_mode = padding_mode

        self.conv = nn.Sequential(
            nn.GroupNorm(1, self.dim),
            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1,
                      padding_mode=self.padding_mode),
            nn.GELU(),
            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=self.dilation, dilation=self.dilation,
                      padding_mode=self.padding_mode),
            nn.GELU(),
            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1,
                      padding_mode=self.padding_mode),
        )

        self.ffn = nn.Sequential(
            nn.GroupNorm(1, self.dim),
            nn.Conv2d(self.dim, self.dim, 1, 1, 0, bias=False),
            nn.GELU(),
            nn.Conv2d(self.dim, self.dim, 1, 1, 0, bias=False))

    def forward(self, x):
        x = x + self.conv(x)
        x = x + self.ffn(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self,
                 latent_dim,  # dimension of the latent space
                 prop_n_block,  # number of residual blocks in the propagation network
                 prop_n_embd,  # number of channels in the propagation network
                 dilation=2,
                 ):
        #
        super(SimpleCNN, self).__init__()
        self.latent_dim = latent_dim
        self.prop_n_block = prop_n_block
        self.prop_n_embd = prop_n_embd

        self.in_proj = nn.Conv2d(self.latent_dim, self.prop_n_embd, 1, 1, 0)

        # n x resnet blocks
        self.net = nn.Sequential(*
                                 [DilatedResidualBlock(self.prop_n_embd,
                                                       dilation=dilation,
                                                       padding_mode='circular')
                                  for _ in range(self.prop_n_block)]
                                 )
        self.out_proj = nn.Sequential(
            GroupNorm(self.prop_n_embd),
            nn.Conv2d(self.prop_n_embd, self.latent_dim, 1, 1, 0))

    def forward(self, z):
        b, c, h, w = z.shape
        z = self.in_proj(z)
        z = self.net(z)   # spatial mixing
        z = self.out_proj(z)
        return z


class latent(nn.Module):
    def __init__(self, args={}):
        super(latent, self).__init__()

        self.vq_ae = SimpleAutoencoder(args)

        self.latent_resolution = args.latent_resolution
        self.latent_dim = args.latent_dim

        self.propagator = SimpleCNN(
            latent_dim=self.latent_dim,
            prop_n_block=args.prop_n_block,
            prop_n_embd=args.prop_n_embd,
            dilation=args.dilation,
        )
        if args.tem_mod == 'next_step':
            self.steps = 1

    def forward(self, data, target, grid, creterion=None):
        z = self.vq_ae.encode(data)
        out_lst = []
        z = z.squeeze()
        for t in range(self.steps):

            z_new = self.propagator(z)
            z = z_new

            y_hat = self.vq_ae.decode(z_new)
            out_lst.append(y_hat)
            
        out_lst = torch.stack(out_lst, dim=1).to(data.device)
        if self.steps == 1:
            out_lst = out_lst.squeeze(1)

        loss = creterion(out_lst, target)

        return out_lst, loss
