
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=8, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        padding_mode='circular')

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="circular")
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     padding_mode='circular')
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     padding_mode='circular')
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     padding_mode='circular')
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)


        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class Denoiser(nn.Module):
    def __init__(self, args):
        super().__init__()
        ch, out_ch, ch_mult = args.denoiser_ch, args.denoiser_out_ch, tuple(args.denoiser_ch_mult)
        num_res_blocks = args.denoiser_num_res_blocks
        attn_resolutions = args.denoiser_attn_resolutions
        dropout = args.denoiser_dropout
        in_channels = args.denoiser_in_channels
        resolution = args.input_size
        resamp_with_conv = args.denoiser_resamp_with_conv
        
        
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       padding_mode='circular')

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)


        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        padding_mode='circular')

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)

                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))


        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)


        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)


        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
    
class diffusion(nn.Module):
    def __init__(self, args):
        super(diffusion, self).__init__()
        self.beta_min = args.beta_min
        self.beta_max = args.beta_max
        self.timesteps = args.timesteps

        self.betas = torch.linspace(start=self.beta_min, end=self.beta_max, steps=self.timesteps)
        self.sqrt_betas = torch.sqrt(self.betas)

        self.alphas = 1 - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)

        self.model = Denoiser(args)
        self.device = None

    def to(self, *args, **kwargs):
        """Override the default `to` method to ensure all tensors are moved to the specified device."""
        super(diffusion, self).to(*args, **kwargs)

        device = next(self.parameters()).device
        self.betas = self.betas.to(device)
        self.sqrt_betas = self.sqrt_betas.to(device)
        self.alphas = self.alphas.to(device)
        self.sqrt_alphas = self.sqrt_alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
        self.sqrt_alpha_bars = self.sqrt_alpha_bars.to(device)

        self.device = device

        return self
    
    def extract(self, a, t):
        b = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(b, 1, 1, 1)

    def forward_diffusion(self, x_zeros, t): 
        epsilon = torch.randn_like(x_zeros).to(x_zeros.device)
        
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t)
        
        noisy_sample = x_zeros * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar

        return noisy_sample, epsilon
    
    def forward(self, x, target, creterion=None):
        t = torch.randint(low=0, high=self.timesteps, size=(target.shape[0],)).to(x.device)

        perturbed, epsilon = self.forward_diffusion(target, t)

        x = torch.concat([perturbed, x], dim = 1)
        pred_epsilon = self.model(x, t)
        
        loss = creterion(epsilon, pred_epsilon)

        return pred_epsilon, loss
    
    def denoise_at_t(self, x_t, condition, t):
        timestep = torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device)
        timestep_minus_1 = torch.full((x_t.shape[0],), t-1, dtype=torch.long, device=x_t.device)

        if t > 1:
            z = torch.randn_like(x_t).to(x_t.device)
            alpha = self.extract(self.alphas, timestep)
            sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep)
            sqrt_alpha = self.extract(self.sqrt_alphas, timestep)
            beta = self.extract(self.betas, timestep)
            alpha_bar = self.extract(self.alpha_bars, timestep)
            alpha_bar_minus_1 = self.extract(self.alpha_bars, timestep_minus_1)

            temp_x_t = torch.concat([x_t, condition], dim = 1)
            pred_epsilon = self.model(temp_x_t, timestep)

            sigma = (1-alpha_bar_minus_1) / (1-alpha_bar) * beta
            x_t_minus_1 = 1/sqrt_alpha * (x_t - (1-alpha) / sqrt_one_minus_alpha_bar * pred_epsilon) + sigma*z

            return x_t_minus_1
        else:
            alpha = self.extract(self.alphas, timestep)
            sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep)
            sqrt_alpha = self.extract(self.sqrt_alphas, timestep)

            temp_x_t = torch.concat([x_t, condition], dim = 1)
            pred_epsilon = self.model(temp_x_t, timestep)

            pred_x0 = 1/sqrt_alpha * (x_t - (1-alpha) / sqrt_one_minus_alpha_bar * pred_epsilon)

            return pred_x0
        
    
    def ddpm_sample(self, x, target, creterion=None):
        x_t = torch.randn(target.shape).to(target.device)

        for t in range(999, 1, -1):
            x_t = self.denoise_at_t(x_t, x, t)

        loss = creterion(x_t, target)

        return x_t, loss

    def ddim_sample(self, x, target, creterion=None):
        pass