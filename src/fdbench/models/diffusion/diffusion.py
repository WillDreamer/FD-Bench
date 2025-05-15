import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UnetDenoiser
from .fourier import FourierDenoiser
from .self_attention import self_atten

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

        if args.denoiser_type == "unet":
            self.model = UnetDenoiser(args)
        elif args.denoiser_type == "fourier":
            self.model = FourierDenoiser(args)
        elif args.denoiser_type == "self_attention":
            self.model = self_atten(args)
            
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

        self.model = self.model.to(device)

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
    
    def forward(self, x, target, grid, criterion=None):
        t = torch.randint(low=0, high=self.timesteps, size=(target.shape[0],)).to(x.device)

        perturbed, epsilon = self.forward_diffusion(target, t)

        x = torch.concat([perturbed, x], dim = 1)
        pred_epsilon = self.model(x, t)
        
        loss = criterion(epsilon, pred_epsilon)

        return pred_epsilon, loss
    
    def denoise_at_t(self, x_t, condition, t):
        timestep = torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device)
        # timestep_minus_1 = torch.full((x_t.shape[0],), t-1, dtype=torch.long, device=x_t.device)

        if t > 0:
            # z = torch.randn_like(x_t).to(x_t.device)
            alpha = self.extract(self.alphas, timestep)
            sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep)
            sqrt_alpha = self.extract(self.sqrt_alphas, timestep)
            # beta = self.extract(self.betas, timestep)
            # alpha_bar = self.extract(self.alpha_bars, timestep)
            # alpha_bar_minus_1 = self.extract(self.alpha_bars, timestep_minus_1)

            temp_x_t = torch.concat([x_t, condition], dim = 1)
            pred_epsilon = self.model(temp_x_t, timestep)

            # sigma = (1-alpha_bar_minus_1) / (1-alpha_bar) * beta
            # x_t_minus_1 = 1/sqrt_alpha * (x_t - (1-alpha) / sqrt_one_minus_alpha_bar * pred_epsilon) + sigma*z

            x_t_minus_1 = 1/sqrt_alpha * (x_t - (1-alpha) / sqrt_one_minus_alpha_bar * pred_epsilon)

            return x_t_minus_1
        else:
            alpha = self.extract(self.alphas, timestep)
            sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep)
            sqrt_alpha = self.extract(self.sqrt_alphas, timestep)

            temp_x_t = torch.concat([x_t, condition], dim = 1)
            pred_epsilon = self.model(temp_x_t, timestep)

            pred_x0 = 1/sqrt_alpha * (x_t - (1-alpha) / sqrt_one_minus_alpha_bar * pred_epsilon)

            return pred_x0
        
    
    def ddpm_sample(self, x, target, grid, criterion=None):
        x_t = torch.randn(target.shape).to(target.device)

        for t in range(self.timesteps-1, 0, -1):
            x_t = self.denoise_at_t(x_t, x, t)

        loss = criterion(x_t, target)

        return x_t, loss

    def ddim_sample(self, x, target, grid, criterion=None):
        pass