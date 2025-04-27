
import torch
import torch.nn as nn
from torch_geometric.utils import grid

from fdbench.models.graph_diffusion.network import MeshGraphNetsDenoiser


class graph_diffusion(nn.Module):
    def __init__(self, args):
        super(graph_diffusion, self).__init__()
        self.beta_min = args.beta_min
        self.beta_max = args.beta_max
        self.timesteps = args.timesteps
        self.batch_size = args.batch_size

        self.betas = torch.linspace(start=self.beta_min, end=self.beta_max, steps=self.timesteps)
        self.sqrt_betas = torch.sqrt(self.betas)

        self.alphas = 1 - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)

        self.model = MeshGraphNetsDenoiser(args)

    def to(self, *args, **kwargs):
        """Override the default `to` method to ensure all tensors are moved to the specified device."""
        super(graph_diffusion, self).to(*args, **kwargs)

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
        return out.reshape(b, 1)

    def forward_diffusion(self, x_zeros, t, batch_index): 
        epsilon = torch.randn_like(x_zeros).to(x_zeros.device)
        
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t)
        sqrt_alpha_bar_expand = sqrt_alpha_bar[batch_index]

        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t)
        sqrt_one_minus_alpha_bar_expand = sqrt_one_minus_alpha_bar[batch_index]
        
        noisy_sample = x_zeros * sqrt_alpha_bar_expand + epsilon * sqrt_one_minus_alpha_bar_expand

        return noisy_sample, epsilon
    
    def forward(self, x, target, grid, criterion=None):
        num_graphs = x.shape[0]
        
        x_node_feature, x_edge_index, x_batch_index = self.grid2graph(x)
        target_node_feature, _, _ = self.grid2graph(target)

        t = torch.randint(low=0, high=self.timesteps, size=(num_graphs,)).to(x.device)

        perturbed_target, epsilon = self.forward_diffusion(target_node_feature, t, x_batch_index)
        pred_epsilon = self.model(perturbed_target, x_node_feature, x_edge_index, x_batch_index, t)

        loss = criterion(epsilon, pred_epsilon)
        
        return pred_epsilon, loss
    
    def grid2graph(self, input_grid):
        batch_size, channels, height, width = input_grid.shape
        
        node_feature = input_grid.permute(0, 2, 3, 1).reshape(-1, channels)

        single_edge_index, _ = grid(height, width)
        single_edge_index = single_edge_index.to(torch.long)
        batch_edge_index = single_edge_index.repeat(1, batch_size)
        
        batch_offsets = torch.arange(batch_size)
        batch_offsets = batch_offsets.repeat_interleave(single_edge_index.shape[1]) * height*width
        batch_offsets = batch_offsets.unsqueeze(0).repeat(2, 1)
        
        edge_index = (batch_edge_index.unsqueeze(0) + batch_offsets).reshape(2, -1)
        batch_index = torch.arange(batch_size).repeat_interleave(height * width)

        return node_feature, edge_index.to(self.device), batch_index.to(self.device)
    
    def denoise_at_t(self, x_t, condition_node_feature, edge_index, batch_index, t):
        timestep = torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device)

        if t > 0:
            alpha = self.extract(self.alphas, timestep)
            sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep)
            sqrt_alpha = self.extract(self.sqrt_alphas, timestep)
            
            alpha_expand = alpha[batch_index]
            sqrt_one_minus_alpha_bar_expand = sqrt_one_minus_alpha_bar[batch_index]
            sqrt_alpha_expand = sqrt_alpha[batch_index]

            pred_epsilon = self.model(x_t, condition_node_feature, edge_index, batch_index, timestep)


            x_t_minus_1 = 1/sqrt_alpha_expand * (x_t - (1-alpha_expand) / sqrt_one_minus_alpha_bar_expand * pred_epsilon)

            return x_t_minus_1
        else:
            alpha = self.extract(self.alphas, timestep)
            sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep)
            sqrt_alpha = self.extract(self.sqrt_alphas, timestep)
            
            alpha_expand = alpha[batch_index]
            sqrt_one_minus_alpha_bar_expand = sqrt_one_minus_alpha_bar[batch_index]
            sqrt_alpha_expand = sqrt_alpha[batch_index]

            pred_epsilon = self.model(x_t, condition_node_feature, edge_index, batch_index, timestep)


            pred_x0 = 1/sqrt_alpha_expand * (x_t - (1-alpha_expand) / sqrt_one_minus_alpha_bar_expand * pred_epsilon)

            return pred_x0
    
    def ddpm_sample(self, x, target, grid, criterion=None):
        x_node_feature, x_edge_index, x_batch_index = self.grid2graph(x)
        target_node_feature, _, _ = self.grid2graph(target)
        x_t = torch.randn(x_node_feature.shape).to(self.device)
        
        
        for t in range(self.timesteps-1, 0, -1):
            timestep = torch.tensor([t]).long().to(self.device)
            x_t = self.denoise_at_t(x_t, x_node_feature, x_edge_index, x_batch_index, t)

        loss = criterion(x_t, target_node_feature)

        return x_t, loss
    
if __name__ == '__main__':
    def grid2graph(input_grid):
        batch_size, channels, height, width = input_grid.shape
        
        node_feature = input_grid.permute(0, 2, 3, 1).reshape(-1, channels)

        single_edge_index, _ = grid(height, width)
        single_edge_index = single_edge_index.to(torch.long)
        print(f"single_edge_index: {single_edge_index} {single_edge_index.shape}")
        batch_edge_index = single_edge_index.repeat(1, batch_size)
        print(f"batch_edge_index: {batch_edge_index} {batch_edge_index.shape}")
        batch_offsets = torch.arange(batch_size)
        batch_offsets = batch_offsets.repeat_interleave(single_edge_index.shape[1]) * height*width
        batch_offsets = batch_offsets.unsqueeze(0).repeat(2, 1)
        print(f"batch_offsets: {batch_offsets} {batch_offsets.shape}")
        
        edge_index = (batch_edge_index.unsqueeze(0) + batch_offsets).reshape(2, -1)
        
        batch_index = torch.arange(batch_size).repeat_interleave(height * width)

        return node_feature, edge_index, batch_index
    
    tensor = torch.randn(3, 1, 4, 4)
    node_feature, edge_index, batch_index = grid2graph(tensor)
    
    print(f"tensor: {tensor}")
    print(f"node_feature: {node_feature}")
    print(f"edge_index: {edge_index}")
    print(f"batch_index: {batch_index}")