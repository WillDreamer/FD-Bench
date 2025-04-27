import math
import torch
import torch.nn as nn
from torch_scatter import scatter_add

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

def build_mlp(in_size, hidden_size, out_size, lay_norm=True):
    module = nn.Sequential(nn.Linear(in_size, hidden_size),
                           nn.ReLU(),
                           nn.Linear(hidden_size, hidden_size),
                           nn.ReLU(),
                           nn.Linear(hidden_size, hidden_size),
                           nn.ReLU(),
                           nn.Linear(hidden_size, out_size))
    if lay_norm:
        return nn.Sequential(module,  nn.LayerNorm(normalized_shape=out_size))
    
    return module

class GraphNetBlock(nn.Module):
    def __init__(self, hidden_size):
        super(GraphNetBlock, self).__init__()
        
        eb_input_dim = 2 * hidden_size
        nb_input_dim = 2 * hidden_size

        self.edge_mlp = build_mlp(eb_input_dim, hidden_size, hidden_size)
        self.node_mlp = build_mlp(nb_input_dim, hidden_size, hidden_size)
    
    def edge_update(self, node_features, edge_index):
        # Edge update
        senders_idx, receivers_idx = edge_index
        senders_attr = node_features[senders_idx]
        receivers_attr = node_features[receivers_idx]
        collected_edges = torch.cat([senders_attr, receivers_attr], dim=-1)
        updated_edge_attr = self.edge_mlp(collected_edges)

        return updated_edge_attr
    
    def node_update(self, node_features, edge_index, updated_edge_attr):
        # Node update
        senders_idx, receivers_idx = edge_index

        aggregated_edges = scatter_add(updated_edge_attr, receivers_idx, dim=0, dim_size=node_features.size(0))
        collected_nodes = torch.cat([node_features, aggregated_edges], dim=-1)
        updated_node_features = self.node_mlp(collected_nodes)

        return updated_node_features

    def forward(self, node_features, edge_index):
        """
        node_features (Tensor): Node feature matrix of shape [num_nodes, node_feature_dim].
        edge_index (Tensor): Edge index tensor of shape [2, num_edges].
        edge_attr (Tensor): Edge feature matrix of shape [num_edges, edge_feature_dim].
        """
        original_node_features = node_features.clone()

        updated_edge_attr = self.edge_update(node_features, edge_index)
        updated_node_features = self.node_update(node_features, edge_index, updated_edge_attr)

        # Add residual connections
        node_features = original_node_features + updated_node_features

        return node_features, edge_index

class MeshGraphNetsDenoiser(nn.Module):
    def __init__(self, args):
        super(MeshGraphNetsDenoiser, self).__init__()
        self.sinu_pos_emb = SinusoidalPosEmb(dim = args.denoiser_time_dim, theta = args.denoiser_theta)
        self.time_mlp = nn.Sequential(
            self.sinu_pos_emb,
            nn.Linear(args.denoiser_time_dim, args.denoiser_time_dim),
            nn.GELU(),
            nn.Linear(args.denoiser_time_dim, args.denoiser_time_dim))
        
        self.node_embedding_mlp = build_mlp(args.denoiser_in_channels+args.denoiser_time_dim, args.denoiser_hidden_channels, args.denoiser_hidden_channels, lay_norm=True)

        processer_list = []
        for _ in range(args.denoiser_message_passing_steps):
            processer_list.append(GraphNetBlock(args.denoiser_hidden_channels))
        self.processer_list = nn.ModuleList(processer_list)

        self.decoder_mlp = build_mlp(args.denoiser_hidden_channels, args.denoiser_hidden_channels, args.denoiser_out_channels, lay_norm=True)

    def forward(self, perturbed_target, condition_node_feature, edge_index, batch_index, t):
        t = self.time_mlp(t)
        t = t[batch_index]
        

        node_feature = torch.concat([perturbed_target, condition_node_feature, t], dim = -1)
        node_feature = self.node_embedding_mlp(node_feature)

        for proc in self.processer_list:
            node_feature, _ = proc(node_feature, edge_index)

        node_feature = self.decoder_mlp(node_feature)

        return node_feature


