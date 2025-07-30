import torch
import torch.nn as nn
from torch_scatter import scatter_add

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
        
        eb_input_dim = 3 * hidden_size
        nb_input_dim = 2 * hidden_size

        self.edge_mlp = build_mlp(eb_input_dim, hidden_size, hidden_size)
        self.node_mlp = build_mlp(nb_input_dim, hidden_size, hidden_size)
    
    def edge_update(self, node_features, edge_index, edge_attr):
        # Edge update
        senders_idx, receivers_idx = edge_index
        senders_attr = node_features[senders_idx]
        receivers_attr = node_features[receivers_idx]
        collected_edges = torch.cat([senders_attr, receivers_attr, edge_attr], dim=-1)
        updated_edge_attr = self.edge_mlp(collected_edges)

        return updated_edge_attr
    
    def node_update(self, node_features, edge_index, updated_edge_attr):
        # Node update
        senders_idx, receivers_idx = edge_index

        aggregated_edges = scatter_add(updated_edge_attr, receivers_idx, dim=0, dim_size=node_features.size(0))
        collected_nodes = torch.cat([node_features, aggregated_edges], dim=-1)
        updated_node_features = self.node_mlp(collected_nodes)

        return updated_node_features

    def forward(self, node_features, edge_index, edge_attr):
        """
        node_features (Tensor): Node feature matrix of shape [num_nodes, node_feature_dim].
        edge_index (Tensor): Edge index tensor of shape [2, num_edges].
        edge_attr (Tensor): Edge feature matrix of shape [num_edges, edge_feature_dim].
        """
        original_node_features = node_features.clone()
        original_edge_attr = edge_attr.clone()

        updated_edge_attr = self.edge_update(node_features, edge_index, edge_attr)
        updated_node_features = self.node_update(node_features, edge_index, updated_edge_attr)

        # Add residual connections
        node_features = original_node_features + updated_node_features
        edge_attr = original_edge_attr + updated_edge_attr

        return node_features, edge_index, edge_attr


class EncodeProcessDecode(nn.Module):
    def __init__(self, args):
        super(EncodeProcessDecode, self).__init__()
        self._latent_size = args.latent_size
        self._output_size = args.output_size
        self._num_layers = args.num_layers
        self._message_passing_steps = args.message_passing_steps
        self._node_input_size = args.node_input_size
        self._edge_input_size = args.edge_input_size
        self._particle_type_latent_size = args.particle_type_latent_size
        self._num_frame = args.seq_length - 1
        self.dt = args.dt

        self.node_embedding_mlp = build_mlp(self._node_input_size + self._particle_type_latent_size, self._latent_size, self._latent_size, lay_norm=True)
        self.edge_embedding_mlp = build_mlp(self._edge_input_size , self._latent_size, self._latent_size, lay_norm=True)
        self.particle_type_mlp = build_mlp(1, self._particle_type_latent_size, self._particle_type_latent_size, lay_norm=True)

        processer_list = []
        for _ in range(self._message_passing_steps):
            processer_list.append(GraphNetBlock(self._latent_size))
        self.processer_list = nn.ModuleList(processer_list)

        self.decoder_mlp = build_mlp(self._latent_size, self._latent_size, self._output_size, lay_norm=False)

    def _encoder(self, x, edge_attr):
        node_latents = self.node_embedding_mlp(x)
        edge_latents = self.edge_embedding_mlp(edge_attr)
        
        return node_latents, edge_latents
    
    def _decoder(self, x):
        node_decoded = self.decoder_mlp(x)
        return node_decoded
    
    def accelertion_to_position(self, last_position, last_velocity, acceleration):
        current_velocity = last_velocity + self.dt * acceleration
        current_position = last_position + self.dt * current_velocity

        return current_position
    
    def forward(self, graph):
        node_feature = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        particle_type = graph.particle_type

        particle_type = self.particle_type_mlp(particle_type.float())
        node_feature = torch.concat([node_feature, particle_type], dim = -1)
        
        node_features, edge_attr = self._encoder(x = node_feature.float(), edge_attr = edge_attr.float())
        for model in self.processer_list:
            node_features, edge_index, edge_attr = model(node_features, edge_index, edge_attr)
        
        decoded = self._decoder(node_features)

        return decoded