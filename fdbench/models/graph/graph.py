import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch_cluster import radius_graph
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool, InstanceNorm, avg_pool_x, BatchNorm
from torchdiffeq import odeint

class Swish(nn.Module):
    """
    Swish activation function
    """
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)


class GNN_Layer(MessagePassing):
    """
    Message passing layer
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 time_window: int,
                 in_chans: int,
                 n_variables: int):
        """
        Initialize message passing layers
        Args:
            in_features (int): number of node input features
            out_features (int): number of node output features
            hidden_features (int): number of hidden features
            time_window (int): number of input/output timesteps (temporal bundling)
            n_variables (int): number of equation specific parameters used in the solver
        """
        super(GNN_Layer, self).__init__(node_dim=-2, aggr='mean')
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.in_chans = in_chans

        self.message_net_1 = nn.Sequential(nn.Linear(2 * in_features + time_window*self.in_chans + 2, hidden_features),
                                           Swish()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                           Swish()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(in_features + hidden_features, hidden_features),
                                          Swish()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features),
                                          Swish()
                                          )
        self.norm = InstanceNorm(hidden_features)

    def forward(self, x, u, pos, edge_index, batch):
        """
        Propagate messages along edges
        """
        x = self.propagate(edge_index, x=x, u=u, pos=pos)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j, u_i, u_j, pos_i, pos_j):
        """
        Message update following formula 8 of the paper
        """
        message = self.message_net_1(torch.cat((pos_i - pos_j, x_i, x_j, u_i - u_j), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x):
        """
        Node update following formula 9 of the paper
        """
        update = self.update_net_1(torch.cat((x, message), dim=-1))
        update = self.update_net_2(update)
        if self.in_features == self.out_features:
            return x + update
        else:
            return update


class graph(torch.nn.Module):
    """
    MP-PDE solver class
    """
    def __init__(self,
                 time_window: int = 1,
                 pred_var: int = -4,
                 eq_variables: dict = {},
                 args={}):
        """
        Initialize MP-PDE solver class.
        It contains 6 MP-PDE layers with skip connections
        The input graph to the forward pass has the shape [batch*n_nodes, time_window].
        The output graph has the shape [batch*n_nodes, time_window].
        Args:
            time_window (int): number of input/output timesteps (temporal bundling)
            hidden features (int): number of hidden features
            hidden_layer (int): number of hidden layers
            eq_variables (dict): dictionary of equation specific parameters
        """
        super(graph, self).__init__()
        # 1D decoder CNN is so far designed time_window = [20,25,50]
        self.hidden_features = args.hidden_features
        self.hidden_layer = args.hidden_layer
        self.pred_var = args.pred_var
        self.eq_variables = eq_variables
        self.in_chans = args.in_chans
        
        if args.tem_mod == 'next_step':
            self.forecast_horizon = 1
            self.time_window = 1
        elif args.tem_mod == 'temporal_bundling':
            self.forecast_horizon = 5
            self.time_window = 5

        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=self.time_window,
            in_chans = self.in_chans,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))

        # The last message passing last layer has a fixed output size to make the use of the decoder 1D-CNN easier
        self.gnn_layers.append(GNN_Layer(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=self.time_window,
                                         in_chans = self.in_chans,
                                         n_variables=len(self.eq_variables) + 1
                                        )
                               )

        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.time_window*self.in_chans + 2 + len(self.eq_variables), self.hidden_features),
            Swish(),
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish()
        )

        self.use_odeint = args.use_odeint
        if self.use_odeint:
            self.decoding_mlp = nn.Sequential(nn.Linear(self.hidden_features, self.hidden_features),
                                              Swish(),
                                              nn.Linear(self.hidden_features, 1),
                                              Swish()
                                              )
            # ODEINT derivative network
            self.derivative_net = nn.Sequential(nn.Linear(self.hidden_features, self.hidden_features),
                                                Swish(),
                                                nn.Linear(self.hidden_features, self.hidden_features),
                                                Swish()
                                                )

        else:
            # Decoder CNN, maps to different outputs (temporal bundling)
            if(self.time_window==20):
                self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 15, stride=4),
                                                Swish(),
                                                nn.Conv1d(8, 1, 10, stride=1)
                                                )
            if self.time_window == 1:
                self.output_mlp = nn.Sequential(
                    nn.Conv1d(1, 8, kernel_size=1, stride=1),  # 使用 kernel_size=1 保持输入大小
                    Swish(),
                    nn.Conv1d(8, 1, kernel_size=1, stride=1)  # 输出保持单通道
                )
            if (self.time_window == 25):
                self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 16, stride=3),
                                                Swish(),
                                                nn.Conv1d(8, 1, 14, stride=1)
                                                )
            if(self.time_window==50):
                self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 12, stride=2),
                                                Swish(),
                                                nn.Conv1d(8, 1, 10, stride=1)
                                                )

    def __repr__(self):
        return f'GNN'

    def forward(self, data, target, grid, creterion=None) -> torch.Tensor:
        """
        Forward pass of MP-PDE solver class.
        The input graph has the shape [batch*n_nodes, time_window].
        The output tensor has the shape [batch*n_nodes, time_window].
        Args:
            data (Data): Pytorch Geometric data graph
        Returns:
            torch.Tensor: data output
        """
        # x dim = [b, c, x1, x2]
        print(data.shape,grid.shape,'++++++++'*10) # torch.Size([8, 4, 128, 128]) torch.Size([8, 128, 128, 2])
        pos = data.x[:,0,:2]
        u = data.x[:,:,2:].reshape(data.x.shape[0],-1)
        
        edge_index = data.edge_index
        batch = data.batch

        # Encoder and processor (message passing)
        node_input = torch.cat((pos, u), -1) 

        h = self.embedding_mlp(node_input)
        for i in range(self.hidden_layer):
            h = self.gnn_layers[i](h, u, pos, edge_index, batch)

        if self.use_odeint:
            def ode_func(t, y):
                return self.derivative_net(y)
            
            # timespan for odeint
            t = torch.linspace(0, 1, self.forecast_horizon + 1).to(h.device)

            # use h as initial condition
            pred_z = odeint(ode_func, h, t, method='dopri5')

            pred_z = (pred_z[1:]).permute(1, 0, 2)

            print("pred_z shape (odeint):", pred_z.shape)

            # TODO: figure out where to use self.pred_var... (just change -1 to self.predvar ?)
            out = self.decoding_mlp(pred_z).squeeze(-1)
            # at this point, out should have shape [batch * n_nodes, forecast_horizon]


        else:
            # Decoder (formula 10 in the paper)
            dt = (torch.ones(1, self.time_window)).to(h.device)
            dt = torch.cumsum(dt, dim=1)
            # [batch*n_nodes, hidden_dim] -> 1DCNN([batch*n_nodes, 1, hidden_dim]) -> [batch*n_nodes, time_window]
            diff = self.output_mlp(h[:, None]).squeeze(1)
            print("diff shape (non-odeint):", diff.shape)
            out = u[:, self.pred_var].repeat(self.time_window, 1).transpose(0, 1) + dt * diff

        return out[:,:self.forecast_horizon]

if __name__ == '__main__':
    model = graph()
    
    from torch_geometric.data import Data
    num_nodes = 100

    edges = []
    for node in range(num_nodes):
        num_edges = np.random.randint(0, 4) 
        connections = np.random.choice(num_nodes, num_edges, replace=False)
        for conn in connections:
            edges.append((node, conn))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    node_features = torch.randn(num_nodes, 1, 5) 

    graphs = Data(x=node_features, edge_index=edge_index)
    print(graphs.x.shape)
    out = model(graphs)
    print(out.shape)