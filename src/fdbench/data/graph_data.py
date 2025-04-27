import torch
import numpy as np
from fdbench.utils.utils import tprint
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, DataLoader
from torch_geometric.transforms import VirtualNode


def compute_knn_graph(coords, k):
    coords_np = coords.cpu().numpy()
    knn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(coords_np)
    distances, indices = knn.kneighbors(coords_np)
    
    # Build edge_index
    row = torch.arange(coords.size(0)).unsqueeze(1).repeat(1, k).flatten()
    col = torch.tensor(indices).flatten()
    edge_index = torch.stack([row, col], dim=0)
    
    return edge_index

def get_graph_dataloader(dataset, rand_idx, batch_size, normalizer, normalizer_new=None, is_train=True, k=20, num_workers=1, shuffle=True):
    data_list = []
    first_iter = True
    # transform = VirtualNode()
    train_mean, train_std = normalizer
    dataset.data = (dataset.data * train_std) + train_mean

    if is_train:
        new_mean = dataset.data.mean(dim=(0, 1, 2, 3), keepdim=True)
        new_std = dataset.data.std(dim=(0, 1, 2, 3), keepdim=True)
        new_std = torch.where(new_std == 0, torch.ones_like(new_std), new_std)
        dataset.data = (dataset.data - new_mean) / new_std
    else:
        new_mean, new_std = normalizer_new
        dataset.data = (dataset.data - new_mean) / new_std

    for i in range(len(dataset)):
        x, y, grid = dataset[i]
        var_dim = x.shape[-1]

        all_grady = torch.cat([
            x[:, 1:2, :] - x[:, 0:1, :],            
            (x[:, 2:, :] - x[:, :-2, :]) / 2,        
            x[:, -1:, :] - x[:, -2:-1, :],                    
        ], dim=1)
        all_gradx = torch.cat([
            x[1:2,:,:] - x[0:1, :, :],            
            (x[2:,:,:] - x[:-2,:,:]) / 2,        
            x[-1:,:,:] - x[-2:-1,:,:],                    
        ], dim=0)

        x = torch.cat([all_gradx,all_grady,x],dim=-1)
        all_var_dim = x.shape[-1]

        if first_iter:
            tprint("x, y, grid shape (in get_graph_dataloader)")
            tprint(x.shape,y.shape,grid.shape,'++++++++'*10)
            first_iter = False
        
        if len(x.shape) == 4:
            temporal_dim = x.shape[-2]
        else:
            temporal_dim = 1

        coords = grid.reshape(-1, 2)  # shape: [16384, 2]
        node_coords = coords[rand_idx]  # shape: [1000, 2]
        x = x.reshape(-1, temporal_dim*all_var_dim)[rand_idx]
        y = y.reshape(-1, temporal_dim*var_dim)[rand_idx]

        edge_index = compute_knn_graph(node_coords, k=k)
        
        senders = edge_index[0].numpy()
        receivers = edge_index[1].numpy()
        crds_diff = x[senders] - x[receivers]
        crds_norm = np.linalg.norm(crds_diff, axis=1, keepdims=True)
        edge_attr = np.concatenate((crds_diff, crds_norm), axis=1)
        edge_attr = torch.from_numpy(edge_attr)

        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, pos=node_coords)
        # data = transform(data)

        data_list.append(data)
        
    dataloader = DataLoader(data_list, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader, (new_mean, new_std)