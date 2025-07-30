import os
import h5py
import os.path as osp
import math
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader

def create_uniform_grid(N):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()]) # (num_points, 2)
    return torch.from_numpy(grid_points)

def quintic_kernel(r, h):
    one_over_h = 1.0 / h
    sigma_2d = 7.0 / (478.0 * np.pi) * one_over_h**2
    q = r * one_over_h
    q1 = np.maximum(0.0, 1.0 - q)
    q2 = np.maximum(0.0, 2.0 - q)
    q3 = np.maximum(0.0, 3.0 - q)

    return sigma_2d * (q3**5 - 6.0 * q2**5 + 15.0 * q1**5)

def compute_eulerian(positions, num_points=32, h=0.0125, n_neighbors=2, loop=False):
    """
    Parameters:
      positions: (num_frames, num_particles, 2)
      N: int
      h: float, smoothing length for the kernel.

    Returns:
      eulerian_velocity: (num_frames-1, num_points, 2)
    """
    velocity = positions[1:] - positions[:-1]
    grid_points = create_uniform_grid(num_points)  # shape: (num_points, 2)

    pos = positions[1:]  # shape: (T, num_particles, 2)
    # (1, num_points, 1, 2) - (T, 1, num_particles, 2)
    diff = grid_points[None, :, None, :] - pos[:, None, :, :] # (T, num_points, num_particles, 2)

    r = np.linalg.norm(diff, axis=-1) # (T, num_points, num_particles)
    weights = quintic_kernel(r, h)  # shape: (T, num_points, num_particles)
    weights = torch.from_numpy(weights)

    # (T, num_points, num_particles, 1) * (T, 1 num_particles, 2)
    numerator = torch.sum(weights[..., None] * velocity[:, None, :, :], axis=2)
    denominator = torch.sum(weights, axis=2)  # shape: (T, num_points)
    
    epsilon = 1e-8
    eulerian_velocity = numerator / (denominator[..., None] + epsilon)

    eulerian_velocity = eulerian_velocity.permute(1, 0, 2)

    lin = torch.linspace(0, 1, num_points)
    xx, yy = torch.meshgrid(lin, lin, indexing="ij")
    pos = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    dx = lin[1] - lin[0]
    radius = n_neighbors * dx * math.sqrt(2) + 1e-5
    dist = torch.cdist(pos, pos)
    mask = dist < radius
    
    if not loop:
        mask.fill_diagonal_(False)
    
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()

    src, dst = edge_index
    edge_vector = pos[dst] - pos[src]
    edge_norm = edge_vector.norm(dim=1, keepdim=True)
    edge_attr = torch.cat((edge_vector, edge_norm), dim=1)
    
    return eulerian_velocity, edge_index, edge_attr


def create_subsequences(position, seq_length, split_interval):
    num_frames, num_particles, _ = position.shape
    ls_data = []

    start_idx = 0
    while (start_idx + seq_length) < num_frames:
        print(f"start_idx: {start_idx}, num_frames: {num_frames}")
        subseq = position[start_idx : start_idx + seq_length]  # Shape: (7, 2500, 2)
        
        velocity, edge_index, edge_attr = compute_eulerian(subseq)

        velocity = velocity[:, :-1, :]
        target = velocity[:, -1, :]

        velocity = velocity.reshape(velocity.shape[0], -1)
        target = target.reshape(target.shape[0], -1)

        data = Data(x=velocity, y=target, edge_index=edge_index, edge_attr=edge_attr)

        ls_data.append(data)

        start_idx += split_interval

    return ls_data

def load_single_dataset(dataset_root, split, batch_size, seq_length, data_save_path, split_interval, max_data):
    p = osp.join(dataset_root, split)
    all_data = []

    if not args.use_huggingface:
        with h5py.File(p, "r") as f:
            print(list(f.keys()))
            for file_key in f.keys():
                tmp_data = f[file_key]
                print(tmp_data)
                
                particle_type = torch.from_numpy(tmp_data["particle_type"][:])
                position = torch.from_numpy(tmp_data["position"][:])
                particle_type = particle_type.unsqueeze(-1)

                position = position[:20000, :, :]
                
                ls_data = create_subsequences(position, seq_length, split_interval)
                all_data = all_data + ls_data
                print(f"len(all_data): {len(all_data)}")
    else:
        ds = load_dataset(args.dataset_root, split="test")
        ds_dict = ds.to_dict()
        data = {k: [np.array(x) for x in v] for k, v in ds_dict.items()}

        for tmp_data in data:               
            position = torch.from_numpy(tmp_data)

            ls_data = create_subsequences(position, seq_length, split_interval)
            all_data = all_data + ls_data
            print(f"len(all_data): {len(all_data)}")
            

    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

    if len(all_data) > max_data:
        all_data = all_data[:max_data]

    torch.save(
        all_data, 
        f"{data_save_path}/{split}_mgn.pth"
    )

    dataloader = DataLoader(all_data, batch_size=batch_size, shuffle=False)

    return dataloader


def load_train(args):
    dataset_root = args.dataset_root
    batch_size = args.batch_size
    seq_length = args.seq_length
    split_interval = args.split_interval
    data_save_path = args.data_save_path
    max_data = args.max_train_data
    split = "train.h5"

    file_path = f"{data_save_path}/{split}_mgn.pth"
    if os.path.exists(file_path):
        print(f"{split}.pth already exists. Load from {file_path}")
        all_data = torch.load(file_path)
        print(len(all_data))
        dataloader = DataLoader(all_data, batch_size=batch_size, shuffle=False)
    else:
        print(f"{split}.pth does not exists. Create dataset")
        dataloader = load_single_dataset(dataset_root, split, batch_size, seq_length, data_save_path, split_interval, max_data)
    
    return dataloader