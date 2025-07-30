import os
import h5py
import os.path as osp
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

def compute_eulerian(positions, velocity, num_points=32, h=0.01, n_neighbors=2, loop=False):
    """
    Parameters:
      positions: (num_frames, num_particles, 2)
      N: int
      h: float, smoothing length for the kernel.

    Returns:
      eulerian_velocity: (num_frames-1, num_points, 2)
    """
    grid_points = create_uniform_grid(num_points)  # shape: (num_points, 2)

    pos = positions  # shape: (T, num_particles, 2)
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

    return eulerian_velocity

def create_edge(second_last_frame, connectivity_radius):
    diffs = second_last_frame.unsqueeze(1) - second_last_frame.unsqueeze(0)  # Shape: (2500, 2500, 2)
    distances = torch.norm(diffs, dim=-1)  # Shape: (2500, 2500)
        
    mask = (distances < connectivity_radius) & ~torch.eye(distances.size(0), dtype=torch.bool)  # Shape: (2500, 2500)
    edge_index = mask.nonzero(as_tuple=False).T
        
    src, dst = edge_index
    relative_positions = diffs[src, dst]  # Shape: (num_edges, 2)
    edge_distances = distances[src, dst].unsqueeze(1)  # Shape: (num_edges, 1)
    edge_attr = torch.cat((relative_positions, edge_distances), dim=-1)  # Shape: (num_edges, 3)

    return edge_index, edge_attr

def create_subsequences(position, particle_type, seq_length, connectivity_radius, split_interval, dt):
    num_frames, num_particles, _ = position.shape
    ls_data = []

    start_idx = 0
    while (start_idx + seq_length) < num_frames:
        print(f"start_idx: {start_idx}, num_frames: {num_frames}")
        subseq = position[start_idx : start_idx + seq_length]  # Shape: (7, 2500, 2)
        
        x = subseq[:-1] # (6, 2500, 2)
        y = subseq[-1]  # Shape: (2500, 2)
        
        velocity = x[1:, :, :] - x[:-1, :, :]
        # print(f"velocity: {velocity.shape}")
        velocity = velocity.permute(1, 0, 2).contiguous()
        # print(f"velocity: {velocity.shape}")
        velocity = velocity.view(num_particles, -1)
        # print(f"velocity: {velocity.shape}")

        
        edge_index, edge_attr = create_edge(x[-1, :, :], connectivity_radius)
        data = Data(x=velocity, particle_type=particle_type, edge_index=edge_index, edge_attr=edge_attr, y=y)
        ls_data.append(data)

        start_idx += split_interval

    return ls_data

def load_single_dataset(dataset_root, split, batch_size, seq_length, connectivity_radius, data_save_path, split_interval, max_data, dt):
    p = osp.join(dataset_root, split)
    all_data = []

    if not args.use_huggingface:
        with h5py.File(p, "r") as f:
            # print(f"f: {f}")
            print(list(f.keys()))
            for file_key in f.keys():
                tmp_data = f[file_key]
                print(tmp_data)
                
                particle_type = torch.from_numpy(tmp_data["particle_type"][:])
                position = torch.from_numpy(tmp_data["position"][:])
                particle_type = particle_type.unsqueeze(-1)

                # position = position[:20000, :, :]
                # print(f"position: {position.shape}")
                
                ls_data = create_subsequences(position, particle_type, seq_length, connectivity_radius, split_interval, dt)
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

    torch.save(all_data, f"{data_save_path}/{split}.pth")

    dataloader = DataLoader(all_data, batch_size=batch_size, shuffle=True)

    return dataloader


def load_train(args):
    dataset_root = args.dataset_root
    batch_size = args.batch_size
    seq_length = args.seq_length
    split_interval = args.split_interval
    connectivity_radius = args.connectivity_radius
    data_save_path = args.data_save_path
    max_data = args.max_train_data
    dt = args.dt
    split = "train.h5"

    file_path = f"{data_save_path}/{split}.pth"
    if False and os.path.exists(file_path):
        print(f"{split}.pth already exists. Load from {file_path}")
        all_data = torch.load(file_path)
        print(len(all_data))
        dataloader = DataLoader(all_data, batch_size=batch_size, shuffle=True)
    else:
        print(f"{split}.pth does not exists. Create dataset")
        dataloader = load_single_dataset(dataset_root, split, batch_size, seq_length, connectivity_radius, data_save_path, split_interval, max_data, dt)
    
    return dataloader