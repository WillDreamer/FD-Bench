import os
import math
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def graph_creator(input_frames, target_frame, grid_size, n_neighbors, loop=False):
    x_features = input_frames.reshape(input_frames.shape[0], -1).permute(1, 0)
    y_labels = target_frame.reshape(target_frame.shape[0], -1).permute(1, 0)

    lin = torch.linspace(0, 1, grid_size)
    xx, yy = torch.meshgrid(lin, lin, indexing="ij")
    pos = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    dx = lin[1] - lin[0]
    radius = n_neighbors * dx * math.sqrt(2) + 1e-5

    dist = torch.cdist(pos, pos)
    mask = dist < radius
    if not loop:
        mask.fill_diagonal_(False)
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()

    return x_features, edge_index, pos, y_labels

def load_single_data(args, data_type):
    # file_name = f"{data_type}_data1.npy"
    if not args.use_huggingface:
        file_name = "train.npy"
        data_path = os.path.join(args.dataset_root, file_name)
        data = np.load(data_path)
    else:
        ds = load_dataset(args.dataset_root, split="test")
        ds_dict = ds.to_dict()
        sorted_keys = sorted(ds_dict.keys(), key=lambda x: int(x))
        data = np.stack([
            np.stack([np.array(t) for t in ds_dict[k]])
            for k in sorted_keys
        ])


    # data = data[:, :1000, :, :]
    print(f"{data_type} data: {data.shape}")

    # num_traj, seq_len, _, grid_x, grid_y = data.shape
    # num_traj, seq_len, grid_x, grid_y = data.shape
    
    data = torch.from_numpy(data).float()
    
    inputs_list = []
    targets_list = []

    for seq_idx in range(data.shape[0]):
        # print(f"seq_idx: {seq_idx}")
        for frame_idx in range(data.shape[1] - args.seq_length):
            input_frames = data[seq_idx, frame_idx : (frame_idx + args.seq_length)]
            # print(f"input_frames: {input_frames.shape}")
            target_frame = data[seq_idx, frame_idx + args.seq_length].unsqueeze(0)
            # print(f"target_frame: {target_frame.shape}")

            # input_frames = input_frames.squeeze(0)
            # target_frame = target_frame.squeeze(0)

            inputs_list.append(input_frames)
            targets_list.append(target_frame)

    inputs_tensor = torch.stack(inputs_list)  # shape: (num_samples, seq_length, grid_x, grid_y)
    targets_tensor = torch.stack(targets_list)  # shape: (num_samples, 1, grid_x, grid_y)

    dataset = TensorDataset(inputs_tensor, targets_tensor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"inputs_tensor: {inputs_tensor.shape}")
    print(f"targets_tensor: {targets_tensor.shape}")

    return dataloader


def load_train(args, split="train"):
    file_path = f"{args.data_save_path}/{split}.pth"

    if False and os.path.exists(file_path):
        print(f"{split}.pth already exists. Load from {file_path}")
        all_data = torch.load(file_path)
        print(len(all_data))
        dataloader = DataLoader(all_data, batch_size=args.batch_size, shuffle=True)
    else:
        print(f"{split}.pth does not exists. Create dataset")
        dataloader = load_single_data(args, split)

    return dataloader

def load_valid(args, split="valid"):
    file_path = f"{args.data_save_path}/{split}.pth"

    if os.path.exists(file_path):
        print(f"{split}.pth already exists. Load from {file_path}")
        all_data = torch.load(file_path)
        print(len(all_data))
        dataloader = DataLoader(all_data, batch_size=args.batch_size, shuffle=True)
    else:
        print(f"{split}.pth does not exists. Create dataset")
        dataloader = load_single_data(args, split)

    return dataloader

def load_test(args, split="test"):
    file_path = f"{args.data_save_path}/{split}.pth"

    if os.path.exists(file_path):
        print(f"{split}.pth already exists. Load from {file_path}")
        all_data = torch.load(file_path)
        print(len(all_data))
        dataloader = DataLoader(all_data, batch_size=args.batch_size, shuffle=True)
    else:
        print(f"{split}.pth does not exists. Create dataset")
        dataloader = load_single_data(args, split)

    return dataloader