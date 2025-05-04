import torch
import h5py
import numpy as np
import copy
import random
from torch.utils.data import Dataset, ConcatDataset


CONSTANTS = {
    "mean": torch.tensor([0.80, 0.0, 0.0, 0.0]).unsqueeze(1).unsqueeze(1),
    "std": torch.tensor([0.31, 0.391, 0.356, 0.185]).unsqueeze(1).unsqueeze(1),
    "time": 20.0,
    "tracer_mean": 0.19586183,
    "tracer_std": 0.37,
}

class DatasetSingle(Dataset):
    def __init__(self, 
                 if_test=False,
                 if_valid=False,
                 test_ratio=0.1,
                 valid_ratio=0.1,
                 normalizer=None,
                 args={}
                 ):
        
        self.reduced_resolution=args.reduced_resolution
        self.reduced_resolution_t=args.reduced_resolution_t
        self.reduced_batch=args.reduced_batch
        initial_step=args.initial_step
        self.tem_mod = args.tem_mod 

        # Time steps used as initial conditions
        if args.tem_mod == 'next_step':
            self.window_size = 1
        elif args.tem_mod == 'auto_regressive':
            self.window_size = args.window_size
        elif args.tem_mod in {'self_atten'}:
            self.window_size = args.window_size
            self.forecast_horizon = args.forecast_horizon
        else:
            self.window_size = args.window_size

        data_path = args.data_path + args.data_set
        self.reader = h5py.File(data_path, "r")
        self.data = self.reader["solution"] # torch.Size([21, 2, 128, 128])
    
        total_samples = self.data.shape[0]
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        test_size = int(total_samples * test_ratio)
        valid_size = int(total_samples * valid_ratio)

        if if_test:
            test_indices_sorted = np.sort(indices[:test_size])
            self.data = self.data[test_indices_sorted]
        elif if_valid:
            valid_indices_sorted = np.sort(indices[test_size:test_size + valid_size])
            self.data = self.data[valid_indices_sorted]
        else:
            train_idx = np.sort(indices[test_size + valid_size:])
            self.data = self.data[train_idx]

        self.resolution = args.input_size

        self.constants = copy.deepcopy(CONSTANTS)
        self.constants["mean"][1] = -2.2424793e-13
        self.constants["mean"][2] = 4.1510376e-12
        self.constants["std"][1] = 0.22017328
        self.constants["std"][2] = 0.22078253

        X, Y = torch.meshgrid(
            torch.linspace(0, 1, self.resolution),
            torch.linspace(0, 1, self.resolution),
            indexing="ij",
        )
        f = lambda x, y: 0.1 * torch.sin(2.0 * np.pi * (x + y))
        self.forcing = f(X, Y).unsqueeze(0)
        self.grid = torch.stack((X, Y), axis=-1)[::self.reduced_resolution, ::self.reduced_resolution]
        self.constants["mean_forcing"] = -1.2996679288335145e-09
        self.constants["std_forcing"] = 0.0707106739282608
        self.forcing = (self.forcing - self.constants["mean_forcing"]) / self.constants[
            "std_forcing"
        ]

    def __len__(self):
        return len(self.data)
    
    @property
    def __normalizer__(self):
        """
        mean and value
        """
        mean_2ch = self.constants["mean"][1:3]
        std_2ch = self.constants["std"][1:3]
    
        return mean_2ch.unsqueeze(0).unsqueeze(0).numpy(), std_2ch.unsqueeze(0).unsqueeze(0).numpy()

    def __getitem__(self, idx):

        max_start = self.data.shape[1] - 2 * self.window_size
        if max_start <= 0:
            raise ValueError("Data length is too short for the given window size.")
        
        rand_idx = random.randint(0, max_start)
        inputs_v = (
            torch.from_numpy(self.data[idx, rand_idx : rand_idx + self.window_size, 0:2])
            .type(torch.float32)
            .reshape(-1, 2, self.resolution, self.resolution)
        )
        label_v = (
            torch.from_numpy(self.data[idx, rand_idx : rand_idx + self.window_size, 0:2])
            .type(torch.float32)
            .reshape(-1, 2, self.resolution, self.resolution)
        )
        self.density = torch.ones(self.window_size, 1, self.resolution, self.resolution)
        self.pressure = torch.zeros(self.window_size, 1, self.resolution, self.resolution)

        inputs = torch.cat([self.density, inputs_v, self.pressure], dim=1)
        label = torch.cat([self.density, label_v, self.pressure], dim=1)

        means = self.constants["mean"].unsqueeze(0).repeat(self.window_size,1,1,1)
        stds = self.constants["std"].unsqueeze(0).repeat(self.window_size,1,1,1)

        input_seq = (inputs - means) / stds
        target_seq = (label - means) / stds

        if self.tem_mod == 'auto_regressive':
            input_seq = torch.cat([input_seq, self.forcing.unsqueeze(0).repeat(self.window_size,1,1,1)], dim=1)
            input_seq = input_seq.permute(2,3,0,1)
            return input_seq, input_seq, self.grid

        elif self.tem_mod == 'self_attn':
            # shape [B, H, W, T, D]
            max_start = self.data.shape[-2] - self.window_size
            if max_start <= 0:
                raise ValueError("Data length is too short for the given window size.")
            rand_idx = random.randint(0, max_start)
            input_seq = self.data[idx, ..., rand_idx : rand_idx + self.forecast_horizon, :]
            target_seq = self.data[idx, ..., rand_idx + self.forecast_horizon : rand_idx + self.window_size, :]
            return input_seq, target_seq, self.grid

        else:
            input_seq = torch.cat([input_seq, self.forcing.unsqueeze(0).repeat(self.window_size,1,1,1)], dim=1)
            target_seq = torch.cat([target_seq, self.forcing.unsqueeze(0).repeat(self.window_size,1,1,1)], dim=1)
            # shape [T, D, H, W]

            input_seq = input_seq.permute(2,3,0,1)
            target_seq = target_seq.permute(2,3,0,1)
            # shape [ H, W, T, D, ]

            if self.window_size == 1:
                input_seq = input_seq.squeeze(-2)
                target_seq = target_seq.squeeze(-2)
            return input_seq, target_seq, self.grid
