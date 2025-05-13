
import torch
from torch.utils.data import Dataset, IterableDataset
from fdbench.utils.utils import tprint
import os
import glob
import h5py
import numpy as np
import math as mt
import random

class DatasetSingle(Dataset):
    def __init__(self, 
                 if_test=False,
                 if_valid=False,
                 test_ratio=0.1,
                 valid_ratio=0.1,
                 num_samples_max = -1,
                 normalizer=None,
                 args={}
                 ):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 1
        :type initial_step: INT, optional

        """
        filename = args.data_set
        reduced_resolution=args.reduced_resolution
        reduced_resolution_t=args.reduced_resolution_t
        reduced_batch=args.reduced_batch
        saved_folder = args.data_path
        self.tem_mod=args.tem_mod
        self.args=args
        
        root_path = os.path.join(os.path.abspath(saved_folder), filename)

        if filename[-2:] == 'h5':  # SWE-2D (RDB)
        
            with h5py.File(root_path, 'r') as f:
                keys = list(f.keys())
                keys.sort()
                data_arrays = [np.array(f[key]['data'], dtype=np.float32) for key in keys]
                _data = torch.from_numpy(np.stack(data_arrays, axis=0))   # [batch, nt, nx, ny, nc]
                _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution, ...]
                _data = torch.permute(_data, (0, 2, 3, 1, 4))   # [batch, nx, ny, nt, nc]
                gridx, gridy = np.array(f['0023']['grid']['x'], dtype=np.float32), np.array(f['0023']['grid']['y'], dtype=np.float32)
                mgridX, mgridY = np.meshgrid(gridx, gridy, indexing='ij')
                _grid = torch.stack((torch.from_numpy(mgridX), torch.from_numpy(mgridY)), axis=-1)
                self.grid = _grid[::reduced_resolution, ::reduced_resolution, ...]
                _tsteps_t = torch.from_numpy(np.array(f['0023']['grid']['t'], dtype=np.float32))
                tsteps_t = _tsteps_t[::reduced_resolution_t]
                self.data = _data
                self.grid = _grid
                self.tsteps_t = tsteps_t

        total_samples = self.data.shape[0]
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        test_size = int(total_samples * test_ratio)
        valid_size = int(total_samples * valid_ratio)

        if if_test:
            self.data = self.data[indices[:test_size]]
        elif if_valid:
            self.data = self.data[indices[test_size:test_size + valid_size]]
        else:
            self.data = self.data[indices[test_size + valid_size:]]
        
        if not if_test and not if_valid:
            self.train_mean = torch.mean(self.data, dim=(0, 1, 2, 3), keepdim=True)
            self.train_std = torch.std(self.data, dim=(0, 1, 2, 3), keepdim=True)

            # 将为 0 的标准差替换为 1
            self.train_std = torch.where(self.train_std == 0,
                                        torch.ones_like(self.train_std),
                                        self.train_std)
            
        else:
            self.train_mean, self.train_std = normalizer
        
        self.data = (self.data - self.train_mean) / self.train_std

        # Time steps used as initial conditions
        if hasattr(args, "if_rollout") and args.if_rollout:
            self.window_size=self.data.shape[-2] - 1
        else:
            if args.tem_mod == 'next_step':
                self.window_size = 1
            elif args.tem_mod in {'self_atten','node'}:
                self.window_size = args.window_size
                self.forecast_horizon = args.forecast_horizon
            else:
                self.window_size = args.window_size

        self.data = self.data if torch.is_tensor(self.data) else torch.tensor(self.data)
        # [B,128,128,101,2]

    def __len__(self):
        return len(self.data)
    
    @property
    def __normalizer__(self):
        """
        mean and value
        """
        return self.train_mean, self.train_std

    def __getitem__(self, idx):
        if self.tem_mod == 'auto_regressive':
            # shape [B, H, W, T, D]
            max_start = self.data.shape[-2] - self.window_size
            if max_start <= 0:
                raise ValueError("Data length is too short for the given window size.")
            rand_idx = random.randint(0, max_start)
            input_seq = self.data[idx, ..., rand_idx : rand_idx + self.window_size, :]
            if self.window_size == 1:
                input_seq = input_seq.squeeze(-2)

            return input_seq, input_seq, self.grid
        elif self.tem_mod in {'self_attn', 'node'}:
            # shape [B, H, W, T, D]
            max_start = self.data.shape[-2] - self.window_size
            if max_start <= 0:
                raise ValueError("Data length is too short for the given window size.")
            rand_idx = random.randint(0, max_start)
            input_seq = self.data[idx, ..., rand_idx : rand_idx + self.forecast_horizon, :]
            target_seq = self.data[idx, ..., rand_idx + self.forecast_horizon : rand_idx + self.window_size, :]
            return input_seq, target_seq, self.grid
        else:
            max_start = max(self.data.shape[-2] - 2 * self.window_size,0)
            
            rand_idx = random.randint(0, max_start)
            # shape [B, H, W, T, D]
            if hasattr(self.args, "if_rollout") and self.args.if_rollout:
                input_seq = self.data[idx, ..., rand_idx : rand_idx + self.window_size, :]
                return input_seq, input_seq, self.grid
            else:
                input_seq = self.data[idx, ..., rand_idx : rand_idx + self.window_size, :]
                target_seq = self.data[idx, ..., rand_idx + self.window_size : rand_idx + 2 * self.window_size, :]
                if self.window_size == 1:
                    input_seq = input_seq.squeeze(-2)
                    target_seq = target_seq.squeeze(-2)

                return input_seq, target_seq, self.grid
