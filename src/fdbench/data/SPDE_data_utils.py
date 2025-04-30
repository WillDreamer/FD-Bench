import torch
from torch.utils.data import Dataset
import os
import scipy.io as sio
import numpy as np
import random
import math as mt
import copy

class DatasetSPDESingle(Dataset):
    def __init__(self, 
                 if_test=False,
                 if_valid=False,
                 test_ratio=0.1,
                 valid_ratio=0.1,
                 num_samples_max=-1,
                 normalizer=None,
                 args={}
                 ):
        """
        Dataset for SPDE data stored in scipy mat files
        
        :param args: arguments containing dataset configuration
        """
        filename = args.data_set
        reduced_resolution = args.reduced_resolution
        reduced_resolution_t = args.reduced_resolution_t
        reduced_batch = args.reduced_batch
        saved_folder = args.data_path
        initial_step = args.initial_step
        window_size = args.window_size
        self.tem_mod = args.tem_mod
        # Load the mat file
        root_path = os.path.join(os.path.abspath(saved_folder), filename)
        mat_data = sio.loadmat(root_path)
        
        # Extract data fields
        _sol = np.array(mat_data['sol'], dtype=np.float32)  # (N, x, t+1) or (N, x, y, t+1)
        # We're not using the forcing term for now
        _t = np.array(mat_data['t'], dtype=np.float32)  # time points
        _param = np.array(mat_data['param'])
        _forcing = np.array(mat_data['forcing'], dtype=np.float32)
        # Determine data dimensionality
        if len(_sol.shape) == 3:  # 1D spatial + time
            # Shape: (N, x, t+1)
            n_samples, n_x, n_t = _sol.shape
            
            # Apply reductions
            _sol = _sol[::reduced_batch, ::reduced_resolution, ::reduced_resolution_t]
            
            # Convert to format [batch, x, t, channels]
            _sol = np.expand_dims(np.transpose(_sol, (0, 1, 2)), axis=-1)

            # Create data array with only solution
            self.data = _sol      # solution as the only channel
            
            # Create grid
            x_coords = np.linspace(0, 1, n_x)[::reduced_resolution]
            self.grid = torch.tensor(x_coords, dtype=torch.float).unsqueeze(-1)
            
        elif len(_sol.shape) == 4:  # 2D spatial + time
            # Shape: (N, x, y, t+1)
            n_samples, n_x, n_y, n_t = _sol.shape
            
            # Apply reductions
            _sol = _sol[::reduced_batch, ::reduced_resolution, ::reduced_resolution, ::reduced_resolution_t]
            
            _sol = np.expand_dims(_sol, axis=-1)

            # Convert to format [batch, x, y, t, cxhannels]
            _sol = np.transpose(_sol, (0, 1, 2, 3,4))

            # Create data array with only solution
            self.data = _sol      # solution as the only channel
            
            # Create grid
            x_coords = np.linspace(0, 1, n_x)[::reduced_resolution]
            y_coords = np.linspace(0, 1, n_y)[::reduced_resolution]
            x = torch.tensor(x_coords, dtype=torch.float)
            y = torch.tensor(y_coords, dtype=torch.float)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            self.grid = torch.stack((X, Y), axis=-1)
        
        # Split data into train/valid/test sets
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
        
        # Normalize data
        if not if_test and not if_valid:
            if len(self.data.shape) == 4:  # 1D
                self.train_mean = np.mean(self.data, axis=(0, 1, 2), keepdims=True)
                self.train_std = np.std(self.data, axis=(0, 1, 2), keepdims=True)
            else:  # 2D
                self.train_mean = np.mean(self.data, axis=(0, 1, 2, 3), keepdims=True)
                self.train_std = np.std(self.data, axis=(0, 1, 2, 3), keepdims=True)
            
            self.train_std = np.where(self.train_std == 0, 1, self.train_std)
        else:
            self.train_mean, self.train_std = normalizer
        
        self.data = (self.data - self.train_mean) / self.train_std
        
        # Time steps used as initial conditions
        if args.tem_mod == 'next_step':
            self.window_size = 1
        elif args.tem_mod in {'auto_regressive', 'temporal_bundling'}:
            self.window_size = window_size
        else:
            self.window_size = initial_step

        self.data = self.data if torch.is_tensor(self.data) else torch.tensor(self.data)

    def __len__(self):
        return len(self.data)
    
    @property
    def __normalizer__(self):
        """
        Return mean and std for normalization
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
        else:
            max_start = self.data.shape[-2] - 2 * self.window_size
            if max_start <= 0:
                raise ValueError("Data length is too short for the given window size.")
            rand_idx = random.randint(0, max_start)
            input_seq = self.data[idx, ..., rand_idx : rand_idx + self.window_size, :]
            target_seq = self.data[idx, ..., rand_idx + self.window_size : rand_idx + 2 * self.window_size, :]
            if self.window_size == 1:
                input_seq = input_seq.squeeze(-2)
                target_seq = target_seq.squeeze(-2)
            
            return input_seq, target_seq, self.grid

class DatasetDRSPDE(Dataset):
    def __init__(self, 
                 if_test=False,
                 if_valid=False,
                 test_ratio=0.1,
                 valid_ratio=0.1,
                 num_samples_max=-1,
                 normalizer=None,
                 args={}
                 ):
        """
        Dataset for SPDE data in DR-style format
        
        :param args: arguments containing dataset configuration
        """
        filename = args.data_set
        reduced_resolution = args.reduced_resolution
        reduced_resolution_t = args.reduced_resolution_t
        reduced_batch = args.reduced_batch
        saved_folder = args.data_path
        initial_step = args.initial_step
        
        # Load the mat file
        root_path = os.path.join(os.path.abspath(saved_folder), filename)
        mat_data = sio.loadmat(root_path)
        
        # Extract data fields
        _sol = np.array(mat_data['sol'], dtype=np.float32)  # (N, x, t+1) or (N, x, y, t+1)
        _t = np.array(mat_data['t'], dtype=np.float32)  # time points
        
        # Determine data dimensionality
        if len(_sol.shape) == 3:  # 1D spatial + time
            # Shape: (N, x, t+1)
            n_samples, n_x, n_t = _sol.shape
            
            # Apply reductions
            _sol = _sol[::reduced_batch, ::reduced_resolution, ::reduced_resolution_t]
            
            # Convert to format [batch, x, t, channels]
            _sol = np.expand_dims(np.transpose(_sol, (0, 1, 2)), axis=-1)
            
            # Create data array with only solution
            self.data = torch.from_numpy(_sol)
            
            # Create grid
            x_coords = np.linspace(0, 1, n_x)[::reduced_resolution]
            self.grid = torch.tensor(x_coords, dtype=torch.float).unsqueeze(-1)
            self.tsteps_t = torch.tensor(_t[::reduced_resolution_t], dtype=torch.float)
            
        elif len(_sol.shape) == 4:  # 2D spatial + time
            # Shape: (N, x, y, t+1)
            n_samples, n_x, n_y, n_t = _sol.shape
            
            # Apply reductions
            _sol = _sol[::reduced_batch, ::reduced_resolution, ::reduced_resolution, ::reduced_resolution_t]
            
            # Expand dimensions to add channel
            _sol = np.expand_dims(_sol, axis=-1)
            
            # Convert to format [batch, x, y, t, channels]
            self.data = torch.from_numpy(_sol)
            
            # Create grid
            x_coords = np.linspace(0, 1, n_x)[::reduced_resolution]
            y_coords = np.linspace(0, 1, n_y)[::reduced_resolution]
            x = torch.tensor(x_coords, dtype=torch.float)
            y = torch.tensor(y_coords, dtype=torch.float)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            self.grid = torch.stack((X, Y), axis=-1)
            self.tsteps_t = torch.tensor(_t[::reduced_resolution_t], dtype=torch.float)

        # Split data into train/valid/test sets
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
        
        # Normalize data
        if not if_test and not if_valid:
            if len(self.data.shape) == 4:  # 1D spatial case: [batch, x, t, channels]
                self.train_mean = torch.mean(self.data, dim=(0, 1, 2), keepdim=True)
                self.train_std = torch.std(self.data, dim=(0, 1, 2), keepdim=True)
            else:  # 2D spatial case: [batch, x, y, t, channels]
                self.train_mean = torch.mean(self.data, dim=(0, 1, 2, 3), keepdim=True)
                self.train_std = torch.std(self.data, dim=(0, 1, 2, 3), keepdim=True)
            
            # Replace zero standard deviations with ones
            self.train_std = torch.where(self.train_std == 0,
                                        torch.ones_like(self.train_std),
                                        self.train_std)
        else:
            self.train_mean, self.train_std = normalizer
        
        self.data = (self.data - self.train_mean) / self.train_std
        
        # Time steps used as initial conditions
        if args.tem_mod == 'next_step':
            self.window_size = 1
        elif args.tem_mod == 'auto_regressive':
            self.window_size = initial_step
        else:
            self.window_size = initial_step

    def __len__(self):
        return len(self.data)
    
    @property
    def __normalizer__(self):
        """
        Return mean and std for normalization
        """
        return self.train_mean, self.train_std

    def __getitem__(self, idx):
        max_start = self.data.shape[-2] - 2 * self.window_size
        if max_start <= 0:
            raise ValueError("Data length is too short for the given window size.")
        
        rand_idx = random.randint(0, max_start)
        
        # Get input and target sequences
        input_seq = self.data[idx, ..., rand_idx : rand_idx + self.window_size, :]
        target_seq = self.data[idx, ..., rand_idx + self.window_size : rand_idx + 2 * self.window_size, :]
        
        if self.window_size == 1:
            input_seq = input_seq.squeeze(-2)
            target_seq = target_seq.squeeze(-2)

        return input_seq, target_seq, self.grid 

CONSTANTS = {
    "mean": torch.tensor([0.80, 0.0, 0.0, 0.0]).unsqueeze(1).unsqueeze(1),
    "std": torch.tensor([0.31, 0.391, 0.356, 0.185]).unsqueeze(1).unsqueeze(1),
    "time": 20.0,
    "tracer_mean": 0.19586183,
    "tracer_std": 0.37,
}

class DatasetGraphSPDE(Dataset):
    def __init__(self, 
                 if_test=False,
                 if_valid=False,
                 test_ratio=0.1,
                 valid_ratio=0.1,
                 normalizer=None,
                 args={}
                 ):
        
        self.reduced_resolution = args.reduced_resolution
        self.reduced_resolution_t = args.reduced_resolution_t
        self.reduced_batch = args.reduced_batch
        initial_step = args.initial_step

        # Time steps used as initial conditions
        if args.tem_mod == 'next_step':
            self.window_size = 1
        elif args.tem_mod == 'auto_regressive':
            self.window_size = initial_step
        else:
            self.window_size = initial_step

        filename = args.data_set
        saved_folder = args.data_path
        
        # Load the mat file
        root_path = os.path.join(os.path.abspath(saved_folder), filename)
        mat_data = sio.loadmat(root_path)
        
        # Extract data fields
        _sol = np.array(mat_data['sol'], dtype=np.float32)  # (N, x, y, t+1)
        _t = np.array(mat_data['t'], dtype=np.float32)  # time points
        
        # Assume 2D spatial + time
        n_samples, n_x, n_y, n_t = _sol.shape
        
        # Apply reductions
        _sol = _sol[::self.reduced_batch, ::self.reduced_resolution, ::self.reduced_resolution, ::self.reduced_resolution_t]
        
        # Expand dimensions to add channel
        _sol = np.expand_dims(_sol, axis=-1)
        
        # Convert to tensor
        self.data = torch.from_numpy(_sol)
        
        # Split data into train/valid/test sets
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

        # Create grid
        self.resolution = args.input_size
        X, Y = torch.meshgrid(
            torch.linspace(0, 1, n_x)[::self.reduced_resolution],
            torch.linspace(0, 1, n_y)[::self.reduced_resolution],
            indexing="ij"
        )
        self.grid = torch.stack((X, Y), axis=-1)
        
        # Normalize data
        if not if_test and not if_valid:
            # Calculate mean and std on the training data
            self.train_mean = torch.mean(self.data, dim=(0, 1, 2, 3), keepdim=True)
            self.train_std = torch.std(self.data, dim=(0, 1, 2, 3), keepdim=True)
            
            # Replace zero standard deviations with ones
            self.train_std = torch.where(self.train_std == 0,
                                        torch.ones_like(self.train_std),
                                        self.train_std)
        else:
            self.train_mean, self.train_std = normalizer
        
        # Normalize the data
        self.data = (self.data - self.train_mean) / self.train_std
        
    def __len__(self):
        return len(self.data)
    
    @property
    def __normalizer__(self):
        """
        Return mean and std for normalization
        """
        return self.train_mean, self.train_std

    def __getitem__(self, idx):
        # Get a random starting index for the time window
        max_start = self.data.shape[3] - 2 * self.window_size
        if max_start <= 0:
            raise ValueError("Data length is too short for the given window size.")
        
        rand_idx = random.randint(0, max_start)
        
        # Get input and target sequences
        input_seq = self.data[idx, :, :, rand_idx:rand_idx + self.window_size, :]
        target_seq = self.data[idx, :, :, rand_idx + self.window_size:rand_idx + 2 * self.window_size, :]
        # Reshape for model input: [H, W, T, C]
        # input_seq = input_seq.permute(1, 2, 3, 0)
        # target_seq = target_seq.permute(1, 2, 3, 0)
        
        if self.window_size == 1:
            input_seq = input_seq.squeeze(2)
            target_seq = target_seq.squeeze(2)


        return input_seq, target_seq, self.grid
