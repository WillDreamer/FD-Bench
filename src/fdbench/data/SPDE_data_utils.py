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
            self.window_size = initial_step
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

class DatasetGraphSPDE(Dataset):
    def __init__(self, 
                 if_test=False,
                 if_valid=False,
                 test_ratio=0.1,
                 valid_ratio=0.1,
                 normalizer=None,
                 args={}
                 ):
        """
        Dataset for SPDE data structured for graph-based processing
        
        :param args: arguments containing dataset configuration
        """
        # Extract arguments
        self.reduced_resolution = args.reduced_resolution
        self.reduced_resolution_t = args.reduced_resolution_t
        self.reduced_batch = args.reduced_batch
        initial_step = args.initial_step
        saved_folder = args.data_path
        filename = args.data_set

        # Time steps used as initial conditions
        if args.tem_mod == 'next_step':
            self.window_size = 1
        elif args.tem_mod == 'auto_regressive':
            self.window_size = initial_step
        else:
            self.window_size = initial_step

        # Load the mat file
        root_path = os.path.join(os.path.abspath(saved_folder), filename)
        mat_data = sio.loadmat(root_path)
        
        # Extract data fields
        _sol = np.array(mat_data['sol'], dtype=np.float32)  # (N, x, t+1) or (N, x, y, t+1)
        _t = np.array(mat_data['t'], dtype=np.float32)  # time points
        
        # Set default constants for normalization
        self.constants = {
            "mean_sol": 0.0,
            "std_sol": 1.0,
            "mean_forcing": 0.0,
            "std_forcing": 1.0,
        }
        
        # Determine data dimensionality and process accordingly
        if len(_sol.shape) == 3:  # 1D spatial + time
            # Shape: (N, x, t+1)
            n_samples, n_x, n_t = _sol.shape
            self.resolution = n_x
            
            # Apply reductions
            self.data = _sol[::self.reduced_batch, ::self.reduced_resolution, ::self.reduced_resolution_t]
            
            # Create grid
            x_coords = np.linspace(0, 1, n_x)[::self.reduced_resolution]
            self.grid = torch.tensor(x_coords, dtype=torch.float).unsqueeze(-1)
            
            # Create forcing (similar to KF_data_utils)
            X = torch.linspace(0, 1, n_x)[::self.reduced_resolution]
            f = lambda x: 0.1 * torch.sin(2.0 * np.pi * x)
            self.forcing = f(X).unsqueeze(0)
            
        elif len(_sol.shape) == 4:  # 2D spatial + time
            # Shape: (N, x, y, t+1)
            n_samples, n_x, n_y, n_t = _sol.shape
            self.resolution = n_x  # Assuming square grid
            
            # Apply reductions
            self.data = _sol[::self.reduced_batch, ::self.reduced_resolution, ::self.reduced_resolution, ::self.reduced_resolution_t]
            
            # Create grid
            X, Y = torch.meshgrid(
                torch.linspace(0, 1, n_x)[::self.reduced_resolution],
                torch.linspace(0, 1, n_y)[::self.reduced_resolution],
                indexing="ij"
            )
            self.grid = torch.stack((X, Y), axis=-1)
            
            # Create forcing (similar to KF_data_utils)
            f = lambda x, y: 0.1 * torch.sin(2.0 * np.pi * (x + y))
            self.forcing = f(X, Y).unsqueeze(0)
        
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
        
        # Calculate normalization constants if they're not provided
        if not if_test and not if_valid:
            if len(self.data.shape) == 3:  # 1D spatial case
                self.constants["mean_sol"] = np.mean(self.data)
                self.constants["std_sol"] = np.std(self.data)
            else:  # 2D spatial case
                self.constants["mean_sol"] = np.mean(self.data)
                self.constants["std_sol"] = np.std(self.data)
            
            self.constants["mean_forcing"] = torch.mean(self.forcing).item()
            self.constants["std_forcing"] = torch.std(self.forcing).item() or 1.0
        else:
            # Use provided normalizer
            if normalizer is not None:
                self.constants = normalizer
        
        # Normalize forcing
        self.forcing = (self.forcing - self.constants["mean_forcing"]) / self.constants["std_forcing"]

    def __len__(self):
        return len(self.data)
    
    @property
    def __normalizer__(self):
        """
        Return normalization constants
        """
        return self.constants

    def __getitem__(self, idx):
        # Calculate maximum start index for time window
        if len(self.data.shape) == 3:  # 1D spatial case
            max_start = self.data.shape[2] - 2 * self.window_size
        else:  # 2D spatial case
            max_start = self.data.shape[3] - 2 * self.window_size
            
        if max_start <= 0:
            raise ValueError("Data length is too short for the given window size.")
        
        rand_idx = random.randint(0, max_start)
        
        # Get input sequences based on data dimensionality
        if len(self.data.shape) == 3:  # 1D spatial case
            input_sol = torch.from_numpy(
                self.data[idx, :, rand_idx:rand_idx+self.window_size]
            ).type(torch.float32)
            
            target_sol = torch.from_numpy(
                self.data[idx, :, rand_idx+self.window_size:rand_idx+2*self.window_size]
            ).type(torch.float32)
            
            # Normalize solution
            input_sol = (input_sol - self.constants["mean_sol"]) / self.constants["std_sol"]
            target_sol = (target_sol - self.constants["mean_sol"]) / self.constants["std_sol"]
            
            # Add channel dimension
            input_sol = input_sol.unsqueeze(-1)
            target_sol = target_sol.unsqueeze(-1)
            
            # Add forcing channel
            forcing_repeated = self.forcing.unsqueeze(1).repeat(1, self.window_size, 1)
            input_seq = torch.cat([input_sol, forcing_repeated.unsqueeze(-1)], dim=-1)
            target_seq = torch.cat([target_sol, forcing_repeated.unsqueeze(-1)], dim=-1)
            
            # Reshape to [x, t, channels]
            input_seq = input_seq.permute(0, 1, 2)
            target_seq = target_seq.permute(0, 1, 2)
            
        else:  # 2D spatial case
            input_sol = torch.from_numpy(
                self.data[idx, :, :, rand_idx:rand_idx+self.window_size]
            ).type(torch.float32)
            
            target_sol = torch.from_numpy(
                self.data[idx, :, :, rand_idx+self.window_size:rand_idx+2*self.window_size]
            ).type(torch.float32)
            
            # Normalize solution
            input_sol = (input_sol - self.constants["mean_sol"]) / self.constants["std_sol"]
            target_sol = (target_sol - self.constants["mean_sol"]) / self.constants["std_sol"]
            
            # Add channel dimension
            input_sol = input_sol.unsqueeze(-1)
            target_sol = target_sol.unsqueeze(-1)
            
            # Add forcing channel
            forcing_repeated = self.forcing.unsqueeze(1).repeat(1, self.window_size, 1, 1)
            input_seq = torch.cat([input_sol, forcing_repeated.unsqueeze(-1)], dim=-1)
            target_seq = torch.cat([target_sol, forcing_repeated.unsqueeze(-1)], dim=-1)
            
            # Reshape to [x, y, t, channels]
            input_seq = input_seq.permute(0, 1, 2, 3)
            target_seq = target_seq.permute(0, 1, 2, 3)
        
        # Squeeze time dimension if window_size is 1
        if self.window_size == 1:
            input_seq = input_seq.squeeze(-2)
            target_seq = target_seq.squeeze(-2)
        
        return input_seq, target_seq, self.grid