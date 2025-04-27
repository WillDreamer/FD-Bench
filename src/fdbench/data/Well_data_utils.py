
import torch
from torch.utils.data import Dataset, IterableDataset
from fdbench.utils.utils import tprint
import os
from the_well.data import WellDataset
from torch.utils.data import DataLoader
import numpy as np
import random

class DatasetSingle(WellDataset):
    def __init__(self, 
                 if_test=False,
                 if_valid=False,
                 test_ratio=0.1,
                 valid_ratio=0.1,
                 normalizer=None,
                 args={}
                 ):
        
        filename = args.data_set
        self.reduced_resolution=args.reduced_resolution
        self.reduced_resolution_t=args.reduced_resolution_t
        self.reduced_batch=args.reduced_batch
        saved_folder = args.data_path
        self.well_dataset_name = args.well_dataset_name
        initial_step=args.initial_step

         # Time steps used as initial conditions
        if args.tem_mod == 'next_step':
            self.window_size = 1
        elif args.tem_mod == 'auto_regressive':
            self.window_size = initial_step
        else:
            self.window_size = initial_step
        

        # The following line may take a couple of minutes to instantiate the datamodule
        # 'MHD_64', 'supernova_explosion_64', 'shear_flow
        if if_test:
            split_name = "test"
        elif if_valid:
            split_name = "valid"
        else:
            split_name = "train"
        super(DatasetSingle, self).__init__(
            well_base_path=saved_folder,
            well_dataset_name=self.well_dataset_name,
            include_filters=[filename],  
            well_split_name=split_name,
            use_normalization=True,      
            return_grid=True,
            n_steps_input = self.window_size,
            n_steps_output = self.window_size
        )

    def __len__(self):
        
        return super().__len__()
    
    @property
    def __normalizer__(self):
        """
        mean and value
        """
        all_tensors = []
        for field_name, field_mean in self.means.items():
            all_tensors.append(field_mean.reshape(-1))

        cat_means = torch.cat(all_tensors, dim=0)  
        self.train_mean = cat_means.view(1, 1, 1, 1, -1)

        all_tensors = []
        for field_name, fieldstd in self.stds.items():
            all_tensors.append(fieldstd.reshape(-1))

        cat_std = torch.cat(all_tensors, dim=0)  
        self.train_std = cat_std.view(1, 1, 1, 1, -1)

        return self.train_mean, self.train_std

    def __getitem__(self, idx):
        sample = super(DatasetSingle, self).__getitem__(idx)

        if self.well_dataset_name == 'shear_flow':
            reduced_resolution_long = self.reduced_resolution * 2
        else:
            reduced_resolution_long = self.reduced_resolution

        input_seq = sample.get("input_fields", None)[:,::self.reduced_resolution,::reduced_resolution_long,:]    # [T, H, W, D]
        target_seq = sample.get("output_fields", None)[:,::self.reduced_resolution,::reduced_resolution_long,:]  
        grid = sample.get("space_grid", None)[::self.reduced_resolution,::reduced_resolution_long,:]           

        if input_seq is not None:
            Ti = input_seq.shape[0] 
                
            if Ti >= 2 * self.window_size:
                rand_idx = random.randint(0, Ti - 2 * self.window_size)
                # shape [ T, H, W,  D, ]
                new_input = input_seq[rand_idx : rand_idx + self.window_size]
                new_target = input_seq[rand_idx + self.window_size : rand_idx + 2 * self.window_size]
                new_input = new_input.permute(1,2,3,0)
                new_target = new_target.permute(1,2,3,0)
                # shape [ H, W, T, D, ]
                return new_input, new_target, grid
            else:
                input_seq = input_seq.squeeze(0)
                target_seq = target_seq.squeeze(0)
                return input_seq, target_seq, grid
