import argparse
from argparse import Namespace
import datetime
import math
import numpy as np
import torch
import os, sys, socket
import json
from pathlib import Path
import importlib
import random
from copy import deepcopy
import logging
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots

from fdbench.utils.utils import *
from fdbench.utils.metrics import metric_func

logger = logging.getLogger(__name__)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def main(args):
    
    # Set up device (CPU only)
    device = torch.device('cpu')
    
    # Set up logging and experiment directory
    from datetime import datetime
    current_time = datetime.now().strftime("%m%d-%H:%M")
    
    exp_name = "PDE_" + args.PDE_type + '_' + args.spa_mod + '_' +  args.tem_mod + '_' + args.pred_tgt
    os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    
    save_dir = os.path.join(args.output_dir, (exp_name+'_'+args.remark+'_'+current_time))
    os.makedirs(save_dir, exist_ok=True)
    args_dict = vars(args)
    # Save to a JSON file
    json_dir = os.path.join(save_dir, "args.json")
    with open(json_dir, 'w') as f:
        json.dump(args_dict, f, indent=4)
    checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(save_dir)
    logger.info(f"Experiment directory created at {save_dir}")
      
    if args.seed is not None:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    logger.info(args)

    #>>>>>> ===============================Model Design==================================
    if args.pred_tgt == 'variable':
        module_name = 'fdbench.models.' + args.spa_mod
    class_name = args.spa_mod
    module = getattr(importlib.import_module(module_name),class_name)
    model = module(args=args)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) 
    n_parameters = count_parameters(model)/(1024**2)
    logger.info(model)
    logger.info(f"Number of Parameters: {n_parameters} Mb")
        
    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    #<<<<<< =================================================================

    #>>>>>> =============================Data Reading==========================
    data_module_name = 'fdbench.data.' + args.PDE_type + '_data_utils'
    
    # Select the appropriate dataset class based on spatial model
    if args.spa_mod == 'graph':
        dataset_class = 'DatasetGraphSPDE'
    elif args.spa_mod == 'latent':
        dataset_class = 'DatasetDRSPDE'
    else:
        dataset_class = 'DatasetSPDESingle'
    
    print(f"dataset_class: {dataset_class}")
    
    data_module = getattr(importlib.import_module(data_module_name), dataset_class)
    train_data = data_module(args = args)
    normalizer = train_data.__normalizer__
    test_data = data_module(if_test=True,args = args,normalizer=normalizer)
    val_data = data_module(if_valid=True,args = args,normalizer=normalizer)

    if not args.spa_mod == 'graph':
        data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                            num_workers=args.num_workers)
        data_loader_test = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                            num_workers=args.num_workers)
        data_loader_val = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                            num_workers=args.num_workers)
    else:
        sample_nodes = 1024
        rand_idx = torch.randperm(args.input_size ** 2)[:sample_nodes]  # Random select N nodes
        from fdbench.data.graph_data import get_graph_dataloader
        data_loader_train, normalizer_new = get_graph_dataloader(train_data, rand_idx, batch_size=args.batch_size, normalizer=normalizer, normalizer_new=None, is_train=True, k=args.neighbor)
        data_loader_val, _ = get_graph_dataloader(val_data, rand_idx, batch_size=args.batch_size, normalizer=normalizer, normalizer_new=normalizer_new, is_train=False, k=args.neighbor)
        data_loader_test, _ = get_graph_dataloader(test_data, rand_idx, batch_size=args.batch_size, normalizer=normalizer, normalizer_new=normalizer_new, is_train=False, k=args.neighbor)
    #<<<<<< =================================================================
    max_train_steps = int(args.epochs * len(data_loader_train))

    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,)   
    if args.scheduler == 'CosineAnnealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*2)
    elif args.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs//3, T_mult=2, eta_min=1e-8)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs, gamma=0.1)
    elif args.scheduler == 'cyc':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr//5, max_lr=args.lr,
                                              mode = 'triangular2', gamma = 0.95,
                                              step_size_up=args.step_size_up, step_size_down=args.step_size_down,cycle_momentum=False)  

    elif args.scheduler == 'lambda':
        from torch.optim.lr_scheduler import LambdaLR
        warm_up_steps = int(0.1*max_train_steps)
        def lr_lambda(current_step):
            if current_step < warm_up_steps:
                return float(current_step)/float(max(1, warm_up_steps))
            progress = float(current_step - warm_up_steps) / float(max(1, max_train_steps - warm_up_steps))
            return 0.5*(1+math.cos(math.pi*progress))
        scheduler = LambdaLR(optimizer, lr_lambda)
        
    criterion = torch.nn.MSELoss()

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            )
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']
        ema.load_state_dict(ckpt['ema'])

    from tqdm import tqdm
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=global_step,
        desc="Steps",
    )
    
    ##### Start Training
    for epoch in range(args.start_epoch, args.epochs):
        
        #### =========1. Data Loading=========
        for batch in data_loader_train:
            if hasattr(batch, 'x') and hasattr(batch, 'y'):
                data = batch.to(device)
                samples = data
                targets = data.y
                grid = getattr(data, 'grid', None)
            else:
                samples, targets, grid = batch
                if len(samples.shape) == 4:
                    samples = samples.permute(0, 3, 1, 2).to(device, non_blocking=True)
                    targets = targets.permute(0, 3, 1, 2).to(device, non_blocking=True)
                elif len(samples.shape) == 5:
                    # [B, H, W, T, D]
                    samples = samples.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    H_field = samples.shape[1]
                    W_field = samples.shape[2]
                    B_field = samples.shape[0]
                grid = grid.to(device) if grid is not None else None

            model.train()

            #### =========2. Model Training=========
            print(f"samples.shape: {samples.shape}")
            print(f"targets.shape: {targets.shape}")
            print(f"grid.shape: {grid.shape}")
            if args.tem_mod == 'next_step':
                outputs, loss = model(samples, targets, grid, criterion)
            elif args.tem_mod == 'auto_regressive':
                loss = 0
                for tt in range(int(args.window_size) - int(args.initial_step)):
                    sample_t = samples[...,tt:tt+args.initial_step,:].reshape(B_field,-1,H_field,W_field)
                    target_t = samples[...,tt+args.initial_step,:].permute(0, 3, 1, 2)
                    output_t, loss_batch = model(sample_t, target_t, grid, criterion)
                    loss += loss_batch

                    print(f"sample_t.shape: {sample_t.shape}")
                    print(f"target_t.shape: {target_t.shape}")
            optimizer.zero_grad()
            loss.backward()
            
            # Manual gradient clipping
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update EMA model
            update_ema(ema, model)

            progress_bar.update(1)
            global_step += 1   

            #### =========3. CKPT Saving=========
            if global_step % args.checkpointing_steps == 0 and global_step > (max_train_steps//2):
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": optimizer.state_dict(),
                    "args": args,
                    "steps": global_step,}
                
                checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            #### =========4. Model Testing=========
            if global_step == 1 or (global_step % args.eval_steps == 0 and global_step > 0) or global_step==max_train_steps:
                model.eval()  # important! This disables randomized embedding dropout
                
                _err_RMSE_avg = 0
                _err_nRMSE_avg = 0
                _err_max_avg = 0
                _err_csv_avg = 0
                _err_BD_avg = 0
                _err_F_avg = 0
                
                # Variable to store a sample for visualization
                vis_sample = None
                vis_target = None
                vis_output = None
                
                with torch.no_grad():
                    
                    for batch in data_loader_val:
                        if hasattr(batch, 'x') and hasattr(batch, 'y'):
                            data = batch.to(device)
                            input_test = data
                            target_test = data.y
                            grid = getattr(data, 'grid', None)
                        else:
                            input_test, target_test, grid = batch
                            if len(input_test.shape) == 4:
                                input_test = input_test.permute(0, 3, 1, 2).to(device, non_blocking=True)
                                target_test = target_test.permute(0, 3, 1, 2).to(device, non_blocking=True)
                            elif len(input_test.shape) == 5:
                                # [B, H, W, T, D]
                                input_test = input_test.to(device, non_blocking=True)
                                target_test_raw = target_test.to(device, non_blocking=True)
                                H_field = input_test.shape[1]
                                W_field = input_test.shape[2]
                                B_field = input_test.shape[0]
                            grid = grid.to(device) if grid is not None else None

                        if args.spa_mod == "diffusion" or args.spa_mod == "graph_diffusion":
                            if args.sample_method == "ddpm":
                                model = model.ddpm_sample
                            else:
                                model = model.ddim_sample
                            outputs, loss = model(input_test,target_test,grid,criterion)
                        elif args.tem_mod == 'next_step':
                            outputs, loss = model(input_test,target_test,grid,criterion)
                        elif args.tem_mod == 'auto_regressive':
                            rolling_input = input_test[..., :args.initial_step, :].clone()  # (B, H, W, initial_step, C)
                            predicted_outputs = []
                            
                            for tt in range(args.window_size - args.initial_step):
                                input_test_t = rolling_input.reshape(B_field, -1, H_field, W_field)
                                target_test_t = target_test_raw[..., tt+args.initial_step, :].permute(0, 3, 1, 2)
                                
                                output_test_t, _ = model(input_test_t, target_test_t, grid, criterion)
                                predicted_outputs.append(output_test_t.unsqueeze(1))  
                                rolling_input = torch.cat([
                                    rolling_input[..., 1:, :],       
                                    output_test_t.permute(0,2,3,1).unsqueeze(-2)        
                                ], dim=-2) 
                            outputs = torch.cat(predicted_outputs, dim=1).contiguous()
                            target_test = target_test_raw[..., args.initial_step:, :].permute(0, 3, 4, 1, 2).contiguous()
                        
                        # Store a sample for visualization (only once)
                        if vis_sample is None:
                            vis_sample = input_test.detach().cpu()
                            vis_target = target_test.detach().cpu()
                            vis_output = outputs.detach().cpu()
                        
                        if hasattr(batch, 'x') and hasattr(batch, 'y'):
                            batch_size = batch.num_graphs
                        else:
                            batch_size = input_test.shape[0]
                        
                        Lx, Ly, Lz = 1., 1., 1.
                        _err_RMSE, _err_nRMSE, _err_CSV, _err_Max, _err_BD, _err_F \
                        = metric_func(outputs, target_test, batch_size, if_mean=True, Lx=Lx, Ly=Ly, Lz=Lz)

                        _err_RMSE_avg += _err_RMSE.item()
                        _err_nRMSE_avg += _err_nRMSE.item()
                        _err_max_avg += _err_Max.item()
                        _err_csv_avg += _err_CSV.item()
                        _err_F_avg += _err_F.item()
                        _err_BD_avg += _err_BD.item()
                    _err_RMSE_avg /= len(data_loader_val)
                    _err_nRMSE_avg /= len(data_loader_val)
                    _err_max_avg /= len(data_loader_val)
                    _err_csv_avg /= len(data_loader_val)
                    _err_F_avg /= len(data_loader_val)
                    _err_BD_avg /= len(data_loader_val)
                    logger.info(f'RMSE: {_err_RMSE_avg:.4f}, nRMSE: {_err_nRMSE_avg:.4f}, fRMSE:{_err_F_avg:.4f}, MAX-ERR:{_err_max_avg:.4f}, BD:{_err_BD_avg:.4f}, CSV:{_err_csv_avg:.4f}')
                    
                    # Generate visualization plots every 100 epochs
                    current_epoch = global_step // len(data_loader_train)
                    if current_epoch % 100 == 0 or global_step == max_train_steps:
                        print("visualization")
                        print(f"Creating visualization at epoch {current_epoch}")
                        vis_dir = os.path.join(save_dir, "visualizations", args.spa_mod)
                        os.makedirs(vis_dir, exist_ok=True)
                        
                        # Determine the shape and prepare data for plotting
                        if len(vis_output.shape) == 4:  # [B, C, H, W]
                            # For next_step mode
                            sample_idx = 0  # First sample in batch
                            
                            # Get number of channels to visualize (multiple timesteps)
                            num_channels = min(vis_output.shape[1], 4)  # Visualize up to 4 channels/timesteps
                            
                            for channel_idx in range(num_channels):
                                # Get the 2D fields
                                prediction = vis_output[sample_idx, channel_idx].numpy()
                                ground_truth = vis_target[sample_idx, channel_idx].numpy()
                                
                                # Create and save prediction plot
                                plt.figure(figsize=(6, 6))
                                plt.contourf(prediction, levels=20, cmap='coolwarm')
                                plt.axis('off')  # Turn off axis
                                plt.savefig(f"{vis_dir}/pred_epoch{current_epoch}_ts{channel_idx}.pdf", 
                                          bbox_inches='tight', pad_inches=0.1, format='pdf')
                                plt.close()
                                
                                # Create and save ground truth plot
                                plt.figure(figsize=(6, 6))
                                plt.contourf(ground_truth, levels=20, cmap='coolwarm')
                                plt.axis('off')  # Turn off axis
                                plt.savefig(f"{vis_dir}/truth_epoch{current_epoch}_ts{channel_idx}.pdf", 
                                          bbox_inches='tight', pad_inches=0.1, format='pdf')
                                plt.close()
                            
                        elif len(vis_output.shape) == 5:  # [B, T, C, H, W]
                            # For auto_regressive mode
                            sample_idx = 0  # First sample in batch
                            channel_idx = 0  # First channel
                            
                            # Get number of time steps to visualize
                            num_timesteps = min(vis_output.shape[1], 4)  # Visualize up to 4 timesteps
                            
                            for time_idx in range(num_timesteps):
                                # Get the 2D fields
                                prediction = vis_output[sample_idx, time_idx, channel_idx].numpy()
                                ground_truth = vis_target[sample_idx, time_idx, channel_idx].numpy()
                                
                                # Create and save prediction plot
                                plt.figure(figsize=(6, 6))
                                plt.contourf(prediction, levels=20, cmap='coolwarm')
                                plt.axis('off')  # Turn off axis
                                plt.savefig(f"{vis_dir}/pred_epoch{current_epoch}_ts{time_idx}.pdf", 
                                          bbox_inches='tight', pad_inches=0.1, format='pdf')
                                plt.close()
                                
                                # Create and save ground truth plot
                                plt.figure(figsize=(6, 6))
                                plt.contourf(ground_truth, levels=20, cmap='coolwarm')
                                plt.axis('off')  # Turn off axis
                                plt.savefig(f"{vis_dir}/truth_epoch{current_epoch}_ts{time_idx}.pdf", 
                                          bbox_inches='tight', pad_inches=0.1, format='pdf')
                                plt.close()
                            
                        logger.info(f"Visualizations saved to {vis_dir}")
                    
                    # Calculate model stats (simplified without thop)
                    if global_step == 10:
                        mem_usage_MB = 0  # No GPU memory tracking on CPU
                        logger.info(f"Model parameters: {n_parameters:.2f} MB")
            
            if global_step == 10 or (global_step % args.eval_steps == 0 and global_step > 0) or global_step==max_train_steps:
                model.eval()  # important! This disables randomized embedding dropout
                
                _err_RMSE_avg = 0
                _err_nRMSE_avg = 0
                _err_max_avg = 0
                _err_csv_avg = 0
                _err_BD_avg = 0
                _err_F_avg = 0
                with torch.no_grad():
                    for batch in data_loader_test:
                        if hasattr(batch, 'x') and hasattr(batch, 'y'):
                            data = batch.to(device)
                            input_test = data
                            target_test = data.y
                            grid = getattr(data, 'grid', None)
                        else:
                            input_test, target_test, grid = batch
                            if len(input_test.shape) == 4:
                                input_test = input_test.permute(0, 3, 1, 2).to(device, non_blocking=True)
                                target_test = target_test.permute(0, 3, 1, 2).to(device, non_blocking=True)
                            elif len(input_test.shape) == 5:
                                # [B, H, W, T, D]
                                input_test = input_test.to(device, non_blocking=True)
                                target_test = target_test.to(device, non_blocking=True)
                                H_field = input_test.shape[1]
                                W_field = input_test.shape[2]
                                B_field = input_test.shape[0]
                            grid = grid.to(device) if grid is not None else None

                        if args.spa_mod == "diffusion" or args.spa_mod == "graph_diffusion":
                            if args.sample_method == "ddpm":
                                samp_algo = model.ddpm_sample
                            else:
                                samp_algo = model.ddim_sample
                            outputs, loss = samp_algo(input_test,target_test,grid,criterion)
                        elif args.tem_mod == 'next_step':
                            outputs, loss = model(input_test,target_test,grid,criterion)
                        elif args.tem_mod == 'auto_regressive':
                            rolling_input = input_test[..., :args.initial_step, :].clone()  # (B, H, W, initial_step, C)
                            predicted_outputs = []
                            
                            for tt in range(args.window_size - args.initial_step):
                                input_test_t = rolling_input.reshape(B_field, -1, H_field, W_field)
                                target_test_t = input_test[..., tt+args.initial_step, :].permute(0,3,1,2)
                                
                                output_test_t, _ = model(input_test_t, target_test_t, grid, criterion)
                                predicted_outputs.append(output_test_t.unsqueeze(1))  
                                rolling_input = torch.cat([
                                    rolling_input[..., 1:, :],       
                                    output_test_t.permute(0,2,3,1).unsqueeze(-2)        
                                ], dim=-2) 
                            outputs = torch.cat(predicted_outputs, dim=1).contiguous()
                            target_test = input_test[..., args.initial_step:, :].permute(0, 3, 4, 1, 2).contiguous()
                        
                        if hasattr(batch, 'x') and hasattr(batch, 'y'):
                            batch_size = batch.num_graphs
                        else:
                            batch_size = input_test.shape[0]
                        
                        Lx, Ly, Lz = 1., 1., 1.
                        _err_RMSE, _err_nRMSE, _err_CSV, _err_Max, _err_BD, _err_F \
                        = metric_func(outputs, target_test, batch_size, if_mean=True, Lx=Lx, Ly=Ly, Lz=Lz)

                        _err_RMSE_avg += _err_RMSE.item()
                        _err_nRMSE_avg += _err_nRMSE.item()
                        _err_max_avg += _err_Max.item()
                        _err_csv_avg += _err_CSV.item()
                        _err_F_avg += _err_F.item()
                        _err_BD_avg += _err_BD.item()
                    _err_RMSE_avg /= len(data_loader_test)
                    _err_nRMSE_avg /= len(data_loader_test)
                    _err_max_avg /= len(data_loader_test)
                    _err_csv_avg /= len(data_loader_test)
                    _err_F_avg /= len(data_loader_test)
                    _err_BD_avg /= len(data_loader_test)
                    logger.info(f'RMSE: {_err_RMSE_avg:.4f}, nRMSE: {_err_nRMSE_avg:.4f}, fRMSE:{_err_F_avg:.4f}, MAX-ERR:{_err_max_avg:.4f}, BD:{_err_BD_avg:.4f}, CSV:{_err_csv_avg:.4f}')

            # Log metrics to console
            logs = {
                "lr": current_lr,
                "loss": loss.item(),    
            }
            progress_bar.set_postfix(**logs)
        
        scheduler.step()
        
    logger.info("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--remark", type=str, default=' ', help="Training remark")
    default_args = parser.parse_args()
    
    args = get_config(config_path=default_args.config_file)
    args = Namespace(**vars(default_args), **vars(args))
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ## Check for complex data
    if args.spa_mod == 'fourier' or args.spa_mod == 'frequency':
        print("Note: Running complex operations on CPU may be slower")
    main(args) 