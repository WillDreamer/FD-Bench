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
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from fdbench.utils.utils import *
from fdbench.utils.metrics import metric_func
import warnings
warnings.filterwarnings('ignore')

logger = get_logger(__name__)


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
    
    logger = get_logger(__name__)
    logging_dir = Path(args.output_dir, args.tensorboard_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    from datetime import datetime
    current_time = datetime.now().strftime("%m%d-%H:%M")

    if accelerator.is_main_process:
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
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
        torch.backends.cudnn.enabled = True
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

    if accelerator.is_main_process:
        logger.info(args)
    
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    #>>>>>> ===============================Model Design==================================
    if args.pred_tgt == 'variable':
        module_name = 'fdbench.models.' + args.spa_mod
    class_name = args.spa_mod
    module = getattr(importlib.import_module(module_name),class_name)
    model = module(args=args)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) 
    n_parameters = count_parameters(model)/(1024**2)
    if accelerator.is_main_process:
        logger.info(model)
        logger.info(f"Number of Parameters: {n_parameters} Mb")
        
    model = model.to(accelerator.device)
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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs, gamma=0.1)  # 根据需要调整参数
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
    
    model, optimizer, data_loader_train, scheduler = accelerator.prepare(
        model, optimizer, data_loader_train, scheduler
    )
    accelerator.register_for_checkpointing(scheduler)
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=exp_name,  
        )

    from tqdm import tqdm
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
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
            with accelerator.accumulate(model):
                if args.tem_mod == 'next_step':
                    outputs, loss = model(samples,targets,grid,criterion)
                
                elif args.tem_mod == 'auto_regressive':
                    loss = 0
                    for tt in range(int(args.window_size) - int(args.initial_step)):

                        sample_t = samples[...,tt:tt+args.initial_step,:].reshape(B_field,-1,H_field,W_field)
                        target_t = samples[...,tt+args.initial_step,:].permute(0, 3, 1, 2)
                        output_t, loss_batch = model(sample_t, target_t, grid, criterion)
                        loss += loss_batch

                optimizer.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.clip_grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                current_lr = optimizer.param_groups[0]['lr']
            
                if accelerator.sync_gradients:
                    update_ema(ema, model) # change ema function

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1   

            #### =========3. CKPT Saving=========
            if global_step % args.checkpointing_steps == 0 and global_step > (max_train_steps//2):
                if accelerator.is_main_process:
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
            if global_step == 10 or (global_step % args.eval_steps == 0 and global_step > 0) or global_step==max_train_steps:
                model.eval()  # important! This disables randomized embedding dropout

                torch.cuda.empty_cache()
                
                _err_RMSE_avg = 0
                _err_nRMSE_avg = 0
                _err_max_avg = 0
                _err_csv_avg = 0
                _err_BD_avg = 0
                _err_F_avg = 0
                with torch.no_grad():
                    
                    for batch in data_loader_val:
                        if hasattr(batch, 'x') and hasattr(batch, 'y'):
                            data = batch.to(device)
                            input_test = data
                            target_test = data.y
                            grid = getattr(data, 'grid', None)
                        else:
                            input_test, target_test, grid = batch
                            if len(samples.shape) == 4:
                                input_test = input_test.permute(0, 3, 1, 2).to(device, non_blocking=True)
                                target_test = target_test.permute(0, 3, 1, 2).to(device, non_blocking=True)
                            elif len(samples.shape) == 5:
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
                                input_test = rolling_input.reshape(B_field, -1, H_field, W_field)
                                target_test = target_test_raw[..., tt+args.initial_step, :].permute(0, 3, 1, 2)
                                
                                output_test_t, _ = model(input_test, target_test, grid, criterion)
                                predicted_outputs.append(output_test_t.unsqueeze(1))  
                                rolling_input = torch.cat([
                                    rolling_input[..., 1:, :],       
                                    output_test_t.permute(0,2,3,1).unsqueeze(-2)        
                                ], dim=-2) 
                            outputs = torch.cat(predicted_outputs, dim=1).contiguous()  # B [T] C H W
                            target_test = target_test_raw[..., args.initial_step:, :].permute(0, 3, 4, 1, 2)
                        
                        if hasattr(batch, 'x') and hasattr(batch, 'y'):
                            batch_size = batch.num_graphs
                        else:
                            batch_size = input_test.shape[0]
                        #     outputs = outputs.unsqueeze(-1).unsqueeze(-1)
                        #     # outputs, target_test, mask = remove_virtual_nodes(outputs, target_test, batch.ptr)
                        
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
                    val_log = {"val/val_RMSE": _err_RMSE_avg, "val/val_nRMSE": _err_nRMSE_avg, "val/fRMSE":_err_F_avg, 'val/MAX-ERR':_err_max_avg, 'val/CSV':_err_csv_avg, 'val/BD':_err_BD_avg}
                    accelerator.log(val_log, step=global_step)

                    if global_step == 10 and accelerator.is_main_process:
                        from thop import profile
                        target_model = model.module if hasattr(model, "module") else model
                        if len(target_test.shape) == 4:
                            flops, params = profile(target_model, inputs=(input_test,target_test,grid,criterion))
                        elif args.tem_mod == 'auto_regressive':
                            flops, params = profile(target_model, inputs=(rolling_input.reshape(B_field, -1, H_field, W_field),target_test_raw[..., 0, :].permute(0, 3, 1, 2),grid,criterion))
                        gflops = flops / 1e9
                        mem_alloc_MB = torch.cuda.memory_allocated(device) / (1024 ** 2)
                        accelerator.log({
                            "model/num_params": n_parameters,
                            "model/GFlops": gflops,
                            "model/memory_MB": mem_alloc_MB,
                        }, step=global_step)
            
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
                            if len(samples.shape) == 4:
                                input_test = input_test.permute(0, 3, 1, 2).to(device, non_blocking=True)
                                target_test = target_test.permute(0, 3, 1, 2).to(device, non_blocking=True)
                            elif len(samples.shape) == 5:
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
                        #     outputs = outputs.unsqueeze(-1).unsqueeze(-1)
                        #     # outputs, target_test, mask = remove_virtual_nodes(outputs, target_test, batch.ptr)
                        
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
                    test_log = {"test/test_RMSE": _err_RMSE_avg, "test/test_nRMSE": _err_nRMSE_avg, "test/fRMSE":_err_F_avg, 'test/MAX-ERR':_err_max_avg, 'test/CSV':_err_csv_avg, 'test/BD':_err_BD_avg}
                    accelerator.log(test_log, step=global_step)

            logs = {
                "train/lr": current_lr,
                "train/grad_norm": accelerator.gather(grad_norm).mean().detach().item(),
                "train/loss": accelerator.gather(loss).mean().detach().item(),    
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        
        scheduler.step()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--remark", type=str, default=' ', help="Training remark")
    default_args = parser.parse_args()
    
    args = get_config(config_path=default_args.config_file)
    args = Namespace(**vars(default_args), **vars(args))
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ## ComplexData <-> DDP
    if args.spa_mod == 'fourier' or args.spa_mod == 'frequency':
        args.mixed_precision = "no"
    main(args) 
    
