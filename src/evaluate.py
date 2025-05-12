import argparse
from argparse import Namespace
import datetime
import math
import numpy as np
import torch
import os
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

def align_and_load_state_dict(model, state_dict, strict=False, verbose=True):
    def strip_prefix_if_present(state_dict, prefix):
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}

    def add_prefix(state_dict, prefix):
        return {f"{prefix}{k}": v for k, v in state_dict.items()}

    # Step 1: Remove profiling keys like 'total_ops' and 'total_params'
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not any(x in k for x in ['total_ops', 'total_params'])
    }

    # Step 2: Handle 'module.' prefix alignment
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(filtered_state_dict.keys())

    model_has_module = any(k.startswith('module.') for k in model_keys)
    ckpt_has_module = any(k.startswith('module.') for k in ckpt_keys)

    if model_has_module and not ckpt_has_module:
        if verbose:
            print("Model expects 'module.' prefix but checkpoint does not have it. Adding prefix...")
        filtered_state_dict = add_prefix(filtered_state_dict, 'module.')
    elif not model_has_module and ckpt_has_module:
        if verbose:
            print("Checkpoint has 'module.' prefix but model does not expect it. Removing prefix...")
        filtered_state_dict = strip_prefix_if_present(filtered_state_dict, 'module.')
    else:
        if verbose:
            print("No prefix adjustment needed.")

    # Step 3: Load the state_dict
    model.load_state_dict(filtered_state_dict, strict=strict)

# def align_and_load_state_dict(model, state_dict):
#     model_keys = list(model.state_dict().keys())
#     ckpt_keys = list(state_dict.keys())

#     model_has_module = any(k.startswith('module.') for k in model_keys)
#     ckpt_has_module = any(k.startswith('module.') for k in ckpt_keys)

#     if model_has_module and not ckpt_has_module:

#         print("Model expects 'module.' prefix but checkpoint does not have it. Adding prefix...")
#         state_dict = {f'module.{k}': v for k, v in state_dict.items()}
#     elif not model_has_module and ckpt_has_module:
#         print("Checkpoint has 'module.' prefix but model does not expect it. Removing prefix...")
#         state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
#     else:
#         print("No prefix adjustment needed.")

#     model.load_state_dict(state_dict)

def main(args):
    
    logger = get_logger(__name__)
    logging_dir = Path(args.output_dir, args.tensorboard_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=args.mixed_precision,
        log_with='tensorboard',
        project_config=accelerator_project_config,
    )

    device = accelerator.device

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
    
    model = model.to(device)
    #<<<<<< =================================================================

    #>>>>>> =============================Data Reading==========================
    data_module_name = 'fdbench.data.' + args.PDE_type + '_data_utils'
    data_module = getattr(importlib.import_module(data_module_name), 'DatasetSingle')
    train_data = data_module(args = args)
    normalizer = train_data.__normalizer__
    test_data = data_module(if_test=True,args = args,normalizer=normalizer)
    # test_data = data_module(if_testid=True,args = args,normalizer=normalizer)

    if not args.spa_mod == 'graph':
        data_loader_test = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                            num_workers=args.num_workers)
    else:
        sample_nodes = 1024
        rand_idx = torch.randperm(args.input_size ** 2)[:sample_nodes]  # Random select N nodes
        from fdbench.data.graph_data import get_graph_dataloader
        data_loader_train, normalizer_new = get_graph_dataloader(train_data, rand_idx, batch_size=args.batch_size, normalizer=normalizer, normalizer_new=None, is_train=True, k=args.neighbor)
        data_loader_test, _ = get_graph_dataloader(test_data, rand_idx, batch_size=args.batch_size, normalizer=normalizer, normalizer_new=normalizer_new, is_train=False, k=args.neighbor)
    #<<<<<< =================================================================
    model, data_loader_test = accelerator.prepare(model, data_loader_test)

    ckpt_name = str(args.resume_step).zfill(7) + '.pt'
    ckpt_path = os.path.join(args.output_dir, args.exp_name, 'checkpoints', ckpt_name)
    ckpt = torch.load(ckpt_path, map_location='cpu')

    align_and_load_state_dict(model, ckpt['model'])
    global_step = ckpt['steps']
    
    criterion = torch.nn.MSELoss()
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.exp_name,  
        )
    #### =========4. Model Testing=========

    model.eval()  # important! This disables randomized embedding dropout
    
    _err_RMSE_avg = 0
    _err_nRMSE_avg = 0
    _err_max_avg = 0
    _err_csv_avg = 0
    _err_BD_avg = 0
    _err_F_avg = 0
    with torch.no_grad():
        
        for bat_idx,batch in enumerate(data_loader_test):
            if hasattr(batch, 'x') and hasattr(batch, 'y'):
                data = batch.to(device)
                input_test = data
                target_test = data.y
                grid = getattr(data, 'grid', None)
            else:
                input_test, target_test, grid = batch
                input_test = input_test.to(device)
                target_test = target_test.to(device)
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
            grid = grid.to(device)

            if hasattr(args, "if_rollout") and args.if_rollout:
               if args.tem_mod in {'next_step'}:
                   outputs = []
                   if args.roll_steps == -1:
                       roll_step = input_test.shape[-2]
                   else:
                       roll_step = args.roll_step
                   input_test_t = input_test[...,0,:].permute(0, 3, 1, 2)
                   for roll_t in range(roll_step-1):
                       target_test_t = input_test[...,roll_t+1,:].permute(0, 3, 1, 2)
                       outputs_t, loss = model(input_test_t,target_test_t,grid,criterion) 
                       input_test_t = outputs_t
                       outputs.append(outputs_t)

                   target_test = input_test[...,1:,:]
                   outputs = torch.cat([x.unsqueeze(1) for x in outputs], dim=1).permute(0,3,4,1,2)  
            else:

                if args.spa_mod == "diffusion" or args.spa_mod == "graph_diffusion":
                    if args.sample_method == "ddpm":
                        sample_fn = model.ddpm_sample
                    else:
                        sample_fn = model.ddim_sample
                    outputs, loss = sample_fn(input_test,target_test,grid,criterion)
                elif args.tem_mod in {'next_step'}:
                    if getattr(args, 'if_coordinate', False):
                        grid_test = grid_test.permute(0,3,1,2)
                        input_test = torch.concat([input_test,grid_test],dim=1)
                    outputs, loss = model(input_test,target_test,grid,criterion)
                elif args.tem_mod in {'self_atten','temporal_bundling','node'}:
                    input_test = input_test.permute(0, 3, 4, 1, 2)
                    target_test = target_test.permute(0, 3, 4, 1, 2)
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
            
            if bat_idx == 0:
                from matplotlib.backends.backend_pdf import PdfPages
                from einops import rearrange 
                import matplotlib.pyplot as plt
                vis_pdf_path = os.path.join('/wanghaixin/FD-Bench/vis', args.exp_name + '.pdf')
                pdf = PdfPages(vis_pdf_path)
                
                fontdict = {
                    'fontsize': 16,
                    'fontweight': 'bold',  #  'normal', 'bold', 'light'
                    'family': 'serif',     # 'sans-serif', 'monospace', etc.
                }

                if len(outputs.shape) == 4:
                    input_test = rearrange(outputs, "B C H W -> B H W C").detach().cpu().unsqueeze(-2)
                    targets = rearrange(target_test, "B C H W -> B H W C").detach().cpu().unsqueeze(-2)
                elif len(outputs.shape) == 5:
                    input_test = rearrange(outputs, "B T C H W -> B H W T C").detach().cpu()
                    targets = rearrange(target_test, "B T C H W -> B H W T C").detach().cpu()

                for i in range(min(input_test.size(0), 4)):
                    T = input_test.size(-2)
                    C = input_test.size(-1)
                    fig, axes = plt.subplots(2 * T, C, figsize=(16, 9))

                    # 安全处理 axes
                    if isinstance(axes, plt.Axes):
                        axes = np.array([axes])
                    else:
                        axes = np.array(axes).flatten()

                    for j in range(C):
                        for k in range(T):
                            idx = k * C + j

                            # 输出预测
                            axes[idx].imshow(input_test[i, :, :, k, j].numpy(), cmap='coolwarm')
                            axes[idx].axis('off')
                            axes[idx].set_title(f'Sample {i+1}, Step {k+1}, Ch {j+1}',fontdict=fontdict)

                            # Ground Truth
                            axes[idx + (len(axes) // 2)].imshow(targets[i, :, :, k, j].numpy(), cmap='coolwarm')
                            axes[idx + (len(axes) // 2)].axis('off')
                            axes[idx + (len(axes) // 2)].set_title(f'GT {i+1}, Step {k+1}, Ch {j+1}',fontdict=fontdict)

                    plt.tight_layout(pad=0.5, w_pad=2, h_pad=2)
                    pdf.savefig(fig, dpi=300)
                    plt.close(fig)

                pdf.close()

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

        
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(test_log)
    accelerator.end_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing Configuration")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--remark", type=str, default=' ', help="Training remark")
    parser.add_argument("--exp_name", type=str, default=' ', help="Training remark")
    parser.add_argument("--resume_step", type=int, default=10000, help="Training remark")
    parser.add_argument("--roll_step", type=int, default=-1, help="Training remark")
    parser.add_argument("--if_rollout", type=bool, default=False, help="Training remark")
    default_args = parser.parse_args()
    
    args = get_config(config_path=default_args.config_file)
    args = Namespace(**vars(default_args), **vars(args))
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ## ComplexData <-> DDP
    if args.spa_mod == 'fourier' or args.spa_mod == 'frequency':
        args.mixed_precision = "no"
    main(args) 
    
