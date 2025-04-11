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

def align_and_load_state_dict(model, state_dict):
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())

    model_has_module = any(k.startswith('module.') for k in model_keys)
    ckpt_has_module = any(k.startswith('module.') for k in ckpt_keys)


    if model_has_module and not ckpt_has_module:

        print("Model expects 'module.' prefix but checkpoint does not have it. Adding prefix...")
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    elif not model_has_module and ckpt_has_module:
        print("Checkpoint has 'module.' prefix but model does not expect it. Removing prefix...")
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    else:
        print("No prefix adjustment needed.")

    model.load_state_dict(state_dict)

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
        for bat_idx, batch in enumerate(data_loader_test):
            if hasattr(batch, 'x') and hasattr(batch, 'y'):
                data = batch.to(device)
                input_test = data
                target_test = data.y
                grid = getattr(data, 'grid', None)
            else:
                input_test, target_test, grid = batch
                input_test = input_test.permute(0, 3, 1, 2).to(device, non_blocking=True)
                target_test = target_test.permute(0, 3, 1, 2).to(device, non_blocking=True)
                grid = grid.to(device) if grid is not None else None

            if args.spa_mod == "diffusion" or args.spa_mod == "graph_diffusion":
                if args.sample_method == "ddpm":
                    samp_algo = model.ddpm_sample
                else:
                    samp_algo = model.ddim_sample
                outputs, _ = samp_algo(input_test,target_test,grid,criterion)
            else:
                outputs, _ = model(input_test,target_test,grid,criterion)
            
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
                    samples = rearrange(outputs, "B C H W -> B H W C").detach().cpu().unsqueeze(-2)
                    targets = rearrange(target_test, "B C H W -> B H W C").detach().cpu().unsqueeze(-2)
                elif len(outputs.shape) == 5:
                    samples = rearrange(outputs, "B T C H W -> B H W T C").detach().cpu()
                    targets = rearrange(target_test, "B T C H W -> B H W T C").detach().cpu()

                for i in range(min(samples.size(0), 4)):
                    T = samples.size(-2)
                    C = samples.size(-1)
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
                            axes[idx].imshow(samples[i, :, :, k, j].numpy(), cmap='coolwarm')
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
    default_args = parser.parse_args()
    
    args = get_config(config_path=default_args.config_file)
    args = Namespace(**vars(default_args), **vars(args))
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ## ComplexData <-> DDP
    if args.spa_mod == 'fourier' or args.spa_mod == 'frequency':
        args.mixed_precision = "no"
    main(args) 
    
