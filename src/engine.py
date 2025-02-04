# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import ModelEma
from fdbench.utils import utils
import random
from datetime import datetime

def tprint(*args, **kwargs):
    """print with time"""
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{time_str}]', *args, **kwargs)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.MSELoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    use_amp=True, set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets, grid in metric_logger.log_every(data_loader, print_freq, header):
        #* data shape [batch_size, 128, 128, 4]
        samples = samples.permute(0, 3, 1, 2).to(device, non_blocking=True)
        targets = targets.permute(0, 3, 1, 2).to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # compute output
        if use_amp:  # 使用混合精度训练
            with torch.cuda.amp.autocast():
                outputs, loss = model(samples,targets,criterion)
                
        else:  # 使用全精度训练
            outputs, loss = model(samples,targets,criterion)
            

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            tprint("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        optimizer.zero_grad()

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                            parameters=model.parameters(), create_graph=is_second_order)
        
        # Calculate gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5  # L2 norm
        metric_logger.update(grad_norm=total_norm)
    
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    tprint("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, use_amp, args):
    criterion = torch.nn.MSELoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target, grid in metric_logger.log_every(data_loader, 10, header):
        images = images.permute(0, 3, 1, 2).to(device, non_blocking=True)
        target = target.permute(0, 3, 1, 2).to(device, non_blocking=True)

        if args.spa_mod == "diffusion":
            if args.sample_method == "ddpm":
                samp_algo = model.ddpm_sample
            else:
                samp_algo = model.ddim_sample
            
            if use_amp:  # 使用混合精度训练
                with torch.cuda.amp.autocast():
                    outputs, loss = samp_algo(images,target,criterion)
            else:  # 使用全精度训练
                outputs, loss = samp_algo(images,target,criterion)

        else:
            # compute output
            if use_amp:  # 使用混合精度训练
                with torch.cuda.amp.autocast():
                    outputs, loss = model(images,target,criterion)
            else:  # 使用全精度训练
                outputs, loss = model(images,target,criterion)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    tprint('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
