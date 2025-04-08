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
from fdbench.utils.utils import tprint
from fdbench.utils import metrics
from torch_geometric.data import Data, DataLoader
from scipy.spatial import distance_matrix
import numpy as np
import random

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.MSELoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    use_amp=True, set_training_mode=True, use_odeint=False, graph_baseline=False):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    #print("use_odeint (train_one_epoch): ", use_odeint)

    if use_odeint or graph_baseline:
        for d in metric_logger.log_every(data_loader, print_freq, header):
            samples = d.x
            targets = d.y

            #print("samples, targets")
            #print(samples.shape,targets.shape,'++++++++'*10)

            
            #if mixup_fn is not None:
            #    samples, targets = mixup_fn(samples, targets)

            # compute output
            if use_amp:  # 使用混合精度训练
                with torch.cuda.amp.autocast():
                    outputs, loss = model(d,targets,criterion)
                    
            else:  # 使用全精度训练
                outputs, loss = model(d,targets,criterion)


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
            else:
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                loss.backward(create_graph=is_second_order)  # Compute gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # Clip gradients
                optimizer.step()  # Update parameters
            
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

    else:
        for samples, targets, grid in metric_logger.log_every(data_loader, print_freq, header):
            #* data shape [batch_size, 128, 128, 4]
            print("samples, targets, grid shape (in train_one_epoch)")
            print(samples.shape,targets.shape,grid.shape,'++++++++'*10)

            samples = samples.permute(0, 3, 1, 2).to(device, non_blocking=True)
            targets = targets.permute(0, 3, 1, 2).to(device, non_blocking=True)

            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)

            # compute output
            if use_amp:  # 使用混合精度训练
                with torch.cuda.amp.autocast():
                    outputs, loss = model(samples,targets,grid,criterion)
                    
            else:  # 使用全精度训练
                outputs, loss = model(samples,targets,grid,criterion)

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
            else:
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                loss.backward(create_graph=is_second_order)  # Compute gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # Clip gradients
                optimizer.step()  # Update parameters
            
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

    if args.use_odeint or args.graph_baseline:  # For graph models
        for data in metric_logger.log_every(data_loader, 10, header):
            # Use the PyG data object directly
            if use_amp:  # Use mixed precision
                with torch.cuda.amp.autocast():
                    outputs, loss = model(data, data.y, criterion)
            else:  # Use full precision
                outputs, loss = model(data, data.y, criterion)
            
            batch_size = data.x.shape[0]
            metric_logger.update(loss=loss.item())
            
            # --- Target Processing for Metrics ---
            target_var_idx = args.pred_var if args.pred_var >= 0 else data.y.shape[3] + args.pred_var
            target_all_steps = data.y[:, :, :, target_var_idx] # Shape: [batch, nodes, time]
            
            # Slice target timesteps to match prediction horizon
            num_target_timesteps = target_all_steps.shape[2]
            num_pred_timesteps = outputs.shape[2]
            num_timesteps = min(num_pred_timesteps, num_target_timesteps)
            target_sliced = target_all_steps[:, :, :num_timesteps] # Shape: [batch, nodes, num_timesteps]

            # Slice prediction timesteps if necessary
            pred_sliced = outputs[:, :, :num_timesteps, :] # Shape: [batch, nodes, num_timesteps, 1]

            # --- Reshape for metrics calculation ---
            outputs_reshaped = pred_sliced.unsqueeze(1)  # Shape: [batch, 1, nodes, num_timesteps, 1]
            target_reshaped = target_sliced.unsqueeze(1).unsqueeze(-1)  # Shape: [batch, 1, nodes, num_timesteps, 1]
            
            Lx, Ly, Lz = 1., 1., 1.
            _err_RMSE, _err_nRMSE, _err_CSV, _err_Max, _err_BD, _err_F \
            = metrics.metric_func(outputs_reshaped, target_reshaped, if_mean=True, Lx=Lx, Ly=Ly, Lz=Lz)

            metric_logger.update(rmse=_err_RMSE.item())
            metric_logger.update(nrmse=_err_nRMSE.item())
            metric_logger.update(frmse=_err_F[0].item())
            
    else:  # For non-graph models
        for input_test, target_test, grid in metric_logger.log_every(data_loader, 10, header):
            input_test = input_test.permute(0, 3, 1, 2).to(device, non_blocking=True)
            target_test = target_test.permute(0, 3, 1, 2).to(device, non_blocking=True)

            if args.spa_mod == "diffusion" or args.spa_mod == "graph_diffusion":
                if args.sample_method == "ddpm":
                    samp_algo = model.ddpm_sample
                else:
                    samp_algo = model.ddim_sample
                
                if use_amp:  # 使用混合精度训练
                    with torch.cuda.amp.autocast():
                        outputs, loss = samp_algo(input_test,target_test,grid,criterion)
                else:  # 使用全精度训练
                    outputs, loss = samp_algo(input_test,target_test,grid,criterion)
            else:
                # compute output
                if use_amp:  # 使用混合精度训练
                    with torch.cuda.amp.autocast():
                        outputs, loss = model(input_test,target_test,grid,criterion)
                else:  # 使用全精度训练
                    outputs, loss = model(input_test,target_test,grid,criterion)

            batch_size = input_test.shape[0]
            metric_logger.update(loss=loss.item())
            Lx, Ly, Lz = 1., 1., 1.
            _err_RMSE, _err_nRMSE, _err_CSV, _err_Max, _err_BD, _err_F \
            = metrics.metric_func(outputs, target_test, if_mean=True, Lx=Lx, Ly=Ly, Lz=Lz)

            metric_logger.update(rmse=_err_RMSE.item())
            metric_logger.update(nrmse=_err_nRMSE.item())
            metric_logger.update(frmse=_err_F[0].item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    tprint('* MSE loss {losses.global_avg:.3f}, RMSE {rmse.global_avg:.3f}, \
        nRMSE {nrmse.global_avg:.3f}, fRMSE {frmse.global_avg:.3f}'.format( \
        losses=metric_logger.loss, rmse=metric_logger.rmse, \
        nrmse=metric_logger.nrmse, frmse=metric_logger.frmse))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def get_graph_dataloader(dataset, batch_size, k=20, num_workers=1, shuffle=True):
    data_list = []

    first_iter = True

    for i in range(len(dataset)):
        x, y, grid = dataset[i]

        if first_iter:
            print("x, y, grid shape (in get_graph_dataloader)")
            print(x.shape,y.shape,grid.shape,'++++++++'*10)
            first_iter = False

        # Reshape grid to [n_nodes, n_dims] e.g. [nx*ny, 2] for 2D
        num_spatial_dims = grid.shape[-1]
        points = grid.reshape(-1, num_spatial_dims).numpy()
        # calculate the distance matrix
        dist_matrix = distance_matrix(points, points)
        # compare with grid
        # eg coord of number 2 and number 3, check if dist_matrix[2, 3] equals distance shown in grid[2,3]

        # TODO: look into downsampling (see graph-pde RandomMultiMeshGenerator sample method)
        # make sure that we drop indices

        # find the nearest neighbors, keep start and end nodes
        start_nodes = []
        end_nodes = []
        for j in range(len(points)):
            nearest_indices = np.argsort(dist_matrix[j])[1:k+1]
            for index in nearest_indices:
                start_nodes.append(j)
                end_nodes.append(index)
        start_nodes_tensor = torch.tensor(start_nodes, dtype=torch.long)
        end_nodes_tensor = torch.tensor(end_nodes, dtype=torch.long)
        edge_index = torch.stack([start_nodes_tensor, end_nodes_tensor], dim=0)

        senders = edge_index[0].numpy()
        receivers = edge_index[1].numpy()
        crds_diff = points[senders] - points[receivers]
        crds_norm = np.linalg.norm(crds_diff, axis=1, keepdims=True)
        edge_attr = np.concatenate((crds_diff, crds_norm), axis=1)
        edge_attr = torch.from_numpy(edge_attr)

        data_list.append(Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, grid=grid))

        # TODO: apply virtual node transform
        """
        data = Data(x=x, edge_index=edge_index)


        # Apply the VirtualNode transform
        transform = VirtualNode()
        data = transform(data)
        """
        

    dataloader = DataLoader(data_list, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
