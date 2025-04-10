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
from scipy.interpolate import griddata

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
def evaluate(data_loader, model, device, use_amp, args, grid_h=None, grid_w=None):
    criterion = torch.nn.MSELoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    if args.use_odeint or args.graph_baseline:  # For graph models
        for data in metric_logger.log_every(data_loader, 10, header):
            # Use the PyG data object directly
            data = data.to(device)
            if use_amp:  # Use mixed precision
                with torch.cuda.amp.autocast():
                    # model expects [B*N, T, C]
                    outputs, loss = model(data, data.y, criterion) # outputs shape: [B, N, T_pred, 1]
            else:  # Use full precision
                outputs, loss = model(data, data.y, criterion) # outputs shape: [B, N, T_pred, 1]
            
            batch_size = data.num_graphs
            n_nodes_per_graph = data.num_nodes // batch_size # N
            forecast_horizon = outputs.shape[2] # T_pred
            in_chans = data.x.shape[2] # C_in
            target_chans = data.y.shape[2] # C_out

            metric_logger.update(loss=loss.item())

            # --- Interpolate graph data to grid for metrics ---
            if grid_h is None or grid_w is None:
                if not hasattr(evaluate, '_printed_grid_fallback'):
                    print(f"Warning [evaluate]: Grid dimensions not provided. Falling back to 128x128.")
                    evaluate._printed_grid_fallback = True 
                interp_h, interp_w = 128, 128
            else:
                interp_h, interp_w = grid_h, grid_w
            
            # Create grid coordinates for interpolation
            grid_x_vals = np.linspace(0, 1, interp_w) # Assuming coordinates are normalized [0,1]
            grid_y_vals = np.linspace(0, 1, interp_h) # Assuming coordinates are normalized [0,1]
            grid_xx, grid_yy = np.meshgrid(grid_x_vals, grid_y_vals)
            target_grid_coords = np.vstack([grid_xx.ravel(), grid_yy.ravel()]).T

            # Reshape predictions and targets
            # Prediction: [B, N, T_pred, 1] -> [B, N, T_pred]
            pred_reshaped = outputs.squeeze(-1)
            # Target: [B*N, T_target, C_out] -> [B, N, T_target, C_out]
            target_reshaped = data.y.reshape(batch_size, n_nodes_per_graph, -1, target_chans)

            # Determine common timesteps
            num_target_timesteps = target_reshaped.shape[2] # T_target
            num_pred_timesteps = forecast_horizon
            num_timesteps = min(num_pred_timesteps, num_target_timesteps)

            # Prepare lists to store interpolated tensors
            interpolated_preds = []
            interpolated_targets = []

            # Get node positions: [B*N, 2] -> [B, N, 2]
            node_positions = data.pos.reshape(batch_size, n_nodes_per_graph, 2).cpu().numpy()

            # Iterate through batch, interpolate each sample
            for b in range(batch_size):
                # Current sample data (move to CPU)
                current_node_pos = node_positions[b] # [N, 2]
                current_pred = pred_reshaped[b, :, :num_timesteps].cpu().numpy() # [N, num_timesteps]
                current_target = target_reshaped[b, :, :num_timesteps, :].cpu().numpy() # [N, num_timesteps, C_out]

                # Interpolate predictions (single channel)
                interpolated_pred_sample = griddata(
                    current_node_pos,
                    current_pred.reshape(n_nodes_per_graph, -1), # Flatten time steps
                    target_grid_coords,
                    method='linear',
                    fill_value=0 # Use 0 for points outside convex hull
                ).reshape(interp_h, interp_w, num_timesteps)
                interpolated_preds.append(torch.from_numpy(interpolated_pred_sample)) # [H, W, T]

                # Interpolate targets (multiple channels)
                interpolated_target_sample_channels = []
                for c in range(target_chans):
                    interpolated_target_chan = griddata(
                        current_node_pos,
                        current_target[:, :, c].reshape(n_nodes_per_graph, -1), # Flatten time steps
                        target_grid_coords,
                        method='linear',
                        fill_value=0 # Use 0 for points outside convex hull
                    ).reshape(interp_h, interp_w, num_timesteps)
                    interpolated_target_sample_channels.append(torch.from_numpy(interpolated_target_chan)) # [H, W, T]
                # Stack channels: List[[H, W, T]] -> [H, W, T, C]
                interpolated_target_sample = torch.stack(interpolated_target_sample_channels, dim=-1)
                interpolated_targets.append(interpolated_target_sample)

            # Stack batch: List[[H, W, T]] -> [B, H, W, T]
            pred_grid = torch.stack(interpolated_preds, dim=0).to(device)
            # Stack batch: List[[H, W, T, C]] -> [B, H, W, T, C]
            target_grid = torch.stack(interpolated_targets, dim=0).to(device)

            # --- Prepare tensors for metric_func (expects [B, C, H, W, T] or [B, C, H, W]) ---
            # Select the target variable for comparison
            target_var_idx = args.pred_var if args.pred_var >= 0 else target_grid.shape[-1] + args.pred_var
            target_grid_var = target_grid[:, :, :, :, target_var_idx] # [B, H, W, T]

            # Format tensor for metric_func: requires [B, H, W, C, T]
            pred_grid_metric = pred_grid.unsqueeze(3) # Shape [B, H, W, 1, T]
            target_grid_metric = target_grid_var.unsqueeze(3) # Shape [B, H, W, 1, T]
            
            # pred_grid_metric = pred_grid_metric.permute(0, 1, 4, 2, 3) # [B, 1, T, H, W]
            # target_grid_metric = target_grid_metric.permute(0, 1, 4, 2, 3) # [B, 1, T, H, W]

            # Check shapes before calling metric_func
            # print(f"Shape before metrics: Pred={pred_grid_metric.shape}, Target={target_grid_metric.shape}")
            
            Lx, Ly, Lz = 1., 1., 1.
            _err_RMSE, _err_nRMSE, _err_CSV, _err_Max, _err_BD, _err_F \
                = metrics.metric_func(pred_grid_metric, target_grid_metric, if_mean=True, Lx=Lx, Ly=Ly, Lz=Lz)

            metric_logger.update(rmse=_err_RMSE.item())
            metric_logger.update(nrmse=_err_nRMSE.item())
            # Add FRMSE to metric logger
            metric_logger.update(frmse=_err_F[0].item() if not torch.isnan(_err_F[0]) else 0.0)

            # --- Old code for reference ---
            # batch_size = data.num_graphs
            # n_nodes = data.num_nodes // batch_size
            # forecast_horizon = outputs.shape[2] 
            # metric_logger.update(loss=loss.item())
            # 
            # # --- Target Processing for Metrics --- 
            # # data.y shape is [B*N, T_target, C]
            # target_var_idx = args.pred_var if args.pred_var >= 0 else data.y.shape[2] + args.pred_var
            # target_all_steps = data.y[:, :, target_var_idx] # Shape: [B*N, T_target]

            # # Slice target timesteps to match prediction horizon
            # num_target_timesteps = target_all_steps.shape[1] # T_target dim
            # num_pred_timesteps = forecast_horizon
            # num_timesteps = min(num_pred_timesteps, num_target_timesteps) # Use common timesteps
            # target_sliced = target_all_steps[:, :num_timesteps] # Shape: [B*N, num_timesteps]

            # # Slice prediction timesteps if necessary
            # # Reshape prediction [B, N, T_pred, 1] -> [B*N, T_pred]
            # pred_reshaped = outputs.reshape(batch_size * n_nodes, forecast_horizon)
            # pred_sliced = pred_reshaped[:, :num_timesteps] # Shape: [B*N, num_timesteps]

            # # --- Reshape for metrics calculation --- 
            # # NOTE: metrics.metric_func expects [B, C, H, W] or [B, C, D, H, W]
            # # The current graph output [B*N, T] or [B, N, T, 1] is incompatible with spatial FFT used in FRMSE.
            # # will reshape to [B*N, 1, T, 1] to calculate RMSE/NRMSE, but FRMSE will likely remain NaN.
            # outputs_reshaped_for_metrics = pred_sliced.unsqueeze(1).unsqueeze(-1) # Shape: [B*N, 1, num_timesteps, 1]
            # target_reshaped_for_metrics = target_sliced.unsqueeze(1).unsqueeze(-1) # Shape: [B*N, 1, num_timesteps, 1]
            # 
            # Lx, Ly, Lz = 1., 1., 1.
            # _err_RMSE, _err_nRMSE, _err_CSV, _err_Max, _err_BD, _err_F \
            # = metrics.metric_func(outputs_reshaped_for_metrics, target_reshaped_for_metrics, if_mean=True, Lx=Lx, Ly=Ly, Lz=Lz)

            # metric_logger.update(rmse=_err_RMSE.item())
            # metric_logger.update(nrmse=_err_nRMSE.item())
            # # Add FRMSE to metric logger, but not currently working
            # metric_logger.update(frmse=_err_F[0].item() if not torch.isnan(_err_F[0]) else 0.0)
            
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

    # --- Get Grid Dimensions from the input dataset (for interpolation) --- 
    grid_h, grid_w = None, None
    if hasattr(dataset, 'grid') and isinstance(dataset.grid, torch.Tensor):
        grid_shape = dataset.grid.shape
        if len(grid_shape) >= 2:
            grid_h = grid_shape[0]
            grid_w = grid_shape[1]
            print(f"[get_graph_dataloader] Determined grid dimensions: H={grid_h}, W={grid_w}")
        else:
            print(f"[get_graph_dataloader] Warning: Input dataset.grid has unexpected shape {grid_shape}. Cannot determine H, W.")
    else:
        print(f"[get_graph_dataloader] Warning: Input dataset object of type {type(dataset).__name__} lacks a 'grid' attribute or it's not a Tensor.")

    if grid_h is None or grid_w is None:
        print(f"[get_graph_dataloader] Warning: Falling back to default grid dimensions 128x128.")
        grid_h, grid_w = 128, 128

    first_iter = True

    for i in range(len(dataset)):
        x, y, grid = dataset[i]

        if first_iter:
            print("x, y, grid shape (in get_graph_dataloader)")
            print(x.shape,y.shape,grid.shape,'++++++++'*10)
            first_iter = False

        # --- Reshape tensors to be node-centric ---
        # Assume input shape is [H, W, T, C]
        # x: [H, W, T_in, C_in] -> [N, T_in, C_in] where N = H*W
        H, W, T_in, C_in = x.shape 
        num_nodes = H * W
        node_x = x.reshape(num_nodes, T_in, C_in)

        # y: [H, W, T_out, C_out] -> [N, T_out, C_out]
        H, W, T_out, C_out = y.shape 
        node_y = y.reshape(num_nodes, T_out, C_out)

        # grid: [H, W, 2] -> [N, 2]
        node_pos = grid.reshape(num_nodes, 2)

        if i == 0:
            print(f"DEBUG: Coordinate range (sample 0):")
            print(f"  X min/max: {node_pos[:, 0].min().item():.4f} / {node_pos[:, 0].max().item():.4f}")
            print(f"  Y min/max: {node_pos[:, 1].min().item():.4f} / {node_pos[:, 1].max().item():.4f}")

        # --- Graph Construction (using node_pos) ---
        points = node_pos.numpy() # Use reshaped node_pos
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

        # --- Create Data object with node-centric tensors ---
        # Store coordinates in 'pos' attribute
        data_list.append(Data(x=node_x, y=node_y, pos=node_pos, edge_index=edge_index, edge_attr=edge_attr))

        # TODO: apply virtual node transform
        """
        data = Data(x=x, edge_index=edge_index)


        # Apply the VirtualNode transform
        transform = VirtualNode()
        data = transform(data)
        """
        

    dataloader = DataLoader(data_list, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader, grid_h, grid_w
