import argparse
import datetime
from re import I
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import os
import json
from pathlib import Path
from timm.data import Mixup
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from functools import partial
import torch.nn as nn
import importlib
import random
from fdbench.utils import utils
from fdbench.utils.metrics import *
from engine import train_one_epoch, evaluate
import warnings
warnings.filterwarnings('ignore')

def tprint(*args, **kwargs):
    """print with time"""
    time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{time_str}]', *args, **kwargs)

def main(args):

    utils.init_distributed_mode(args)
    tprint(args)
    # Tensorboard Initialization
    if utils.is_main_process():
        writer = SummaryWriter(
            log_dir=args.tensorboard_dir+args.spa_mod+'_'+args.tem_mod,)

    # AutoResume
    import sys
    import warnings
    warnings.filterwarnings("ignore", message="Argument interpolation should be")
    sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
    AutoResume = None
    try:
        from userlib.auto_resume import AutoResume
    except ImportError:
        tprint(AutoResume)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #>>>>>> ===============================Model Design==================================
    module_name = 'fdbench.models.' + args.spa_mod
    class_name = args.spa_mod
    module = getattr(importlib.import_module(module_name),class_name)
    model = module(args=args)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) 
    n_parameters = count_parameters(model)/1e6
    tprint("Number of Parameters:", n_parameters, "Mb")
    #<<<<<< =================================================================

    #>>>>>> =============================Data Reading==========================
    data_module_name = 'fdbench.data.' + args.PDE_type + '_data_utils'
    data_module = getattr(importlib.import_module(data_module_name),'DatasetSingle')
    train_data = data_module(args = args)
    normalizer = train_data.__normalizer__
    test_data = data_module(if_test=True,args = args,normalizer=normalizer)
    val_data = data_module(if_valid=True,args = args,normalizer=normalizer)
    data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                               num_workers=args.num_workers)
    data_loader_test = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size//2,
                                             num_workers=args.num_workers)
    data_loader_val = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size//2,
                                             num_workers=args.num_workers)

        # if args.distributed:  
    #     num_tasks = utils.get_world_size()
    #     global_rank = utils.get_rank()
    #     if args.repeated_aug:
    #         sampler_train = RASampler(
    #             dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    #         )
    #     else:
    #         sampler_train = torch.utils.data.DistributedSampler(
    #             dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    #         )
    #     if args.dist_eval:
    #         if len(dataset_val) % num_tasks != 0:
    #             print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
    #                   'This will slightly alter validation results as extra duplicate entries are added to achieve '
    #                   'equal num of samples per-process.')
    #         sampler_val = torch.utils.data.DistributedSampler(
    #             dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    #     else:
    #         sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    # else:
    #     sampler_train = torch.utils.data.RandomSampler(dataset_train)
    #     sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train, sampler=sampler_train,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    # )

    # data_loader_val = torch.utils.data.DataLoader(
    #     dataset_val, sampler=sampler_val,
    #     batch_size=int(1.5 * args.batch_size),
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=False
    # )
    #<<<<<< =================================================================

    mixup_fn = None
    # mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    mixup_active = False
    if mixup_active:
        tprint('standard mix up')
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    else:
        tprint('mix up is not used')

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = torch.nn.MSELoss()

    output_dir = Path(args.output_dir)
    if args.resume and os.path.exists(args.resume):
        tprint("Resuming from checkpoint.")
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.autoresume:
        AutoResume.init()

    if utils.is_main_process():
        tprint(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            use_amp = args.use_amp,
            set_training_mode=args.finetune == ''  # keep in eval mode during finetuning
        )

        lr_scheduler.step(epoch)

        if epoch > (args.epochs//2) and (epoch+1)%50==0 and args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint_{}_ep{}.pth'.format(args.spa_mod,epoch)]
            for checkpoint_path in checkpoint_paths:
                if model_ema is not None:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)
                else:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)

        if (epoch +1) % args.eval_step == 0: 
            val_stats = evaluate(data_loader_val, model, device, args.use_amp, args)
            test_stats = evaluate(data_loader_test, model, device, args.use_amp, args)


            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        **{f'val_{k}': v for k, v in val_stats.items()},
                        'epoch': epoch}

            if utils.is_main_process():
                writer.add_scalar("train_lr", log_stats["train_lr"], log_stats["epoch"])
                writer.add_scalar("train_loss", log_stats["train_loss"], log_stats["epoch"])
                writer.add_scalar("test_loss", log_stats["test_loss"], log_stats["epoch"])
                writer.add_scalar("test_rmse", log_stats["test_rmse"], log_stats["epoch"])
                writer.add_scalar("test_nrmse", log_stats["test_nrmse"], log_stats["epoch"])
                writer.add_scalar("test_frmse", log_stats["test_frmse"], log_stats["epoch"])
                writer.add_scalar("val_rmse", log_stats["val_rmse"], log_stats["epoch"])
                writer.add_scalar("val_loss", log_stats["val_loss"], log_stats["epoch"])
                writer.add_scalar("val_nrmse", log_stats["val_nrmse"], log_stats["epoch"])
                writer.add_scalar("val_frmse", log_stats["val_frmse"], log_stats["epoch"])

            if args.output_dir and utils.is_main_process():
                log_file_path = os.path.join(args.tensorboard_dir, f"{args.spa_mod}_{args.tem_mod}", "log.txt")
                os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                with open(log_file_path, 'a') as f:  
                    f.write(json.dumps(log_stats) + "\n")

        # # AutoResume
        # if args.autoresume and AutoResume.termination_requested():
        #     print("AutoResume Termination Requested. Exiting.")
        #     if utils.is_main_process():
        #         AutoResume.request_resume()            
        #         return 0
        #     else:
        #         return 0


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    tprint('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file")
    default_args = parser.parse_args()
    
    args = utils.get_config(config_path=default_args.config_file)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args) 
    
