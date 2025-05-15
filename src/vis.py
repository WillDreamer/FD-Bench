import argparse
from argparse import Namespace
import numpy as np
import torch
import os
import json
from pathlib import Path
import importlib

from fdbench.utils.utils import *
import warnings
warnings.filterwarnings('ignore')


def main(args):

    #>>>>>> =============================Data Reading==========================
    data_module_name = 'fdbench.data.' + args.PDE_type + '_data_utils'
    data_module = getattr(importlib.import_module(data_module_name), 'DatasetSingle')
    train_data = data_module(args = args)
    normalizer = train_data.__normalizer__
    test_data = data_module(if_test=True,args = args,normalizer=normalizer)

    data_loader_test = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                        num_workers=args.num_workers)
   
    with torch.no_grad():
        for bat_idx, batch in enumerate(data_loader_test):
           
            input_test, target_test, grid = batch
            input_test = input_test.permute(0, 3, 1, 2)
            target_test = target_test.permute(0, 3, 1, 2)
            
            if bat_idx == 0:
                from matplotlib.backends.backend_pdf import PdfPages
                from einops import rearrange 
                import matplotlib.pyplot as plt
                vis_pdf_path = os.path.join('/home/mosaicml/FD-Bench/vis', args.exp_name + '.pdf')
                pdf = PdfPages(vis_pdf_path)
                
                fontdict = {
                    'fontsize': 16,
                    'fontweight': 'bold',  #  'normal', 'bold', 'light'
                    'family': 'serif',     # 'sans-serif', 'monospace', etc.
                }
                targets = rearrange(target_test, "B C H W -> B H W C").detach().cpu().unsqueeze(-2)
               

                for i in range(min(targets.size(0), 4)):
                    T = targets.size(-2)
                    C = targets.size(-1)
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
                            axes[idx].imshow(targets[i, :, :, k, j].numpy(), cmap='coolwarm')
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

           


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing Configuration")
    parser.add_argument("--config_file", type=str, default='/home/mosaicml/FD-Bench/config/vis/vis.yaml', help="Path to the configuration file")
    parser.add_argument("--remark", type=str, default=' ', help="Training remark")
    parser.add_argument("--exp_name", type=str, default='vis_CNS', help="Training remark")
    default_args = parser.parse_args()
    
    args = get_config(config_path=default_args.config_file)
    args = Namespace(**vars(default_args), **vars(args))
    main(args) 
    
