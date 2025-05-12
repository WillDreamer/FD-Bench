import math
import logging
from functools import partial
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from torch.utils.checkpoint import checkpoint_sequential
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from fdbench.models.self_atten.afno2d import AFNO2D
from fdbench.models.self_atten.bfno2d import BFNO2D
from fdbench.models.self_atten.ls import AttentionLS
from fdbench.models.self_atten.sa import SelfAttention
from fdbench.models.self_atten.gfn import GlobalFilter


_logger = logging.getLogger(__name__)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Swish(nn.Module):
    """
    Swish activation function
    """
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8, use_fno=False, use_blocks=False, args={}):
        super().__init__()
        self.norm1 = norm_layer(dim)

        #to be added soon ... @John: pls double check
        if args.mixing_type == "afno":
            self.filter = AFNO2D(hidden_size=args.hidden_size, num_blocks=args.fno_blocks, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1)
        elif args.mixing_type == "bfno":
            self.filter = BFNO2D(hidden_size=args.hidden_size, num_blocks=args.num_attention_heads, hard_thresholding_fraction=1)
        elif args.mixing_type == "sa":
            self.filter = SelfAttention(dim=args.hidden_size, heads=args.num_attention_heads)
        if args.mixing_type == "gfn":
            self.filter = GlobalFilter(dim=args.hidden_size, h=14, w=8)
        elif args.mixing_type == "ls":
            self.filter = AttentionLS(dim=args.hidden_size, num_heads=args.num_attention_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., rpe=False, nglo=1, dp_rank=2, w=2)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.double_skip = args.double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class UnpatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=4, embed_dim=256, out_chans=4):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.embed_dim = embed_dim
        self.out_chans = out_chans
        
        self.H = self.img_size[0] // self.patch_size[0]  # 32 for 128/4
        self.W = self.img_size[1] // self.patch_size[1]  # 32 for 128/4
        
        self.proj = nn.ConvTranspose2d(
            embed_dim,
            out_chans,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
    
        B, N, C = x.shape
        assert N == self.H * self.W, f"input patches number ({N}) does not match ({self.H * self.W})"
        
        x = x.reshape(B, self.H, self.W, C)
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)  # [B, out_chans, img_size, img_size]
        
        return x


class DownLayer(nn.Module):
    def __init__(self, img_size=56, dim_in=64, dim_out=128):
        super().__init__()
        self.img_size = img_size
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2)
        self.num_patches = img_size * img_size // 4

    def forward(self, x):
        B, N, C = x.size()
        x = x.view(B, self.img_size, self.img_size, C).permute(0, 3, 1, 2)
        x = self.proj(x).permute(0, 2, 3, 1)
        x = x.reshape(B, -1, self.dim_out)
        return x

class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x):
        
        x = self.linear(x)

        return x




### Main function for self-atten
class SADenoiser(nn.Module):
    def __init__(self, img_size=128, patch_size=4, in_chans=4,
                 embed_dim=256, depth=12,
                 mlp_ratio=4., representation_size=None, uniform_drop=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 dropcls=0, use_fno=False, use_blocks=False, args={}):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        self.config = args
        img_size = args.input_size
        patch_size = args.patch_size
        in_chans = args.in_chans * args.in_chans_ratio
        out_chans = args.in_chans
        embed_dim = args.hidden_size 
        depth = args.num_layers
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate * 0.5)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        if self.config.tem_mod in ['self_atten']:
            self.forecast_horizon = args.forecast_horizon
            patch_dim = in_chans * patch_size ** 2
            self.patch_embed = nn.Sequential(
                Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
                nn.Linear(patch_dim, embed_dim),
            )
            self.unpatch = Rearrange('b t (h w) (c p1 p2) -> b t c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=img_size//patch_size)
            grid_size = tuple([s // p for s, p in zip(to_2tuple(img_size), to_2tuple(patch_size))])
            num_patches = grid_size[0] * grid_size[1]
        
        elif self.config.tem_mod in ['temporal_bundling']:
            patch_dim = in_chans * patch_size ** 2
            self.patch_embed = nn.Sequential(
                Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
                nn.Linear(patch_dim, embed_dim),)
            self.unpatch = Rearrange('b t (h w) (c p1 p2) -> b t c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=img_size//patch_size)
            grid_size = tuple([s // p for s, p in zip(to_2tuple(img_size), to_2tuple(patch_size))])
            num_patches = grid_size[0] * grid_size[1]

            self.forecast_horizon = args.window_size
            self.time_window = args.window_size
            self.latent_dim=32
            
            conv_params = {
                5:  (23, 3, 5, 1)}
                
            if self.time_window in conv_params:
                in_ker_size, stride_bund, out_ker_size, out_stride = conv_params[self.time_window]
                self.output_mlp = nn.Sequential(
                    nn.Conv1d(1, 8, in_ker_size, stride=stride_bund),
                    Swish(),  
                    nn.Conv1d(8, 1, out_ker_size, stride=out_stride),
                )
            self.final_layer = FinalLayer(self.latent_dim, patch_size, out_chans)
        
        elif self.config.tem_mod in ['node']:
            self.forecast_horizon = args.forecast_horizon
            patch_dim = in_chans * patch_size ** 2
            self.patch_embed = nn.Sequential(
                Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
                nn.Linear(patch_dim, embed_dim),
            )
            self.unpatch = Rearrange('b t (h w) (c p1 p2) -> b t c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=img_size//patch_size)
            grid_size = tuple([s // p for s, p in zip(to_2tuple(img_size), to_2tuple(patch_size))])
            num_patches = grid_size[0] * grid_size[1]
            self.decoding_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              Swish(),
                                              nn.Linear(embed_dim, 1),
                                              Swish()
                                              )
            # ODEINT derivative network
            self.derivative_net = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                                Swish(),
                                                nn.Linear(embed_dim, self.embed_dim),
                                                Swish()
                                                )
            self.final_layer = FinalLayer(embed_dim, patch_size, out_chans)
        
        elif self.config.tem_mod in ['auto_regressive']:
            patch_dim = args.initial_step * in_chans * patch_size ** 2
            self.patch_embed = nn.Sequential(
                Rearrange('b (t c) (h p1) (w p2) -> b (h w) (t p1 p2 c)', p1 = patch_size, p2 = patch_size, t=args.initial_step),
                nn.Linear(patch_dim, embed_dim),
            )
            self.unpatch = UnpatchEmbed(
                img_size=img_size, patch_size=patch_size, out_chans=out_chans, embed_dim=embed_dim)
            grid_size = tuple([s // p for s, p in zip(to_2tuple(img_size), to_2tuple(patch_size))])
            num_patches = grid_size[0] * grid_size[1]
        
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            self.unpatch = UnpatchEmbed(
                img_size=img_size, patch_size=patch_size, out_chans=out_chans, embed_dim=embed_dim)
            num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        h = img_size // patch_size
        w = h // 2 + 1
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, mlp_ratio=mlp_ratio,
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                h=h, w=w, use_fno=use_fno, use_blocks=use_blocks,
                args = self.config)
            for i in range(depth)])
        
        if self.config.tem_mod in ['self_atten']:
            self.temporal_blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, mlp_ratio=mlp_ratio,
                    drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                    h=h, w=w, use_fno=use_fno, use_blocks=use_blocks,
                    args = self.config)
                for i in range(depth//2)])
            self.final_layer = FinalLayer(embed_dim, patch_size, out_chans)
        
        self.norm = norm_layer(embed_dim)
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        
        self.mixing_type = args.mixing_type

        self.sinu_pos_emb = SinusoidalPosEmb(dim = args.denoiser_time_dim, theta = args.denoiser_theta)

        self.time_mlp = nn.Sequential(
            self.sinu_pos_emb,
            nn.Linear(args.denoiser_time_dim, args.denoiser_time_dim),
            nn.GELU(),
            nn.Linear(args.denoiser_time_dim, args.denoiser_time_dim))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):

        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed

        if self.config.tem_mod in {'self_atten','temporal_bundling','node'}:
            b, ts, seq_len, _ = x.shape
            x = rearrange(x, 'b t n d -> (b t) n d')
        
        ## Must do spatial self-attention
        x = self.pos_drop(x)
        if not self.config.checkpoint_activations:
            for blk in self.blocks:
                x = blk(x)
        else:
            x = checkpoint_sequential(self.blocks, 4, x)
        x = self.norm(x)

        if self.config.tem_mod == 'self_atten':
            x = rearrange(x, '(b t) n d -> (b n) t d',t=ts)
            
            # temporal attention 
            for tem_blk in self.temporal_blocks:
                x = tem_blk(x)
                
            x = self.norm(x)
            x = rearrange(x, '(b n) t d -> b t n d',t=ts,b=b)
            x = self.final_layer(x)
        
        elif self.config.tem_mod == 'temporal_bundling':

            x = rearrange(x, '(b t) n d -> b t n d',t=ts)[:, -1, :, :]
            x = rearrange(x, 'b n d -> (b n) d')

            dt = torch.cumsum(torch.ones(1, self.time_window, 1, device=x.device), dim=1).repeat(1,1,self.latent_dim)
            # [batch*n_nodes, hidden_dim] -> 1DCNN([batch*n_nodes, 1, hidden_dim]) -> [batch*n_nodes, time_window]
            diff = self.output_mlp(x[:, None]).squeeze(1)
            diff = diff.reshape(-1, self.time_window, self.latent_dim)

            u_last = x[:, -self.latent_dim:] 
            u_last = u_last.unsqueeze(1) 
            x = u_last + dt * diff

            x = rearrange(x, '(b n) t d -> b t n d',b=b)
            x = self.final_layer(x)
            # x = x.reshape(-1, self.time_window*self.pred_var)  
            

        elif self.config.tem_mod == 'node':

            # from torchdiffeq import odeint
            from torchdiffeq import odeint_adjoint as odeint
            
            class ODEFunc(nn.Module):
                def __init__(self, derivative_net):
                    super().__init__()
                    self.net = derivative_net

                def forward(self, t, y):
                    return self.net(y)
            ode_func = ODEFunc(self.derivative_net)
            
            x = rearrange(x, '(b t) n d -> b t n d',t=ts)[:, -1, :, :].reshape(-1, x.size(-1))
            t = torch.linspace(0, 1, self.forecast_horizon + 1).to(x.device)
            pred_z = odeint(ode_func, x, t, method='dopri5')
            # Remove the initial state, permute to [batch*seq_len, time, features]
            pred_z = (pred_z[1:]).permute(1, 0, 2) 
            x = rearrange(pred_z, '(b n) t d -> b t n d',n=seq_len)
            x = self.final_layer(x)

        x = self.unpatch(x)

        return x

    def forward(self, x, t):
        t_embed = self.time_mlp(t)

        t_embed = t_embed.unsqueeze(2).unsqueeze(3)
        t_embed = t_embed.expand(-1, -1, x.shape[-1], x.shape[-1])
        x = torch.cat((x, t_embed), dim=1)

        x = self.forward_features(x)
        x = self.final_dropout(x)
        return x


def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict

if __name__ == '__main__':
    args = {
        "patch_size": 4,
        "num_attention_heads": 8,
        "hidden_size": 768,
        "embed_dim": 768,
        "num_layers": 12,
        "mixing_type": "sa",  # choices: ['afno', 'sa', 'ls', 'gfn', 'bfno']
        "fno_bias": False,
        "fno_blocks": 1,
        "fno_softshrink": 0.00,
        "double_skip": False,
        "checkpoint_activations": False,
        "ls_w": 4,
        "ls_dp_rank": 16,
        "input_size": 128,
        "in_chans": 4,
        "reduced_resolution": 4,
        "initial_step": 5,
        "forecast_horizon": 5,
        "tem_mod": "self_atten"
    }
    from argparse import Namespace
    args = Namespace(**args)
    # [B,C,H,W] or [B,T,C,H,W]
    a = torch.randn((1,5,4,128,128))
    model = self_atten(args=args)
    print(model(a,a,None,torch.nn.MSELoss())[0].shape)
