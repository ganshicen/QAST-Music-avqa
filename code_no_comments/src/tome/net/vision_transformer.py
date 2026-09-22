import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, RmsNorm, PatchDropout, use_fused_attn, SwiGLUPacked
from ._builder import build_model_with_cfg
from ._manipulate import named_apply, checkpoint_seq, adapt_input_conv
from ._registry import generate_default_cfgs, register_model, register_model_deprecations
__all__ = ['VisionTransformer']
_logger = logging.getLogger(__name__)

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0.0, proj_drop=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.fused_attn = use_fused_attn()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        (B, N, C) = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        (q, k, v) = qkv.unbind(0)
        (q, k) = (self.q_norm(q), self.k_norm(k))
        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):

    def __init__(self, dim, init_values=1e-05, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_norm=False, proj_drop=0.0, attn_drop=0.0, init_values=None, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_layer=Mlp):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=proj_drop, norm_layer=norm_layer)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class ResPostBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_norm=False, proj_drop=0.0, attn_drop=0.0, init_values=None, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_layer=Mlp):
        super().__init__()
        self.init_values = init_values
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=proj_drop, norm_layer=norm_layer)
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.init_weights()

    def init_weights(self):
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x):
        x = x + self.drop_path1(self.norm1(self.attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x

class ParallelScalingBlock(nn.Module):
    fused_attn: Final[bool]

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_norm=False, proj_drop=0.0, attn_drop=0.0, init_values=None, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_layer=None):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.fused_attn = use_fused_attn()
        mlp_hidden_dim = int(mlp_ratio * dim)
        in_proj_out_dim = mlp_hidden_dim + 3 * dim
        self.in_norm = norm_layer(dim)
        self.in_proj = nn.Linear(dim, in_proj_out_dim, bias=qkv_bias)
        self.in_split = [mlp_hidden_dim] + [dim] * 3
        if qkv_bias:
            self.register_buffer('qkv_bias', None)
            self.register_parameter('mlp_bias', None)
        else:
            self.register_buffer('qkv_bias', torch.zeros(3 * dim), persistent=False)
            self.mlp_bias = nn.Parameter(torch.zeros(mlp_hidden_dim))
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_out_proj = nn.Linear(dim, dim)
        self.mlp_drop = nn.Dropout(proj_drop)
        self.mlp_act = act_layer()
        self.mlp_out_proj = nn.Linear(mlp_hidden_dim, dim)
        self.ls = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        (B, N, C) = x.shape
        y = self.in_norm(x)
        if self.mlp_bias is not None:
            y = F.linear(y, self.in_proj.weight, torch.cat((self.qkv_bias, self.mlp_bias)))
        else:
            y = self.in_proj(y)
        (x_mlp, q, k, v) = torch.split(y, self.in_split, dim=-1)
        q = self.q_norm(q.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        if self.fused_attn:
            x_attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_attn = attn @ v
        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)
        x_attn = self.attn_out_proj(x_attn)
        x_mlp = self.mlp_act(x_mlp)
        x_mlp = self.mlp_drop(x_mlp)
        x_mlp = self.mlp_out_proj(x_mlp)
        y = self.drop_path(self.ls(x_attn + x_mlp))
        x = x + y
        return x

class ParallelThingsBlock(nn.Module):

    def __init__(self, dim, num_heads, num_parallel=2, mlp_ratio=4.0, qkv_bias=False, qk_norm=False, init_values=None, proj_drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_layer=Mlp):
        super().__init__()
        self.num_parallel = num_parallel
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(num_parallel):
            self.attns.append(nn.Sequential(OrderedDict([('norm', norm_layer(dim)), ('attn', Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=proj_drop, norm_layer=norm_layer)), ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()), ('drop_path', DropPath(drop_path) if drop_path > 0.0 else nn.Identity())])))
            self.ffns.append(nn.Sequential(OrderedDict([('norm', norm_layer(dim)), ('mlp', mlp_layer(dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)), ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()), ('drop_path', DropPath(drop_path) if drop_path > 0.0 else nn.Identity())])))

    def _forward_jit(self, x):
        x = x + torch.stack([attn(x) for attn in self.attns]).sum(dim=0)
        x = x + torch.stack([ffn(x) for ffn in self.ffns]).sum(dim=0)
        return x

    @torch.jit.ignore
    def _forward(self, x):
        x = x + sum((attn(x) for attn in self.attns))
        x = x + sum((ffn(x) for ffn in self.ffns))
        return x

    def forward(self, x):
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return self._forward_jit(x)
        else:
            return self._forward(x)

class VisionTransformer(nn.Module):
    dynamic_img_size: Final[bool]

    def __init__(self, img_size: Union[int, Tuple[int, int]]=224, patch_size: Union[int, Tuple[int, int]]=16, in_chans: int=3, num_classes: int=1000, global_pool: str='token', embed_dim: int=768, depth: int=12, num_heads: int=12, mlp_ratio: float=4.0, qkv_bias: bool=True, qk_norm: bool=False, init_values: Optional[float]=None, class_token: bool=True, no_embed_class: bool=False, pre_norm: bool=False, fc_norm: Optional[bool]=None, dynamic_img_size: bool=False, dynamic_img_pad: bool=False, drop_rate: float=0.0, pos_drop_rate: float=0.0, patch_drop_rate: float=0.0, proj_drop_rate: float=0.0, attn_drop_rate: float=0.0, drop_path_rate: float=0.0, weight_init: str='', embed_layer: Callable=PatchEmbed, norm_layer: Optional[Callable]=None, act_layer: Optional[Callable]=None, block_fn: Callable=Block, mlp_layer: Callable=Mlp):
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-06)
        act_layer = act_layer or nn.GELU
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False
        embed_args = {}
        if dynamic_img_size:
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, bias=not pre_norm, dynamic_img_pad=dynamic_img_pad, **embed_args)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(patch_drop_rate, num_prefix_tokens=self.num_prefix_tokens)
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, init_values=init_values, proj_drop=proj_drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, mlp_layer=mlp_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.0
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-06)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^cls_token|pos_embed|patch_embed', blocks=[('^blocks\\.(\\d+)', None), ('^norm', (99999,))])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x):
        if self.dynamic_img_size:
            (B, H, W, C) = x.shape
            pos_embed = resample_abs_pos_embed(self.pos_embed, (H, W), num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens)
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed
        if self.no_embed_class:
            x = x + pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + pos_embed
        return self.pos_drop(x)

    def _intermediate_layers(self, x: torch.Tensor, n: Union[int, Sequence]=1):
        (outputs, num_blocks) = ([], len(self.blocks))
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for (i, blk) in enumerate(self.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)
        return outputs

    def get_intermediate_layers(self, x: torch.Tensor, n: Union[int, Sequence]=1, reshape: bool=False, return_class_token: bool=False, norm: bool=False) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]
        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous() for out in outputs]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and (not torch.jit.is_scripting()):
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

def init_weights_vit_timm(module: nn.Module, name: str=''):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()

def init_weights_vit_jax(module: nn.Module, name: str='', head_bias: float=0.0):
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-06) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()

def init_weights_vit_moco(module: nn.Module, name: str=''):
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            val = math.sqrt(6.0 / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()

def get_init_weights_vit(mode='jax', head_bias: float=0.0):
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm

def resize_pos_embed(posemb, posemb_new, num_prefix_tokens=1, gs_new=(), interpolation='bicubic', antialias=False):
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        (posemb_prefix, posemb_grid) = (posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:])
        ntok_new -= num_prefix_tokens
    else:
        (posemb_prefix, posemb_grid) = (posemb[:, :0], posemb[0])
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info(f'Resized position embedding: {posemb.shape} ({[gs_old, gs_old]}) to {posemb_new.shape} ({gs_new}).')
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode=interpolation, antialias=antialias, align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb

@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str=''):
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)
    w = np.load(checkpoint_path)
    interpolation = 'bilinear'
    antialias = False
    big_vision = False
    if not prefix:
        if 'opt/target/embedding/kernel' in w:
            prefix = 'opt/target/'
        elif 'params/embedding/kernel' in w:
            prefix = 'params/'
            big_vision = True
    if hasattr(model.patch_embed, 'backbone'):
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for (i, stage) in enumerate(backbone.stages):
                for (j, block) in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    if embed_conv_w.shape[-2:] != model.patch_embed.proj.weight.shape[-2:]:
        embed_conv_w = resample_patch_embed(embed_conv_w, model.patch_embed.proj.weight.shape[-2:], interpolation=interpolation, antialias=antialias, verbose=True)
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    if model.cls_token is not None:
        model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    if big_vision:
        pos_embed_w = _n2p(w[f'{prefix}pos_embedding'], t=False)
    else:
        pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        old_shape = pos_embed_w.shape
        num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
        pos_embed_w = resample_abs_pos_embed(pos_embed_w, new_size=model.patch_embed.grid_size, num_prefix_tokens=num_prefix_tokens, interpolation=interpolation, antialias=antialias, verbose=True)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    (mha_sub, b_sub, ln1_sub) = (0, 0, 1) if big_vision else (1, 3, 2)
    for (i, block) in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_{mha_sub}/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([_n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([_n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/bias']))

def _convert_openai_clip(state_dict, model):
    out_dict = {}
    swaps = [('visual.', ''), ('conv1', 'patch_embed.proj'), ('positional_embedding', 'pos_embed'), ('transformer.resblocks.', 'blocks.'), ('ln_pre', 'norm_pre'), ('ln_post', 'norm'), ('ln_', 'norm'), ('in_proj_', 'qkv.'), ('out_proj', 'proj'), ('mlp.c_fc', 'mlp.fc1'), ('mlp.c_proj', 'mlp.fc2')]
    for (k, v) in state_dict.items():
        if not k.startswith('visual.'):
            continue
        for sp in swaps:
            k = k.replace(sp[0], sp[1])
        if k == 'proj':
            k = 'head.weight'
            v = v.transpose(0, 1)
            out_dict['head.bias'] = torch.zeros(v.shape[0])
        elif k == 'class_embedding':
            k = 'cls_token'
            v = v.unsqueeze(0).unsqueeze(1)
        elif k == 'pos_embed':
            v = v.unsqueeze(0)
            if v.shape[1] != model.pos_embed.shape[1]:
                v = resize_pos_embed(v, model.pos_embed, 0 if getattr(model, 'no_embed_class') else getattr(model, 'num_prefix_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict

def _convert_dinov2(state_dict, model):
    import re
    out_dict = {}
    for (k, v) in state_dict.items():
        if k == 'mask_token':
            continue
        elif re.match('blocks\\.(\\d+)\\.mlp\\.w12\\.(?:weight|bias)', k):
            out_dict[k.replace('w12', 'fc1')] = v
            continue
        elif re.match('blocks\\.(\\d+)\\.mlp\\.w3\\.(?:weight|bias)', k):
            out_dict[k.replace('w3', 'fc2')] = v
            continue
        out_dict[k] = v
    return out_dict

def _convert_ijepa(state_dict, model):
    out_dict = {}
    for (k, v) in state_dict['encoder'].items():
        if k.startswith('module.'):
            k = k[7:]
        if k.startswith('norm.'):
            k = 'fc_norm.' + k[5:]
        out_dict[k] = v
    return out_dict

def checkpoint_filter_fn(state_dict, model, adapt_layer_scale=False, interpolation='bicubic', antialias=True):
    import re
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    if 'visual.class_embedding' in state_dict:
        return _convert_openai_clip(state_dict, model)
    if 'mask_token' in state_dict:
        state_dict = _convert_dinov2(state_dict, model)
    if 'encoder' in state_dict:
        state_dict = _convert_ijepa(state_dict, model)
    for (k, v) in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            (O, I, H, W) = model.patch_embed.proj.weight.shape
            if len(v.shape) < 4:
                (O, I, H, W) = model.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(v, (H, W), interpolation=interpolation, antialias=antialias, verbose=True)
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(v, new_size=model.patch_embed.grid_size, num_prefix_tokens=num_prefix_tokens, interpolation=interpolation, antialias=antialias, verbose=True)
        elif adapt_layer_scale and 'gamma_' in k:
            k = re.sub('gamma_([0-9])', 'ls\\1.gamma', k)
        elif 'pre_logits' in k:
            continue
        out_dict[k] = v
    return out_dict

def _cfg(url='', **kwargs):
    return {'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None, 'crop_pct': 0.9, 'interpolation': 'bicubic', 'fixed_input_size': True, 'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD, 'first_conv': 'patch_embed.proj', 'classifier': 'head', **kwargs}
default_cfgs = generate_default_cfgs({})

def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')
    if 'flexi' in variant:
        _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn
    return build_model_with_cfg(VisionTransformer, variant, pretrained, pretrained_filter_fn=_filter_fn, **kwargs)

@register_model
def vit_tiny_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_tiny_patch16_384(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer('vit_tiny_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_small_patch32_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch32_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_small_patch32_384(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch32_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_small_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_small_patch16_384(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_small_patch8_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=8, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch32_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vit_base_patch32_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch32_384(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vit_base_patch32_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch16_384(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vit_base_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch8_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vit_base_patch8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_large_patch32_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('vit_large_patch32_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_large_patch32_384(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('vit_large_patch32_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_large_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('vit_large_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_large_patch16_384(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('vit_large_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_large_patch14_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('vit_large_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_huge_patch14_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16)
    model = _create_vision_transformer('vit_huge_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_giant_patch14_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=14, embed_dim=1408, mlp_ratio=48 / 11, depth=40, num_heads=16)
    model = _create_vision_transformer('vit_giant_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_gigantic_patch14_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=14, embed_dim=1664, mlp_ratio=64 / 13, depth=48, num_heads=16)
    model = _create_vision_transformer('vit_gigantic_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch16_224_miil(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False)
    model = _create_vision_transformer('vit_base_patch16_224_miil', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_medium_patch16_gap_240(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=512, depth=12, num_heads=8, class_token=False, global_pool='avg', qkv_bias=False, init_values=1e-06, fc_norm=False)
    model = _create_vision_transformer('vit_medium_patch16_gap_240', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_medium_patch16_gap_256(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=512, depth=12, num_heads=8, class_token=False, global_pool='avg', qkv_bias=False, init_values=1e-06, fc_norm=False)
    model = _create_vision_transformer('vit_medium_patch16_gap_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_medium_patch16_gap_384(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=512, depth=12, num_heads=8, class_token=False, global_pool='avg', qkv_bias=False, init_values=1e-06, fc_norm=False)
    model = _create_vision_transformer('vit_medium_patch16_gap_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch16_gap_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=16, class_token=False, global_pool='avg', fc_norm=False)
    model = _create_vision_transformer('vit_base_patch16_gap_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch32_clip_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer('vit_base_patch32_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch32_clip_384(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer('vit_base_patch32_clip_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch32_clip_448(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer('vit_base_patch32_clip_448', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch16_clip_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer('vit_base_patch16_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch16_clip_384(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer('vit_base_patch16_clip_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_large_patch14_clip_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer('vit_large_patch14_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_large_patch14_clip_336(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer('vit_large_patch14_clip_336', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_huge_patch14_clip_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer('vit_huge_patch14_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_huge_patch14_clip_336(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer('vit_huge_patch14_clip_336', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_giant_patch14_clip_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=14, embed_dim=1408, mlp_ratio=48 / 11, depth=40, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer('vit_giant_patch14_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_gigantic_patch14_clip_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=14, embed_dim=1664, mlp_ratio=64 / 13, depth=48, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer('vit_gigantic_patch14_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch32_plus_256(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=32, embed_dim=896, depth=12, num_heads=14, init_values=1e-05)
    model = _create_vision_transformer('vit_base_patch32_plus_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch16_plus_240(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=896, depth=12, num_heads=14, init_values=1e-05)
    model = _create_vision_transformer('vit_base_patch16_plus_240', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch16_rpn_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, init_values=1e-05, class_token=False, block_fn=ResPostBlock, global_pool='avg')
    model = _create_vision_transformer('vit_base_patch16_rpn_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_small_patch16_36x1_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=384, depth=36, num_heads=6, init_values=1e-05)
    model = _create_vision_transformer('vit_small_patch16_36x1_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_small_patch16_18x2_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=384, depth=18, num_heads=6, init_values=1e-05, block_fn=ParallelThingsBlock)
    model = _create_vision_transformer('vit_small_patch16_18x2_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch16_18x2_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=768, depth=18, num_heads=12, init_values=1e-05, block_fn=ParallelThingsBlock)
    model = _create_vision_transformer('vit_base_patch16_18x2_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def eva_large_patch14_196(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, global_pool='avg')
    model = _create_vision_transformer('eva_large_patch14_196', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def eva_large_patch14_336(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, global_pool='avg')
    model = _create_vision_transformer('eva_large_patch14_336', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def flexivit_small(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, no_embed_class=True)
    model = _create_vision_transformer('flexivit_small', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def flexivit_base(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, no_embed_class=True)
    model = _create_vision_transformer('flexivit_base', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def flexivit_large(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, no_embed_class=True)
    model = _create_vision_transformer('flexivit_large', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch16_xp_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, pre_norm=True, no_embed_class=True, norm_layer=RmsNorm, block_fn=ParallelScalingBlock, qkv_bias=False, qk_norm=True)
    model = _create_vision_transformer('vit_base_patch16_xp_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_large_patch14_xp_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, pre_norm=True, no_embed_class=True, norm_layer=RmsNorm, block_fn=ParallelScalingBlock, qkv_bias=False, qk_norm=True)
    model = _create_vision_transformer('vit_large_patch14_xp_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_huge_patch14_xp_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, pre_norm=True, no_embed_class=True, norm_layer=RmsNorm, block_fn=ParallelScalingBlock, qkv_bias=False, qk_norm=True)
    model = _create_vision_transformer('vit_huge_patch14_xp_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_small_patch14_dinov2(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=14, embed_dim=384, depth=12, num_heads=6, init_values=1e-05, img_size=518)
    model = _create_vision_transformer('vit_small_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch14_dinov2(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=14, embed_dim=768, depth=12, num_heads=12, init_values=1e-05, img_size=518)
    model = _create_vision_transformer('vit_base_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_large_patch14_dinov2(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, init_values=1e-05, img_size=518)
    model = _create_vision_transformer('vit_large_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_giant_patch14_dinov2(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=14, embed_dim=1536, depth=40, num_heads=24, init_values=1e-05, mlp_ratio=2.66667 * 2, mlp_layer=SwiGLUPacked, img_size=518, act_layer=nn.SiLU)
    model = _create_vision_transformer('vit_giant_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_huge_patch14_224_ijepa(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, class_token=False, global_pool='avg')
    model = _create_vision_transformer('vit_huge_patch14_224_ijepa', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_huge_patch16_448_ijepa(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=1280, depth=32, num_heads=16, class_token=False, global_pool='avg', img_size=448)
    model = _create_vision_transformer('vit_huge_patch16_448_ijepa', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_gigantic_patch16_224_ijepa(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=1664, mlp_ratio=64 / 13, depth=48, num_heads=16)
    model = _create_vision_transformer('vit_gigantic_patch16_224_ijepa', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
register_model_deprecations(__name__, {'vit_tiny_patch16_224_in21k': 'vit_tiny_patch16_224.augreg_in21k', 'vit_small_patch32_224_in21k': 'vit_small_patch32_224.augreg_in21k', 'vit_small_patch16_224_in21k': 'vit_small_patch16_224.augreg_in21k', 'vit_base_patch32_224_in21k': 'vit_base_patch32_224.augreg_in21k', 'vit_base_patch16_224_in21k': 'vit_base_patch16_224.augreg_in21k', 'vit_base_patch8_224_in21k': 'vit_base_patch8_224.augreg_in21k', 'vit_large_patch32_224_in21k': 'vit_large_patch32_224.orig_in21k', 'vit_large_patch16_224_in21k': 'vit_large_patch16_224.augreg_in21k', 'vit_huge_patch14_224_in21k': 'vit_huge_patch14_224.orig_in21k', 'vit_base_patch32_224_sam': 'vit_base_patch32_224.sam', 'vit_base_patch16_224_sam': 'vit_base_patch16_224.sam', 'vit_small_patch16_224_dino': 'vit_small_patch16_224.dino', 'vit_small_patch8_224_dino': 'vit_small_patch8_224.dino', 'vit_base_patch16_224_dino': 'vit_base_patch16_224.dino', 'vit_base_patch8_224_dino': 'vit_base_patch8_224.dino', 'vit_base_patch16_224_miil_in21k': 'vit_base_patch16_224_miil.in21k', 'vit_base_patch32_224_clip_laion2b': 'vit_base_patch32_clip_224.laion2b', 'vit_large_patch14_224_clip_laion2b': 'vit_large_patch14_clip_224.laion2b', 'vit_huge_patch14_224_clip_laion2b': 'vit_huge_patch14_clip_224.laion2b', 'vit_giant_patch14_224_clip_laion2b': 'vit_giant_patch14_clip_224.laion2b'})
