from typing import Tuple
import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from src.tome.merge import bipartite_soft_matching, merge_source, merge_wavg
from src.tome.utils import parse_r

class ToMeBlock(Block):

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, 'drop_path1') else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, 'drop_path2') else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_size = self._tome_info['size'] if self._tome_info['prop_attn'] else None
        (x_attn, metric) = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)
        r = self._tome_info['r'].pop(0)
        if r > 0:
            (merge, _) = bipartite_soft_matching(metric, r, self._tome_info['class_token'], self._tome_info['distill_token'])
            if self._tome_info['trace_source']:
                self._tome_info['source'] = merge_source(merge, x, self._tome_info['source'])
            (x, self._tome_info['size']) = merge_wavg(merge, x, self._tome_info['size'])
        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x

class ToMeAttention(Attention):

    def forward(self, x: torch.Tensor, size: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        (B, N, C) = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        (q, k, v) = (qkv[0], qkv[1], qkv[2])
        attn = q @ k.transpose(-2, -1) * self.scale
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return (x, k.mean(1))

def make_tome_class(transformer_class):

    class ToMeVisionTransformer(transformer_class):

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info['r'] = parse_r(len(self.blocks), self.r)
            self._tome_info['size'] = None
            self._tome_info['source'] = None
            return super().forward(*args, **kwdargs)
            return x
    return ToMeVisionTransformer

def apply_patch(model: VisionTransformer, trace_source: bool=False, prop_attn: bool=True):
    ToMeVisionTransformer = make_tome_class(model.__class__)
    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {'r': model.r, 'size': None, 'source': None, 'trace_source': trace_source, 'prop_attn': prop_attn, 'class_token': model.cls_token is not None, 'distill_token': False}
    if hasattr(model, 'dist_token') and model.dist_token is not None:
        model._tome_info['distill_token'] = True
    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention
