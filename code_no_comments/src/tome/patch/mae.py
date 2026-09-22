import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from src.tome.utils import parse_r
from .timm import ToMeBlock, ToMeAttention

def make_tome_class(transformer_class):

    class ToMeVisionTransformer(transformer_class):

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info['r'] = parse_r(len(self.blocks), self.r)
            self._tome_info['size'] = None
            self._tome_info['source'] = None
            return super().forward(*args, **kwdargs)

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            B = x.shape[0]
            x = self.patch_embed(x)
            T = x.shape[1]
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)
            for blk in self.blocks:
                x = blk(x)
            if self.global_pool:
                if self._tome_info['size'] is not None:
                    x = (x * self._tome_info['size'])[:, 1:, :].sum(dim=1) / T
                else:
                    x = x[:, 1:, :].mean(dim=1)
                outcome = self.fc_norm(x)
            else:
                x = self.norm(x)
                outcome = x[:, 0]
            return outcome
    return ToMeVisionTransformer

def apply_patch(model: VisionTransformer, trace_source: bool=False, prop_attn: bool=False):
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
