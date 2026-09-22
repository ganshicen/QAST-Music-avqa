from __future__ import annotations
from contextlib import contextmanager
from typing import Dict, Iterator
import torch
import torch.nn as nn
from torch import Tensor

class ModelEMA:

    def __init__(self, model: nn.Module, decay: float=0.999):
        if not 0.0 <= decay < 1.0:
            raise ValueError('EMA decay must be in [0, 1)')
        self.decay = float(decay)
        self.shadow: Dict[str, Tensor] = {name: tensor.detach().clone().float() for (name, tensor) in model.state_dict().items() if torch.is_floating_point(tensor)}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        for (name, shadow) in self.shadow.items():
            current = state[name].detach().to(device=shadow.device, dtype=shadow.dtype)
            shadow.mul_(self.decay).add_(current, alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, Tensor]:
        return {name: tensor.detach().clone() for (name, tensor) in self.shadow.items()}

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        state = model.state_dict()
        backup = {name: state[name].detach().clone() for name in self.shadow}
        try:
            with torch.no_grad():
                for (name, shadow) in self.shadow.items():
                    state[name].copy_(shadow.to(device=state[name].device, dtype=state[name].dtype))
            yield
        finally:
            with torch.no_grad():
                restored = model.state_dict()
                for (name, tensor) in backup.items():
                    restored[name].copy_(tensor)
