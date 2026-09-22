from __future__ import annotations
import argparse
import importlib.util
import math
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from box import Box
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.models.net_qast_ehr import QASTEHR
from src.trainutils import get_dloaders, get_items

def load_config(path: Path) -> Box:
    spec = importlib.util.spec_from_file_location('ehr_smoke_config', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = Box(module.config)
    cfg.mode = 'train'
    cfg.debug = False
    cfg.weight = ''
    return cfg

def total_loss(output: dict, labels: torch.Tensor) -> torch.Tensor:
    result = F.cross_entropy(output['out'], labels)
    return result + sum((value for (key, value) in output.items() if key.startswith('loss_')))

def require_finite(name: str, named_tensors) -> None:
    failed = [key for (key, tensor) in named_tensors if not torch.isfinite(tensor).all()]
    if failed:
        raise FloatingPointError(f'non-finite {name}: {failed}')

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=150)
    args = parser.parse_args()
    cfg = load_config(args.config.resolve())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QASTEHR(**cfg.hyper_params.model).to(device).train()
    model.set_training_epoch(1)
    loader = get_dloaders(cfg)['train']
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.hyper_params.optim.lr, weight_decay=cfg.hyper_params.optim.weight_decay, betas=tuple(cfg.hyper_params.optim.betas))
    first = get_items(next(iter(loader)), device)
    amp = device.type == 'cuda'
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        output = model(first)
        loss = total_loss(output, first['label'])
    loss.backward()
    missing = [name for (name, parameter) in model.named_parameters() if parameter.requires_grad and parameter.grad is None]
    if missing:
        raise RuntimeError(f'missing gradients: {missing}')
    require_finite('gradients', ((name, parameter.grad) for (name, parameter) in model.named_parameters()))
    optimizer.zero_grad(set_to_none=True)
    initial = None
    final = None
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            output = model(first)
            batch_loss = total_loss(output, first['label'])
        if initial is None:
            initial = float(batch_loss.detach())
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.hyper_params.optim.grad_clip)
        optimizer.step()
        final = float(batch_loss.detach())
    if initial is None or final is None or final >= initial:
        raise RuntimeError(f'one-batch loss did not decrease: {initial} -> {final}')
    iterator = iter(loader)
    ratio_sum = 0.0
    for step in range(1, args.steps + 1):
        try:
            sample = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            sample = next(iterator)
        batch = get_items(sample, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            output = model(batch)
            loss = total_loss(output, batch['label'])
        if not math.isfinite(float(loss.detach())):
            raise FloatingPointError(f'non-finite loss at step {step}')
        loss.backward()
        require_finite('gradients', ((name, parameter.grad) for (name, parameter) in model.named_parameters()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.hyper_params.optim.grad_clip)
        optimizer.step()
        require_finite('parameters', model.named_parameters())
        ratio_sum += float(output['position_ratio'].detach())
        if step == 1 or step % 25 == 0 or step == args.steps:
            print(f"step={step}/{args.steps} loss={float(loss.detach()):.6f} position_ratio={float(output['position_ratio'].detach()):.5f}", flush=True)
    mean_ratio = ratio_sum / max(args.steps, 1)
    if not 0.05 <= mean_ratio <= 0.1:
        raise RuntimeError(f'position utilization outside target range: {mean_ratio}')
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == 'cuda' else 0.0
    print(f'all trainable parameters received finite gradients; mean_position_ratio={mean_ratio:.5f}; peak_gpu_gib={peak:.3f}')
    print(f'{args.steps}-step BF16 smoke passed')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
