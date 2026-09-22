from typing import Dict, List, Tuple
from collections import defaultdict
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from src.utils import get_logger
from src.dataset import AVQA_dataset, qtype2idx
from src.utils import calculate_parameters
from src.models.net import QA_AVAF
from src.models.net_tiger_adaptive import QA_TIGERAdaptive
from src.models.net_qast_native import QASTNative
from src.models.net_qast_grounded import QASTGroundedFusion
from src.models.net_qast_recent import QASTRecentFusion, QASTRecentFusionV2, QASTRecentFusionV3
from src.models.net_qast_evidence import QASTSingleEvidence
from src.models.net_qast_ehr import QASTEHR
from src.models.tspm import TSPM

class AverageMeter(object):

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self):
        self.values = defaultdict(float)
        self.count = 0

    def update(self, val: List[Tuple[str, float]], step_n: int):
        for (key, val) in val:
            self.values[key] += val
        self.count += step_n

    def get(self, key: str):
        return self.values[key] / self.count

def sync_processes():
    if dist.is_initialized():
        dist.barrier()

def gather_losses(epoch: int, batch_idx: int, loader_size: int, losses: List[Tuple[str, Tensor]], writer: SummaryWriter, device: torch.device) -> float:
    if dist.is_initialized():
        gather_losses = []
        with torch.no_grad():
            for (key, value) in losses:
                gather_v = torch.tensor(value).to(device)
                dist.all_reduce(gather_v, op=dist.ReduceOp.SUM)
                gather_losses.append((key, gather_v.item() / dist.get_world_size()))
        if dist.get_rank() == 0:
            for (key, value) in gather_losses:
                writer.add_scalar(f'train/loss/{key}', value, (epoch - 1) * loader_size + batch_idx)
        return gather_losses
    else:
        for (key, value) in losses:
            if writer is not None:
                writer.add_scalar(f'train/loss/{key}', value, (epoch - 1) * loader_size + batch_idx)
        return losses

def get_model(cfg: dict, device: torch.device):
    hyper_params = cfg.hyper_params
    if hyper_params.model_type == 'QAST-EHR-2025':
        model = QASTEHR(**hyper_params.model)
    elif hyper_params.model_type == 'QAST-SINGLE-EVIDENCE-2025':
        model = QASTSingleEvidence(**hyper_params.model)
    elif hyper_params.model_type.startswith('QAST-RECENT-V3'):
        model = QASTRecentFusionV3(**hyper_params.model)
    elif hyper_params.model_type.startswith('QAST-RECENT-V2'):
        model = QASTRecentFusionV2(**hyper_params.model)
    elif hyper_params.model_type.startswith('QAST-RECENT'):
        model = QASTRecentFusion(**hyper_params.model)
    elif hyper_params.model_type.startswith('QAST-GROUNDED'):
        model = QASTGroundedFusion(**hyper_params.model)
    elif hyper_params.model_type.startswith('QAST-NATIVE'):
        model = QASTNative(**hyper_params.model)
    elif hyper_params.model_type.startswith('QAST-STANDARD'):
        model = QA_TIGERAdaptive(**hyper_params.model)
    elif hyper_params.model_type.startswith(('QAST', 'QA-AVAF')):
        model = QA_AVAF(**hyper_params.model)
    elif hyper_params.model_type.startswith('TSPM'):
        model = TSPM(**hyper_params.model)
    else:
        raise NotImplementedError(f'Model type {hyper_params.model_type} is not implemented')
    model = model.to(device)
    if getattr(model, 'pretrained_report', None):
        report = model.pretrained_report
        get_logger().info('Initialized QAST shared blocks from %s (%d tensors); new grounding paths remain independently initialized.', report['checkpoint'], report['loaded_tensors'])
    if cfg.weight is not None and cfg.weight != '':
        logger = get_logger()
        weight = cfg.weight
        state = torch.load(weight)
        if getattr(cfg, 'skip_quest_encoder_weight', False):
            state = {k: v for (k, v) in state.items() if not k.startswith('quest_encoder.')}
        msg = model.load_state_dict(state, strict=False)
        logger.info(f'Missing keys: {json.dumps(msg.missing_keys, indent=4)}')
        logger.info(f'Unexpected keys: {json.dumps(msg.unexpected_keys, indent=4)}')
        logger.info(f"=> loaded successfully '{weight}'")
    if dist.is_initialized():
        model = DDP(model, device_ids=[dist.get_rank()], find_unused_parameters=False)
    else:
        model = nn.DataParallel(model)
    if cfg.mode == 'train':
        calculate_parameters(model)
    return model

def get_optim(cfg: dict, model: nn.Module, train_loader: DataLoader):
    logger = get_logger()
    new_lr = getattr(cfg.hyper_params.optim, 'new_lr', None)
    if new_lr is not None:
        new_module_names = ('adaptive_position.', 'cross_modal_grounding.', 'cross_ground_norm.', 'cross_ground_scale', 'spatial_grounding.question_to_query.', 'spatial_grounding.prompt_to_query.', 'spatial_grounding.motion_to_query.', 'spatial_grounding.context_refine.', 'spatial_grounding.query_scale', 'spatial_grounding.refine_scale', 'semantic_context.', 'semantic_project.', 'semantic_scale', 'recent_position.', 'path_router.', 'position_answer_adapter.', 'position_qtype_router.')
        named_parameters = list(model.named_parameters())
        new_params = [p for (n, p) in named_parameters if any((key in n for key in new_module_names))]
        base_params = [p for (n, p) in named_parameters if not any((key in n for key in new_module_names))]
        params = [{'params': base_params, 'lr': cfg.hyper_params.optim.lr}, {'params': new_params, 'lr': new_lr}]
        logger.info('Two-speed fine-tuning: %d base and %d new-module tensors.', len(base_params), len(new_params))
    elif cfg.hyper_params.optim.encoder_lr is not None:
        m = model.module if hasattr(model, 'module') else model
        other_params = [param for (name, param) in model.named_parameters() if 'video_encoder' not in name and 'quest_encoder' not in name and ('audio_encoder' not in name) and ('mllm' not in name)]
        encoder_params = [param for (name, param) in model.named_parameters() if 'video_encoder' in name or 'quest_encoder' in name or 'audio_encoder' in name or ('mllm' in name)]
        params = [{'params': other_params, 'lr': cfg.hyper_params.optim.lr}, {'params': encoder_params, 'lr': cfg.hyper_params.optim.encoder_lr}]
    else:
        params = model.parameters()
    optimizer_name = getattr(cfg.hyper_params.optim, 'name', 'Adam')
    optimizer_cls = optim.AdamW if optimizer_name == 'AdamW' else optim.Adam
    optimizer = optimizer_cls(params, lr=cfg.hyper_params.optim.lr, weight_decay=cfg.hyper_params.optim.weight_decay, betas=cfg.hyper_params.optim.betas)
    for param_group in optimizer.param_groups:
        logger.info('\n-------------- optimizer info --------------')
        logger.info(f"Learning rate: {param_group['lr']}")
        logger.info(f"Betas: {param_group['betas']}")
        logger.info(f"Eps: {param_group['eps']}")
        logger.info(f"Weight decay: {param_group['weight_decay']}")
    if 'cosine' in cfg.hyper_params.sched.name:
        from timm.scheduler.cosine_lr import CosineLRScheduler
        scheduler = CosineLRScheduler(optimizer, t_initial=cfg.epochs, lr_min=cfg.hyper_params.optim.min_lr, cycle_mul=1.0, cycle_decay=1.0, cycle_limit=1, warmup_t=cfg.hyper_params.sched.warmup_epochs, warmup_lr_init=cfg.hyper_params.optim.min_lr, warmup_prefix=False, t_in_epochs=True, noise_range_t=None, noise_pct=0.67, noise_std=1.0, noise_seed=42, k_decay=1.0, initialize=True)
    elif 'StepLR' in cfg.hyper_params.sched.name:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.hyper_params.sched.step_size, gamma=cfg.hyper_params.sched.gamma)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=cfg.hyper_params.sched.mode, factor=cfg.hyper_params.sched.factor, patience=cfg.hyper_params.sched.patience, verbose=cfg.hyper_params.sched.verbose)
    return (optimizer, scheduler)

def get_dloaders(cfg: dict) -> Dict[str, DataLoader]:
    hyper_params = cfg.data
    train_dataset = AVQA_dataset(cfg, mode=cfg.mode)
    val_dataset = AVQA_dataset(cfg, mode='valid')
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset)
        valid_sampler = DistributedSampler(val_dataset, shuffle=False)
        b_size = hyper_params.batch_size // dist.get_world_size()

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
    else:
        train_sampler = None
        valid_sampler = None
        b_size = hyper_params.batch_size
    train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=train_sampler is None and cfg.mode == 'train', num_workers=hyper_params.num_workers, pin_memory=True, worker_init_fn=seed_worker if dist.is_initialized() else None, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=hyper_params.eval_batch_size, shuffle=False, num_workers=hyper_params.num_workers, sampler=valid_sampler, pin_memory=True)
    return {f'{cfg.mode}': train_loader, 'val': val_loader}

def get_items(batch: dict, device: torch.device):
    reshaped_data = dict(quest=batch['quest'], audio=batch['audio'].to(device).float(), video=batch['video'].to(device).float(), qtype_label=batch['qtype_label'].to(device).long(), patch=batch['patch'].to(device).float() if 'patch' in batch else None, n_quest=batch['n_quest'] if 'n_quest' in batch else None, n_video=batch['n_video'].to(device).float() if 'n_video' in batch else None, n_audio=batch['n_audio'].to(device).float() if 'n_audio' in batch else None, n_patch=batch['n_patch'].to(device).float() if 'n_patch' in batch else None, prompt=batch['prompt'].to(device) if 'prompt' in batch else None, label=batch['label'].to(device).long().reshape(-1))
    if isinstance(reshaped_data['quest'], dict):
        reshaped_data['quest'] = {key: value.to(device) for (key, value) in reshaped_data['quest'].items()}
        if reshaped_data['n_quest'] is not None:
            reshaped_data['n_quest'] = {key: value.to(device) for (key, value) in reshaped_data['n_quest'].items()}
    else:
        reshaped_data['quest'] = reshaped_data['quest'].to(device)
        if reshaped_data['n_quest'] is not None:
            reshaped_data['n_quest'] = reshaped_data['n_quest'].to(device)
    return reshaped_data

def train(cfg: dict, epoch: int, device: torch.device, train_loader: DataLoader, optimizer: Optimizer, criterion: nn.Module, model: nn.Module, writer: SummaryWriter=None, ema=None):
    logger = get_logger()
    model.train()
    raw_model = model.module if hasattr(model, 'module') else model
    if hasattr(raw_model, 'set_training_epoch'):
        raw_model.set_training_epoch(epoch)
    avg_meter = AverageMeter()
    tot_batch = len(train_loader) - 1
    teacher_logp = None
    distill_path = getattr(cfg, 'distill_logits', None)
    if distill_path:
        teacher_logp = torch.load(distill_path, map_location='cpu')
        if len(teacher_logp) != len(train_loader.dataset):
            raise ValueError(f'teacher logit count {len(teacher_logp)} != dataset {len(train_loader.dataset)}')
    epoch_time = time.time()
    accumulation_steps = max(1, int(getattr(cfg, 'gradient_accumulation_steps', 1)))
    optimizer.zero_grad(set_to_none=True)
    for (batch_idx, sample) in enumerate(train_loader):
        start_time = time.time()
        reshaped_data = get_items(sample, device)
        amp_enabled = bool(getattr(cfg, 'amp', False)) and device.type == 'cuda'
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp_enabled):
            output = model(reshaped_data)
            target = reshaped_data['label']
            ce_loss = criterion(output['out'], target)
            loss = ce_loss
            losses = [('ce_loss', ce_loss)]
            if teacher_logp is not None:
                temperature = float(getattr(cfg, 'distill_temperature', 2.0))
                kd_weight = float(getattr(cfg, 'distill_weight', 0.35))
                indices = sample['_index'].long()
                soft_target = F.softmax(teacher_logp[indices].to(device) / temperature, dim=-1)
                kd_loss = F.kl_div(F.log_softmax(output['out'] / temperature, dim=-1), soft_target, reduction='batchmean') * temperature ** 2
                loss = (1.0 - kd_weight) * ce_loss + kd_weight * kd_loss
                losses.append(('distill_loss', kd_loss))
            for key in output:
                if 'loss' in key:
                    losses.append((key, output[key]))
                    loss += output[key]
        losses.append(('total_loss', loss))
        is_last_batch = batch_idx == len(train_loader) - 1
        window_start = batch_idx // accumulation_steps * accumulation_steps
        window_size = min(accumulation_steps, len(train_loader) - window_start)
        (loss / window_size).backward()
        if (batch_idx + 1) % accumulation_steps == 0 or is_last_batch:
            grad_clip = float(getattr(cfg.hyper_params.optim, 'grad_clip', 0.0))
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            warmup_epochs = int(getattr(cfg.hyper_params.sched, 'warmup_epochs', 0))
            if ema is not None and epoch > warmup_epochs:
                ema.update(raw_model)
            optimizer.zero_grad(set_to_none=True)
        losses = gather_losses(epoch, batch_idx, tot_batch, losses, writer, device)
        avg_meter.update(losses, step_n=1)
        if batch_idx % cfg.log_interval == 0 or batch_idx == len(train_loader) - 1:
            batch_t = time.time() - start_time
            elapsed_t = time.time() - epoch_time
            avg_time = elapsed_t / (batch_idx + 1)
            est_time = (tot_batch - batch_idx - 1) * avg_time / 60
            cur_batch = str(batch_idx).zfill(len(str(tot_batch)))
            batch_ratio = 100.0 * batch_idx / tot_batch
            log_string = f'[EST: {est_time:7.2f}m][Process Time: {batch_t:7.2f}s]- Epoch: {epoch} [{cur_batch}/{tot_batch} ({batch_ratio:3.0f}%)]\tLosses: '
            loss_string = ' '.join([f'{key}-{value:.4f}({avg_meter.get(key):.4f})' for (key, value) in losses])
            logger.info(msg=log_string + loss_string)
        if cfg.debug and batch_idx == 10:
            break

def evaluate(cfg: dict, epoch: int, device: torch.device, val_loader: DataLoader, criterion: nn.Module, model: nn.Module, writer: SummaryWriter=None):
    global qtype2idx
    logger = get_logger()
    model.eval()
    loss = 0
    (total, correct) = (0, 0)
    tot_tensor = torch.zeros(9, dtype=torch.long).to(device)
    correct_tensor = torch.zeros(9, dtype=torch.long).to(device)
    with torch.no_grad():
        for (batch_idx, sample) in enumerate(val_loader):
            reshaped_data = get_items(sample, device)
            qst_types = sample['type']
            target = reshaped_data['label']
            amp_enabled = bool(getattr(cfg, 'amp', False)) and device.type == 'cuda'
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp_enabled):
                output = model(reshaped_data)
            (_, predicted) = torch.max(output['out'].data, 1)
            total += predicted.size(0)
            correct += (predicted == target).sum().item()
            loss += criterion(output['out'], target) / len(val_loader)
            for (idx, (modal_type, qst_type)) in enumerate(zip(qst_types[0], qst_types[1])):
                gather_idx = qtype2idx[modal_type][qst_type]
                tot_tensor[gather_idx] += 1
                correct_tensor[gather_idx] += (predicted[idx] == target[idx]).long().item()
            if cfg.debug and batch_idx == 10:
                break
            if batch_idx % cfg.log_interval == 0 or batch_idx == len(val_loader) - 1:
                logger.info(f'Test progress: {batch_idx:3.0f}/{len(val_loader) - 1}')
    sync_processes()
    if dist.is_initialized():
        correct = torch.tensor(correct).to(device)
        total = torch.tensor(total).to(device)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        for idx in range(9):
            dist.all_reduce(tot_tensor[idx], op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_tensor[idx], op=dist.ReduceOp.SUM)
    acc = correct / total * 100.0
    loss = loss.item()
    if writer is not None:
        writer.add_scalar('valid/acc/Total', acc, epoch)
    for modality in ['Audio', 'Visual', 'Audio-Visual']:
        modality_corr = 0
        modality_tot = 0
        for qst_type in qtype2idx[modality]:
            corr = correct_tensor[qtype2idx[modality][qst_type]].item()
            tot = tot_tensor[qtype2idx[modality][qst_type]].item()
            modality_corr += corr
            modality_tot += tot
            value = corr / tot * 100.0
            key = f'{modality}/{qst_type}'
            logger.info(f'Epoch {epoch} - {key:>24} accuracy: {value:.2f}({corr}/{tot})')
            if writer is not None:
                writer.add_scalar(f'valid/acc/{key}', corr / tot * 100.0, epoch)
        modality_acc = modality_corr / modality_tot * 100.0
        logger.info(f'Epoch {epoch} - {modality:>24} accuracy: {modality_acc:.2f}({modality_corr}/{modality_tot})')
        if writer is not None:
            writer.add_scalar(f'valid/acc/{modality}', modality_acc, epoch)
    key = 'Total'
    logger.info(f'Epoch {epoch} - {key:>24} accuracy: {acc:.2f}({correct}/{total})')
    return (acc, loss)

def test(cfg: dict, device: torch.device, val_loader: DataLoader, model: nn.Module):
    global qtype2idx
    logger = get_logger()
    model.eval()
    (total, correct) = (0, 0)
    tot_tensor = torch.zeros(9, dtype=torch.long).to(device)
    correct_tensor = torch.zeros(9, dtype=torch.long).to(device)
    with torch.no_grad():
        for (batch_idx, sample) in enumerate(val_loader):
            reshaped_data = get_items(sample, device)
            qst_types = sample['type']
            target = reshaped_data['label']
            amp_enabled = bool(getattr(cfg, 'amp', False)) and device.type == 'cuda'
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp_enabled):
                output = model(reshaped_data)
            (_, predicted) = torch.max(output['out'].data, 1)
            total += predicted.size(0)
            correct += (predicted == target).sum().item()
            for (idx, (modal_type, qst_type)) in enumerate(zip(qst_types[0], qst_types[1])):
                gather_idx = qtype2idx[modal_type][qst_type]
                tot_tensor[gather_idx] += 1
                correct_tensor[gather_idx] += (predicted[idx] == target[idx]).long().item()
            if cfg.debug and batch_idx == 10:
                break
            if batch_idx % cfg.log_interval == 0 or batch_idx == len(val_loader) - 1:
                logger.info(f'Test progress: {batch_idx:3.0f}/{len(val_loader) - 1}')
    sync_processes()
    if dist.is_initialized():
        correct = torch.tensor(correct).to(device)
        total = torch.tensor(total).to(device)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        for idx in range(9):
            dist.all_reduce(tot_tensor[idx], op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_tensor[idx], op=dist.ReduceOp.SUM)
    acc = correct / total * 100.0
    for modality in ['Audio', 'Visual', 'Audio-Visual']:
        modality_corr = 0
        modality_tot = 0
        for qst_type in qtype2idx[modality]:
            corr = correct_tensor[qtype2idx[modality][qst_type]].item()
            tot = tot_tensor[qtype2idx[modality][qst_type]].item()
            modality_corr += corr
            modality_tot += tot
            value = corr / tot * 100.0
            key = f'{modality}/{qst_type}'
            logger.info(f'Test {key:>24} accuracy: {value:.2f}({corr}/{tot})')
        modality_acc = modality_corr / modality_tot * 100.0
        logger.info(f'Test {modality:>24} accuracy: {modality_acc:.2f}({modality_corr}/{modality_tot})')
    key = 'Total avg'
    logger.info(f'Test {key:>24} accuracy: {acc:.2f}({correct}/{total})')
    return acc
