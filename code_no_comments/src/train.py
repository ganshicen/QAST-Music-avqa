from __future__ import print_function
import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
sys.path.append(ROOT.as_posix())
import torch
import torch.nn as nn
import torch.distributed as dist
import json
import os
from src.utils import arg_parse, seed_everything, setting, get_logger, set_logger, logging_config
from src.trainutils import get_model, get_dloaders, get_optim, train, evaluate, sync_processes, test
from src.training.ema import ModelEMA

def validate_standard_protocol(cfg):
    if not getattr(cfg, 'standard_protocol', False):
        return
    split_paths = {'train': os.path.abspath(cfg.data.train_annot), 'val': os.path.abspath(cfg.data.valid_annot), 'test': os.path.abspath(cfg.data.test_annot)}
    if len(set(split_paths.values())) != 3:
        raise ValueError(f'Standard protocol requires three distinct annotation files: {split_paths}')
    if 'trainval' in os.path.basename(split_paths['train']).lower():
        raise ValueError('Standard protocol forbids a combined train+val annotation file.')
    distill_path = getattr(cfg, 'distill_logits', None)
    if distill_path:
        name = os.path.basename(str(distill_path)).lower()
        if 'test' in name:
            raise ValueError('Standard protocol forbids test-derived distillation logits.')

    def sample_key(item):
        video = item.get('video_id', item.get('video', item.get('vid', '')))
        qid = item.get('question_id', item.get('qid', ''))
        return (str(video), str(qid), item.get('question', ''))
    split_keys = {}
    for (name, path) in split_paths.items():
        with open(path, 'r', encoding='utf-8') as stream:
            split_keys[name] = {sample_key(item) for item in json.load(stream)}
    for (left, right) in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        overlap = split_keys[left] & split_keys[right]
        if overlap:
            raise ValueError(f'Detected {len(overlap)} overlapping samples in {left}/{right}.')
    get_logger().info('Standard protocol audit passed: train=%d, val=%d, test=%d; no split overlap; no test-derived distillation.', len(split_keys['train']), len(split_keys['val']), len(split_keys['test']))

def main():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    args = arg_parse()
    (cfg, device, cur_rank) = setting(args)
    (writer, timestamp) = set_logger(cfg)
    logger = get_logger()
    save_dir = os.path.join(cfg.output_dir, timestamp)
    logging_config(cfg)
    seed_everything(cfg.seed)
    validate_standard_protocol(cfg)
    d_loaders = get_dloaders(cfg)
    model = get_model(cfg, device)
    (optim, sched) = get_optim(cfg, model, d_loaders['train'])
    raw_model = model.module if hasattr(model, 'module') else model
    ema_cfg = getattr(cfg.hyper_params, 'ema', None)
    ema_enabled = bool(ema_cfg and getattr(ema_cfg, 'enabled', False))
    ema = ModelEMA(raw_model, decay=float(ema_cfg.decay)) if ema_enabled else None
    best_acc = 0
    best_epoch = -1
    criterion = nn.CrossEntropyLoss()
    sync_processes()
    for epoch in range(1, cfg.epochs + 1):
        if dist.is_initialized() and dist.get_rank() == 0 or not dist.is_initialized():
            if writer is not None:
                for (idx, param_group) in enumerate(optim.param_groups):
                    current_lr = param_group['lr']
                    writer.add_scalar(f'train/lr', current_lr, epoch)
        logger.info(f'\n-------------- training epoch {epoch} --------------')
        train(cfg, epoch, device, d_loaders['train'], optim, criterion, model, writer, ema)
        if dist.is_initialized() and cur_rank == 0 or not dist.is_initialized():
            logger.info(f'\n-------------- validation epoch {epoch} --------------')
        sync_processes()
        (raw_acc, raw_loss) = evaluate(cfg, epoch, device, d_loaders['val'], criterion, model, writer)
        (acc, loss) = (raw_acc, raw_loss)
        weight_kind = 'raw'
        warmup_epochs = int(getattr(cfg.hyper_params.sched, 'warmup_epochs', 0))
        if ema is not None and epoch > warmup_epochs:
            with ema.average_parameters(raw_model):
                (ema_acc, ema_loss) = evaluate(cfg, epoch, device, d_loaders['val'], criterion, model, writer)
            if ema_acc > raw_acc:
                (acc, loss) = (ema_acc, ema_loss)
                weight_kind = 'ema'
        if cfg.hyper_params.sched.name == 'ReduceLROnPlateau':
            if cfg.hyper_params.sched.mode == 'max':
                sched.step(acc)
            elif cfg.hyper_params.sched.mode == 'min':
                sched.step(loss)
        else:
            sched.step(epoch)
        if acc >= best_acc and (not cfg.debug):
            best_acc = acc
            best_epoch = epoch
            if weight_kind == 'ema':
                context = ema.average_parameters(raw_model)
            else:
                from contextlib import nullcontext
                context = nullcontext()
            with context:
                sd = {key: value.detach().cpu().clone() for (key, value) in raw_model.state_dict().items()}
            new_sd = {}
            for (k, v) in sd.items():
                if 'video_encoder' not in k:
                    new_sd[k] = v
            logger.info(f'best model saved at epoch {epoch} with acc {best_acc} ({weight_kind})')
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    torch.save(new_sd, os.path.join(save_dir, f'best.pt'))
            else:
                torch.save(new_sd, os.path.join(save_dir, f'best.pt'))
            if not dist.is_initialized() or dist.get_rank() == 0:
                with open(os.path.join(save_dir, 'best_metadata.json'), 'w', encoding='utf-8') as stream:
                    json.dump({'epoch': epoch, 'validation_accuracy': float(best_acc), 'weight_kind': weight_kind}, stream, indent=2)
        logger.info(f'Epoch {epoch} done with {acc:3.2f} and loss {loss:.5f}.')
        logger.info(f'At epoch{best_epoch} best acc: {best_acc:3.2f}.')
    if not cfg.debug and getattr(cfg, 'run_test_after_training', True):
        logger.info(f'\nTesting with Best validation model... {cfg.data.test_annot}')
        cfg.mode = 'test'
        d_loaders = get_dloaders(cfg)['test']
        save_dir = Path(save_dir).absolute()
        best_path = save_dir / f'best.pt'
        original_dict = torch.load(best_path.as_posix())
        update_dict = {}
        for (name, param) in original_dict.items():
            if hasattr(model, 'module'):
                name = 'module.' + name
            update_dict[name] = param
        model.load_state_dict(update_dict, strict=False)
        test(cfg, device, d_loaders, model)
        if isinstance(cfg.data.test_annots, (list, tuple)):
            for (idx, test_annot) in enumerate(cfg.data.test_annots):
                logger.info(f'\nTesting with Best validation model... {test_annot}')
                cfg.data.test_annot = test_annot
                d_loaders = get_dloaders(cfg)['test']
                test(cfg, device, d_loaders, model)
    if dist.is_initialized():
        dist.destroy_process_group()
if __name__ == '__main__':
    main()
