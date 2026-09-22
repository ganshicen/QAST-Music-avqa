import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
sys.path.append(ROOT.as_posix())
import json
import random
import distro
import logging
import platform
import argparse
import warnings
import numpy as np
import importlib.util
import zipfile
from box import Box
from typing import Tuple
from logging import getLogger
from datetime import datetime
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Question Answering')
    parser.add_argument('--config', type=str, default='./configs/stage2.py', help='Path to the config file')
    parser.add_argument('--distributed', action='store_true', help='Use Distributed Data Parallel (DDP) if set; otherwise use Data Parallel (DP)')
    parser.add_argument('--debug', action='store_true', help='Debugging')
    parser.add_argument('--weight', type=str, default='', help='Path to the model weight file')
    parser.add_argument('--mode', type=str, default='train', help='Mode (train or test)')
    parser.add_argument('--topK', type=int, default=-1, help='topK number for selection of experts')
    parser.add_argument('--n_experts', type=int, default=-1, help='Number of experts')
    parser.add_argument('--seed', type=int, default=713, help='Random seed')
    parser.add_argument('--output_path', type=str, default='', help='Path to save the output')
    return parser.parse_args()

def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        torch.cuda.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
        np.random.seed(seed + rank)
        random.seed(seed + rank)

def setting(args: argparse.Namespace) -> Tuple[Box, torch.device]:
    spec = importlib.util.spec_from_file_location('config', args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    conf = Box(config_module.config)
    conf.seed = args.seed
    conf.mode = args.mode
    conf.debug = args.debug
    conf.weight = args.weight
    conf.output_path = args.output_path
    if args.topK > 0:
        conf.hyper_params.model.topK = args.topK
    if args.n_experts > 0:
        conf.hyper_params.model.num_experts = args.n_experts
    seed_everything(conf.seed)
    if args.distributed:
        cur_rank = int(os.environ['LOCAL_RANK'])
        conf.cur_rank = cur_rank
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(cur_rank)
        device = torch.device('cuda', cur_rank)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = conf.hyper_params.gpus
        cur_rank = torch.cuda.current_device()
        conf.cur_rank = cur_rank
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return (conf, device, cur_rank)

def get_logger():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        if rank == 0:
            getLogger('AVQA').setLevel(logging.INFO)
            return getLogger('AVQA')
        else:
            getLogger('AVQA').setLevel(logging.WARNING)
            return getLogger('AVQA')
    else:
        getLogger('AVQA').setLevel(logging.INFO)
        return getLogger('AVQA')

def save_code_snapshot(folder: str, logging_path: str, file_name: str='code_snapshot.zip'):
    if folder is None:
        raise ValueError('Please specify the directories to include in the code snapshot')
    save = False
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        if rank == 0:
            save = True
    else:
        save = True
    if save:
        save_name = str(logging_path / file_name)
        with zipfile.ZipFile(save_name, 'w') as zipf:
            for (foldername, subfolders, filenames) in os.walk(folder):
                for filename in filenames:
                    if filename.endswith('.py'):
                        file_path = os.path.join(foldername, filename)
                        zipf.write(file_path, os.path.relpath(file_path, folder))
        print(f'Code snapshot saved as {save_name}')

def set_logger(cfg: dict) -> Tuple[SummaryWriter, logging.Logger, str, str]:
    warnings.filterwarnings('ignore')
    if cfg.mode == 'test':
        if cfg.output_path is not None and cfg.output_path != '':
            logging_path = Path(cfg.output_path)
            if not logging_path.exists():
                logging_path.mkdir(parents=True, exist_ok=True)
            logging_path = logging_path / (str(Path(cfg.weight).stem) + '_result.txt')
        else:
            logging_path = cfg.weight
            logging_path = logging_path.replace('.pt', '_result.txt')
        if dist.is_available() and (dist.is_initialized() and dist.get_rank() == 0 or not dist.is_initialized()):
            logger = logging.getLogger(name='AVQA')
            logger.setLevel(logging.INFO)
            file_handler = logging.FileHandler(logging_path, mode='w')
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s]-[%(filename)s line:%(lineno)d]:%(message)s ')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
    else:
        out_dir = ROOT / cfg.output_dir
        writer = None
        logger = None
        hyper_params = [f'seed{cfg.seed}']
        topk = cfg.hyper_params.model.topK
        n_experts = cfg.hyper_params.model.num_experts
        TIMESTAMP = '{0:%Y-%m-%d-%H-%M-%S}'.format(datetime.now())
        TIMESTAMP = TIMESTAMP + f"_{'_'.join(hyper_params)}"
        if not cfg.debug:
            logging_path = out_dir / TIMESTAMP
            if dist.is_available() and (dist.is_initialized() and dist.get_rank() == 0 or not dist.is_initialized()):
                logging_path.mkdir(parents=True, exist_ok=True)
                writer = SummaryWriter(logging_path.as_posix())
                logger = logging.getLogger(name='AVQA')
                logger.setLevel(logging.INFO)
                file_handler = logging.FileHandler(str(logging_path / 'log.txt'))
                console_handler = logging.StreamHandler()
                formatter = logging.Formatter('[%(asctime)s]-[%(filename)s line:%(lineno)d]:%(message)s ')
                file_handler.setFormatter(formatter)
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
                if not cfg.debug:
                    logger.addHandler(file_handler)
                save_code_snapshot('./src', logging_path, file_name='code_snapshot.zip')
        return (writer, TIMESTAMP)

def calculate_parameters(model: torch.nn.Module) -> None:
    tune_param_list = []
    tot_params = 0
    tune_params = 0
    for (name, param) in model.named_parameters():
        if param.requires_grad:
            tune_param_list.append(name)
            tune_params += param.numel()
        tot_params += param.numel()
    param_ratio = tune_params / tot_params * 100
    logger = get_logger()
    logger.info('\n-------------- parameter info --------------')
    logger.info(f'num total params: {tot_params}')
    logger.info(f'num tunable params: {tune_params}')
    logger.info(f'tunable param ratio: {param_ratio:.2f}%')
    logger.info('tunable params:')
    logger.info(json.dumps(tune_param_list, indent=4))

def logging_config(config: dict) -> None:
    os_version = f'{distro.name()} {distro.version()}'
    kernel_version = platform.platform()
    logger = get_logger()
    logger.info('\n-------------- config --------------')
    logger.info(json.dumps(config, indent=4))
    logger.info('\n-------------- environment --------------')
    logger.info(f'OS version: {os_version}')
    logger.info(f'Kernel version: {kernel_version}')
    logger.info(f'Python version: {platform.python_version()}')
    logger.info(f'torch version: {torch.__version__}')
    logger.info(f'cuda version: {torch.version.cuda}')
    logger.info(f'cuDNN version: {torch.backends.cudnn.version()}')
    logger.info(f'gpu device count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory
        mem = round(mem / 1024 ** 3, 1)
        logger.info(f'gpu device {i}: {name} - {mem}GB')
