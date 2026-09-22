from __future__ import print_function
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
sys.path.append(ROOT.as_posix())
import torch
import torch.nn as nn
import torch.distributed as dist
from src.utils import arg_parse, seed_everything, setting, get_logger, set_logger, logging_config
from src.trainutils import get_model, get_dloaders, test, sync_processes

def main():
    args = arg_parse()
    args.mode = 'test'
    (cfg, device, cur_rank) = setting(args)
    set_logger(cfg)
    logger = get_logger()
    logging_config(cfg)
    seed_everything(cfg.seed)
    d_loaders = get_dloaders(cfg)
    model = get_model(cfg, device)
    sync_processes()
    if isinstance(cfg.data.test_annots, (list, tuple)):
        for (idx, test_annot) in enumerate(cfg.data.test_annots):
            cfg.data.test_annot = test_annot
            d_loaders = get_dloaders(cfg)['test']
            logger.info(f'\n-------------- evaluating test dataset {cfg.data.test_annot} --------------')
            test(cfg, device, d_loaders, model)
    else:
        logger.info(f'\n-------------- evaluating test dataset {cfg.data.test_annot} --------------')
        test(cfg, device, d_loaders['test'], model)
    if dist.is_initialized():
        dist.destroy_process_group()
if __name__ == '__main__':
    main()
