from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import List
import torch
import torch.nn as nn
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.models.net_qast_ehr import QASTEHR
FORBIDDEN_MODULES = ('routing_expert', 'complementary_model')
FORBIDDEN_CONFIG_KEYS = {'source_checkpoint', 'complementary_checkpoint', 'routing_expert_checkpoint', 'recent_mix', 'primary_mix'}

def audit_model(model: nn.Module) -> List[str]:
    errors: List[str] = []
    module_names = [name for (name, _) in model.named_modules()]
    for forbidden in FORBIDDEN_MODULES:
        matches = [name for name in module_names if forbidden in name]
        if matches:
            errors.append(f'forbidden module {forbidden}: {matches}')
    answer_heads = [name for (name, module) in model.named_modules() if name.endswith('answer_head') and isinstance(module, nn.Linear)]
    if answer_heads != ['answer_head']:
        errors.append(f'expected exactly one answer_head, found {answer_heads}')
    frozen = [name for (name, parameter) in model.named_parameters() if not parameter.requires_grad]
    if frozen:
        errors.append(f'frozen task parameters: {frozen}')
    non_finite = [name for (name, parameter) in model.named_parameters() if not torch.isfinite(parameter.detach()).all()]
    if non_finite:
        errors.append(f'non-finite parameters: {non_finite}')
    return errors

def load_config(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location('audited_ehr_config', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import config: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config

def sample_key(item: dict) -> tuple:
    video = item.get('video_id', item.get('video', item.get('vid', '')))
    question_id = item.get('question_id', item.get('qid', ''))
    return (str(video), str(question_id), str(item.get('question', '')))

def audit_splits(data: dict) -> List[str]:
    errors: List[str] = []
    paths = {'train': Path(data['train_annot']).resolve(), 'val': Path(data['valid_annot']).resolve(), 'test': Path(data['test_annot']).resolve()}
    if len(set(paths.values())) != 3:
        errors.append(f'split paths are not distinct: {paths}')
        return errors
    keys = {}
    for (name, path) in paths.items():
        if not path.is_file():
            errors.append(f'missing {name} annotations: {path}')
            continue
        with path.open('r', encoding='utf-8') as stream:
            keys[name] = {sample_key(item) for item in json.load(stream)}
    for (left, right) in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if left in keys and right in keys:
            overlap = keys[left] & keys[right]
            if overlap:
                errors.append(f'{left}/{right} sample overlap: {len(overlap)}')
    return errors

def audit_config(config: dict) -> List[str]:
    errors: List[str] = []
    if config.get('epochs') != 15:
        errors.append('epochs must equal 15')
    if config.get('weight', '') != '':
        errors.append('task checkpoint weight must be empty')
    if config.get('distill_logits') is not None:
        errors.append('distill_logits must be None')
    hyper = config['hyper_params']
    if hyper.get('model_type') != 'QAST-EHR-2025':
        errors.append('model_type must be QAST-EHR-2025')
    model_config = hyper['model']
    present = FORBIDDEN_CONFIG_KEYS & set(model_config)
    if present:
        errors.append(f'forbidden model config keys: {sorted(present)}')
    if model_config.get('num_experts') != 1:
        errors.append('num_experts metadata must equal 1')
    errors.extend(audit_splits(config['data']))
    return errors

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    model = QASTEHR(**config['hyper_params']['model'])
    errors = audit_config(config) + audit_model(model)
    if errors:
        print('AUDIT FAILED')
        for error in errors:
            print(f'- {error}')
        return 1
    print('AUDIT PASSED')
    print('random-init task model; one answer head; no expert or task checkpoint')
    print('15 epochs; official disjoint train/validation/test splits')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
