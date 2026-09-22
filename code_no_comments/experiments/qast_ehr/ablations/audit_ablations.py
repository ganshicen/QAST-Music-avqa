from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.models.net_qast_ehr import QASTEHR
SWITCH_MODULES = {'enable_cmg': ('cmg', 'cmg_norm'), 'enable_scm': ('semantic_context',), 'enable_adaptive_position': ('adaptive_position',), 'enable_patch_grounder': ('patch_grounder',)}
FORBIDDEN_MODULE_NAMES = ('expert_models', 'routing_expert', 'complementary_model', 'teacher', 'ensemble')
EXPECTED_OUTPUTS = ('out', 'router_weights', 'patch_attention', 'event_gates', 'temporal_attention', 'position_ratio', 'loss_tacl', 'loss_sacl', 'loss_event', 'loss_evidence', 'loss_query_diversity', 'loss_position')

@dataclass(frozen=True)
class AuditResult:
    disabled: str | None
    errors: tuple[str, ...]

@dataclass
class _SplitIdentities:
    pairs: set[tuple[str, str]]
    videos: set[str]
    question_ids: set[str]
    videos_without_question_id: set[str]
    question_ids_without_video: set[str]
    standalone_ids: set[str]

def load_config(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location('audited_ehr_ablation', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import config: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = getattr(module, 'config', None)
    if not isinstance(config, dict):
        raise TypeError(f'config must be a dict: {path}')
    return config

def _identifiers(item: Mapping[str, object], names: Sequence[str]) -> set[str]:
    values: set[str] = set()
    for name in names:
        value = item.get(name)
        if value is not None:
            normalized = str(value).strip()
            if normalized:
                values.add(normalized)
    return values

def _split_identities(items: object, split: str) -> tuple[_SplitIdentities, list[str]]:
    if not isinstance(items, list):
        raise ValueError(f'{split} annotations must contain a JSON list')
    identities = _SplitIdentities(set(), set(), set(), set(), set(), set())
    errors: list[str] = []
    for (index, item) in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f'{split} sample {index} must be a JSON object')
        videos = _identifiers(item, ('video_id', 'video', 'vid', 'videoId'))
        question_ids = _identifiers(item, ('question_id', 'qid', 'question_idx', 'question_index', 'questionId'))
        standalone_ids = _identifiers(item, ('sample_id', 'sampleId', 'uid', 'id'))
        if len(videos) > 1:
            errors.append(f'{split} sample {index} has conflicting video aliases: {sorted(videos)}')
        if len(question_ids) > 1:
            errors.append(f'{split} sample {index} has conflicting question ID aliases: {sorted(question_ids)}')
        if len(standalone_ids) > 1:
            errors.append(f'{split} sample {index} has conflicting standalone ID aliases: {sorted(standalone_ids)}')
        identities.videos.update(videos)
        identities.question_ids.update(question_ids)
        identities.standalone_ids.update(standalone_ids)
        if videos and question_ids:
            identities.pairs.update(((video, question_id) for video in videos for question_id in question_ids))
        elif videos:
            identities.videos_without_question_id.update(videos)
        elif question_ids:
            identities.question_ids_without_video.update(question_ids)
        elif not standalone_ids:
            raise ValueError(f'{split} sample {index} has no stable sample identifier')
    return (identities, errors)

def _identity_overlap(left: _SplitIdentities, right: _SplitIdentities) -> set[tuple[str, object]]:
    overlap: set[tuple[str, object]] = {('video_question', pair) for pair in left.pairs & right.pairs}
    overlap.update((('video_without_question_id', video) for video in left.videos_without_question_id & right.videos))
    overlap.update((('video_without_question_id', video) for video in right.videos_without_question_id & left.videos))
    overlap.update((('question_id_without_video', question_id) for question_id in left.question_ids_without_video & right.question_ids))
    overlap.update((('question_id_without_video', question_id) for question_id in right.question_ids_without_video & left.question_ids))
    overlap.update((('standalone_id', standalone_id) for standalone_id in left.standalone_ids & right.standalone_ids))
    return overlap

def audit_ablation_splits(data: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    paths = {'train': Path(str(data['train_annot'])).resolve(), 'val': Path(str(data['valid_annot'])).resolve(), 'test': Path(str(data['test_annot'])).resolve()}
    if len(set(paths.values())) != 3:
        return [f'split paths are not distinct: {paths}']
    identities: dict[str, _SplitIdentities] = {}
    for (split, path) in paths.items():
        if not path.is_file():
            errors.append(f'missing {split} annotations: {path}')
            continue
        with path.open('r', encoding='utf-8') as stream:
            (split_identities, split_errors) = _split_identities(json.load(stream), split)
        identities[split] = split_identities
        errors.extend(split_errors)
    for (left_name, right_name) in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if left_name not in identities or right_name not in identities:
            continue
        overlap = _identity_overlap(identities[left_name], identities[right_name])
        if overlap:
            errors.append(f'{left_name}/{right_name} sample ID overlap: {len(overlap)}')
    return errors

def _configuration_errors(config: dict) -> tuple[str | None, list[str]]:
    errors: list[str] = []
    if config.get('seed') != 713:
        errors.append('seed must equal 713')
    if config.get('epochs') != 15:
        errors.append('epochs must equal 15')
    if config.get('weight') != '':
        errors.append('weight must be empty')
    if config.get('distill_logits') is not None:
        errors.append('distill_logits must be None')
    hyper = config.get('hyper_params')
    if not isinstance(hyper, dict):
        errors.append('hyper_params must be a dict')
        return (None, errors)
    if hyper.get('model_type') != 'QAST-EHR-2025':
        errors.append('model_type must be QAST-EHR-2025')
    model_config = hyper.get('model')
    if not isinstance(model_config, dict):
        errors.append('hyper_params.model must be a dict')
        return (None, errors)
    values = {name: model_config.get(name) for name in SWITCH_MODULES}
    invalid = [name for (name, value) in values.items() if type(value) is not bool]
    if invalid:
        errors.append(f'ablation switches must be explicit booleans: {invalid}')
    disabled_switches = [name for (name, value) in values.items() if value is False]
    disabled = disabled_switches[0] if len(disabled_switches) == 1 else None
    if len(disabled_switches) != 1 or invalid:
        errors.append(f'exactly one ablation switch must be False; found {disabled_switches}')
    forbidden_keys = [key for key in model_config if key != 'num_experts' and any((token in key.lower() for token in FORBIDDEN_MODULE_NAMES))]
    if forbidden_keys:
        errors.append(f'forbidden expert model config keys: {sorted(forbidden_keys)}')
    if model_config.get('num_experts') != 1:
        errors.append('num_experts must equal 1')
    data = config.get('data')
    if not isinstance(data, dict):
        errors.append('data must be a dict')
    else:
        try:
            errors.extend(audit_ablation_splits(data))
        except (KeyError, OSError, TypeError, ValueError) as exc:
            errors.append(f'invalid annotation splits: {exc}')
    return (disabled, errors)

def _model_structure_errors(model: nn.Module, model_config: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    module_names = [name for (name, _module) in model.named_modules()]
    for forbidden in FORBIDDEN_MODULE_NAMES:
        matches = [name for name in module_names if forbidden in name.lower()]
        if matches:
            errors.append(f'forbidden module {forbidden}: {matches}')
    answer_heads = [name for (name, module) in model.named_modules() if name.endswith('answer_head') and isinstance(module, nn.Linear)]
    if answer_heads != ['answer_head']:
        errors.append(f'expected exactly one answer_head, found {answer_heads}')
    for (switch, attributes) in SWITCH_MODULES.items():
        enabled = model_config[switch]
        for attribute in attributes:
            value = getattr(model, attribute, None)
            if enabled and value is None:
                errors.append(f'enabled {switch} module {attribute} is missing')
            if not enabled and value is not None:
                errors.append(f'disabled {switch} module {attribute} is not None')
    parameters = list(model.named_parameters())
    if not parameters:
        errors.append('model has no parameters')
    frozen = [name for (name, parameter) in parameters if not parameter.requires_grad]
    if frozen:
        errors.append(f'non-trainable parameters: {frozen}')
    non_finite = [name for (name, parameter) in parameters if not torch.isfinite(parameter.detach()).all()]
    if non_finite:
        errors.append(f'non-finite parameters: {non_finite}')
    return errors

def _synthetic_batch(model_config: Mapping[str, object]) -> dict[str, torch.Tensor]:
    batch_size = 2
    steps = 6
    patch_count = max(5, int(model_config.get('keep_patches', 1)))
    num_labels = int(model_config.get('num_labels', 42))
    return {'audio': torch.randn(batch_size, steps, int(model_config['audio_dim'])), 'video': torch.randn(batch_size, steps, int(model_config['video_dim'])), 'patch': torch.randn(batch_size, steps, patch_count, int(model_config['patch_dim'])), 'quest': torch.randn(batch_size, 1, int(model_config['question_dim'])), 'prompt': torch.randn(batch_size, 1, int(model_config['question_dim'])), 'label': torch.randint(0, num_labels, (batch_size,))}

def _output_shapes(model_config: Mapping[str, object], data: Mapping[str, torch.Tensor]) -> dict[str, tuple[int, ...]]:
    (batch_size, steps) = data['audio'].shape[:2]
    patch_count = data['patch'].size(2)
    return {'out': (batch_size, int(model_config.get('num_labels', 42))), 'router_weights': (batch_size, 4), 'patch_attention': (batch_size, steps, patch_count), 'event_gates': (batch_size, len(model_config['event_scales'])), 'temporal_attention': (batch_size, int(model_config['num_temporal_queries']), steps), 'position_ratio': (), 'loss_tacl': (), 'loss_sacl': (), 'loss_event': (), 'loss_evidence': (), 'loss_query_diversity': (), 'loss_position': ()}

def _forward_backward_errors(model: nn.Module, model_config: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    model.train()
    model.zero_grad(set_to_none=True)
    if hasattr(model, 'set_training_epoch'):
        model.set_training_epoch(max(1, int(model_config.get('aux_warmup_epochs', 3))))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(713)
        data = _synthetic_batch(model_config)
        output = model(data)
    if not isinstance(output, dict):
        return [f'forward output must be a dict, found {type(output).__name__}']
    actual_keys = set(output)
    expected_keys = set(EXPECTED_OUTPUTS)
    if actual_keys != expected_keys:
        errors.append(f'invalid output keys: missing={sorted(expected_keys - actual_keys)}, extra={sorted(actual_keys - expected_keys)}')
    expected_shapes = _output_shapes(model_config, data)
    for name in EXPECTED_OUTPUTS:
        value = output.get(name)
        if not isinstance(value, torch.Tensor):
            errors.append(f'output {name} must be a Tensor')
            continue
        actual_shape = tuple(value.shape)
        if actual_shape != expected_shapes[name]:
            errors.append(f'output {name} has shape {actual_shape}, expected {expected_shapes[name]}')
        if not torch.isfinite(value.detach()).all():
            errors.append(f'output {name} contains non-finite values')
    if errors:
        return errors
    loss = F.cross_entropy(output['out'], data['label'])
    loss = loss + sum((output[name] for name in EXPECTED_OUTPUTS if name.startswith('loss_')))
    if not torch.isfinite(loss.detach()):
        return ['synthetic loss is non-finite']
    loss.backward()
    missing_gradients = [name for (name, parameter) in model.named_parameters() if parameter.requires_grad and parameter.grad is None]
    if missing_gradients:
        errors.append(f'trainable parameters without gradients: {missing_gradients}')
    non_finite_gradients = [name for (name, parameter) in model.named_parameters() if parameter.requires_grad and parameter.grad is not None and (not torch.isfinite(parameter.grad).all())]
    if non_finite_gradients:
        errors.append(f'non-finite gradients: {non_finite_gradients}')
    return errors

def audit_ablation(config: dict, model: nn.Module | None=None) -> AuditResult:
    (disabled, errors) = _configuration_errors(config)
    if errors:
        return AuditResult(disabled=disabled, errors=tuple(errors))
    model_config = config['hyper_params']['model']
    audited_model = QASTEHR(**model_config) if model is None else model
    errors.extend(_model_structure_errors(audited_model, model_config))
    errors.extend(_forward_backward_errors(audited_model, model_config))
    return AuditResult(disabled=disabled, errors=tuple(errors))

def main(argv: Sequence[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description='Audit a QAST-EHR ablation config')
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config.resolve())
        result = audit_ablation(config)
    except Exception as exc:
        print('AUDIT FAILED')
        print(f'- {type(exc).__name__}: {exc}')
        return 1
    if result.errors:
        print('AUDIT FAILED')
        for error in result.errors:
            print(f'- {error}')
        return 1
    print('AUDIT PASSED')
    print(f'disabled: {result.disabled}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
