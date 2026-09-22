from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
import torch
from box import Box
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.models.net_qast_ehr import QASTEHR
from src.trainutils import get_dloaders, get_items

def load_config(path: Path) -> Box:
    spec = importlib.util.spec_from_file_location('ehr_validation_config', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = Box(module.config)
    cfg.mode = 'valid'
    cfg.debug = False
    cfg.weight = ''
    return cfg

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda : stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def evaluate(config_path: Path, checkpoint: Path) -> dict:
    cfg = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QASTEHR(**cfg.hyper_params.model).to(device)
    state = torch.load(checkpoint, map_location='cpu')
    state = {name.removeprefix('module.'): value for (name, value) in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    loader = get_dloaders(cfg)['valid']
    totals = defaultdict(int)
    correct = defaultdict(int)
    position_sum = 0.0
    query_share_sum = 0.0
    batches = 0
    with torch.inference_mode():
        for sample in loader:
            data = get_items(sample, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                output = model(data)
            matched = output['out'].argmax(dim=-1).eq(data['label'])
            (modalities, question_types) = sample['type']
            for (index, (modality, question_type)) in enumerate(zip(modalities, question_types)):
                modality = str(modality)
                totals[modality] += 1
                correct[modality] += int(matched[index])
                totals['overall'] += 1
                correct['overall'] += int(matched[index])
                if modality == 'Audio-Visual' and str(question_type) == 'Temporal':
                    totals['av_temporal'] += 1
                    correct['av_temporal'] += int(matched[index])
            position_sum += float(output['position_ratio'])
            query_share_sum += float(output['temporal_attention'].max(dim=-1).values.mean())
            batches += 1

    def accuracy(key: str) -> float:
        return 100.0 * correct[key] / max(totals[key], 1)
    result = {'overall': accuracy('overall'), 'audio': accuracy('Audio'), 'visual': accuracy('Visual'), 'audio_visual': accuracy('Audio-Visual'), 'av_temporal': accuracy('av_temporal'), 'position_ratio': position_sum / max(batches, 1), 'max_query_share': query_share_sum / max(batches, 1), 'eligible_for_test': False, 'checkpoint': str(checkpoint.resolve()), 'config': str(config_path.resolve()), 'sha256': sha256(checkpoint), 'counts': {key: {'correct': correct[key], 'total': totals[key]} for key in ('overall', 'Audio', 'Visual', 'Audio-Visual', 'av_temporal')}}
    result['eligible_for_test'] = bool(result['overall'] >= 78.3 and result['av_temporal'] >= 72.5 and (0.05 <= result['position_ratio'] <= 0.1) and (result['max_query_share'] <= 0.7))
    return result

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.config.resolve(), args.checkpoint.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
