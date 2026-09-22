from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import torch
from box import Box
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.qast_ehr.audit_protocol import audit_config, audit_model
from experiments.qast_ehr.evaluate_validation import sha256
from src.models.net_qast_ehr import QASTEHR
from src.trainutils import get_dloaders, get_items

def load_config(path: Path) -> Box:
    spec = importlib.util.spec_from_file_location('ehr_final_config', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = Box(module.config)
    cfg.mode = 'test'
    cfg.debug = False
    cfg.weight = ''
    return cfg

def require_manifest(manifest_path: Path, flag_path: Path) -> tuple[Path, Path]:
    if not manifest_path.is_file():
        raise RuntimeError('validation selection manifest required')
    if flag_path.exists():
        raise RuntimeError('final test has already been consumed')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    selected = manifest.get('selected', {})
    if not selected.get('eligible_for_test', False):
        raise RuntimeError('selected validation candidate is not eligible for test')
    checkpoint = Path(selected['checkpoint']).resolve()
    config = Path(selected['config']).resolve()
    if not checkpoint.is_file():
        raise RuntimeError(f'selected checkpoint is missing: {checkpoint}')
    if sha256(checkpoint) != selected.get('sha256'):
        raise RuntimeError('selected checkpoint SHA-256 does not match manifest')
    return (config, checkpoint)

def evaluate(config_path: Path, checkpoint: Path) -> dict:
    cfg = load_config(config_path)
    errors = audit_config(cfg.to_dict())
    model = QASTEHR(**cfg.hyper_params.model)
    errors.extend(audit_model(model))
    if errors:
        raise RuntimeError('audit failed: ' + '; '.join(errors))
    state = torch.load(checkpoint, map_location='cpu')
    state = {name.removeprefix('module.'): value for (name, value) in state.items()}
    model.load_state_dict(state, strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    loader = get_dloaders(cfg)['test']
    totals = defaultdict(int)
    correct = defaultdict(int)
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

    def metric(key: str) -> dict:
        total = totals[key]
        hits = correct[key]
        return {'accuracy': 100.0 * hits / max(total, 1), 'correct': hits, 'total': total}
    return {'audio': metric('Audio'), 'visual': metric('Visual'), 'audio_visual': metric('Audio-Visual'), 'overall': metric('overall'), 'av_temporal': metric('av_temporal'), 'checkpoint': str(checkpoint), 'config': str(config_path), 'sha256': sha256(checkpoint)}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    flag = output_dir / 'final_test_consumed.flag'
    (config, checkpoint) = require_manifest(args.manifest.resolve(), flag)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate(config, checkpoint)
    result['evaluated_at'] = datetime.now(timezone.utc).isoformat()
    (output_dir / 'final_test_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output_dir / 'final_test_results.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(('metric', 'accuracy', 'correct', 'total'))
        for name in ('audio', 'visual', 'audio_visual', 'overall', 'av_temporal'):
            value = result[name]
            writer.writerow((name, value['accuracy'], value['correct'], value['total']))
    flag.write_text(f"evaluated_at={result['evaluated_at']}\nsha256={result['sha256']}\n", encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
