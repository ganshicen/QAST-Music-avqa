from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
import torch
from box import Box
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
VALIDATION_MANIFEST = HERE / 'validation_manifest.json'
SEED713_RESULTS = ROOT / 'experiments/qast_ehr/ablations/ablation_test_subtask_results.json'
SUBTASKS = ('Audio/Counting', 'Audio/Comparative', 'Visual/Counting', 'Visual/Location', 'Audio-Visual/Existential', 'Audio-Visual/Counting', 'Audio-Visual/Location', 'Audio-Visual/Comparative', 'Audio-Visual/Temporal')
SUMMARY_KEYS = ('Audio', 'Visual', 'Audio-Visual', 'Overall')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.models.net_qast_ehr import QASTEHR
from src.trainutils import get_dloaders, get_items

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda : stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def load_config(path: Path) -> Box:
    name = f'without_scm_test_{uuid.uuid4().hex}'
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import config: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = Box(module.config)
    cfg.mode = 'test'
    cfg.debug = False
    cfg.weight = ''
    return cfg

def load_validation_entries() -> dict[int, dict]:
    manifest = json.loads(VALIDATION_MANIFEST.read_text(encoding='utf-8-sig'))
    entries = {int(item['seed']): item for item in manifest['results']}
    if set(entries) != {713, 123, 456}:
        raise RuntimeError('validation manifest must contain seeds 713, 123, and 456')
    for (seed, entry) in entries.items():
        checkpoint = Path(entry['checkpoint'])
        config = Path(entry['config'])
        if not checkpoint.is_file() or not config.is_file():
            raise RuntimeError(f'seed {seed} checkpoint/config missing')
        actual = sha256(checkpoint)
        if actual.lower() != str(entry['sha256']).lower():
            raise RuntimeError(f'seed {seed} checkpoint SHA256 mismatch')
        log = HERE / 'logs' / f'seed{seed}_train.log'
        if seed in {123, 456}:
            text = log.read_text(encoding='utf-8-sig', errors='replace')
            if 'training epoch 15' not in text or 'Epoch 15 done' not in text:
                raise RuntimeError(f'seed {seed} epoch-15 evidence missing')
    return entries

def metric(correct: Mapping[str, int], totals: Mapping[str, int], key: str) -> dict:
    total = int(totals[key])
    hits = int(correct[key])
    return {'accuracy': 100.0 * hits / total, 'correct': hits, 'total': total}

def evaluate(seed: int, entry: Mapping[str, object]) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    config_path = Path(str(entry['config'])).resolve()
    checkpoint = Path(str(entry['checkpoint'])).resolve()
    cfg = load_config(config_path)
    model = QASTEHR(**cfg.hyper_params.model)
    state = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict({str(key).removeprefix('module.'): value for (key, value) in state.items()}, strict=True)
    device = torch.device('cuda')
    model.to(device).eval()
    loader = get_dloaders(cfg)['test']
    totals: defaultdict[str, int] = defaultdict(int)
    correct: defaultdict[str, int] = defaultdict(int)
    with torch.inference_mode():
        for (batch_index, sample) in enumerate(loader, start=1):
            data = get_items(sample, device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(data)
            hits = output['out'].argmax(dim=-1).eq(data['label'])
            (modalities, question_types) = sample['type']
            for (row, (modality, question_type)) in enumerate(zip(modalities, question_types)):
                modality = str(modality)
                key = f'{modality}/{question_type}'
                hit = int(hits[row].item())
                for group in (key, modality, 'Overall'):
                    totals[group] += 1
                    correct[group] += hit
            if batch_index % 50 == 0 or batch_index == len(loader):
                print(f'seed {seed}: {batch_index}/{len(loader)}', flush=True)
    result = {'seed': seed, 'split': 'official test split', 'selection': 'frozen best validation checkpoint; inference only', 'subtasks': {key: metric(correct, totals, key) for key in SUBTASKS}, 'summary': {key: metric(correct, totals, key) for key in SUMMARY_KEYS}, 'av_temporal': metric(correct, totals, 'Audio-Visual/Temporal'), 'checkpoint': str(checkpoint), 'config': str(config_path), 'sha256': sha256(checkpoint), 'evaluated_at': datetime.now(timezone.utc).isoformat()}
    return result

def seed713_existing(entry: Mapping[str, object]) -> dict:
    payload = json.loads(SEED713_RESULTS.read_text(encoding='utf-8-sig'))
    row = next((item for item in payload['results'] if item['model'] == 'without_scm'))
    checkpoint = Path(row['checkpoint']).resolve()
    if str(checkpoint) != str(Path(str(entry['checkpoint'])).resolve()):
        raise RuntimeError('seed 713 existing test checkpoint does not match validation manifest')
    if sha256(checkpoint).lower() != str(entry['sha256']).lower():
        raise RuntimeError('seed 713 existing test SHA256 does not match validation manifest')
    return {'seed': 713, 'split': 'official test split', 'selection': 'reused existing frozen best validation checkpoint result', 'subtasks': row['subtasks'], 'summary': row['summary'], 'av_temporal': row['subtasks']['Audio-Visual/Temporal'], 'checkpoint': row['checkpoint'], 'config': str(entry['config']), 'sha256': row['sha256'], 'evaluated_at': payload.get('generated_at')}

def load_or_evaluate(seed: int, entry: Mapping[str, object]) -> dict:
    output = HERE / f'test_seed{seed}.json'
    flag = HERE / f'test_seed{seed}_consumed.flag'
    if output.is_file() and flag.is_file():
        result = json.loads(output.read_text(encoding='utf-8-sig'))
        if result['sha256'].lower() != str(entry['sha256']).lower():
            raise RuntimeError(f'seed {seed} saved test result SHA256 mismatch')
        return result
    if output.exists() or flag.exists():
        raise RuntimeError(f'seed {seed} has incomplete one-time test artifacts')
    result = evaluate(seed, entry)
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(output)
    flag.write_text(f"evaluated_at={result['evaluated_at']}\nsha256={result['sha256']}\n", encoding='utf-8')
    return result

def sample_stats(values: list[float]) -> dict:
    return {'mean': statistics.mean(values), 'sample_std': statistics.stdev(values), 'formatted': f'{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}'}

def save_summary(results: list[dict]) -> None:
    metric_names = [*SUBTASKS, *SUMMARY_KEYS, 'AV-Temporal']

    def accuracy(result: Mapping[str, object], name: str) -> float:
        if name in SUBTASKS:
            return float(result['subtasks'][name]['accuracy'])
        if name == 'AV-Temporal':
            return float(result['av_temporal']['accuracy'])
        return float(result['summary'][name]['accuracy'])
    stats = {name: sample_stats([accuracy(result, name) for result in results]) for name in metric_names}
    payload = {'protocol': 'QAST-EHR; random initialization; 15 epochs; best validation checkpoint; standard train/val/test', 'generated_at': datetime.now(timezone.utc).isoformat(), 'results': results, 'mean_sample_std': stats}
    (HERE / 'multiseed_test_summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (HERE / 'multiseed_test_summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['seed', *metric_names, 'checkpoint', 'sha256'])
        for result in results:
            writer.writerow([result['seed'], *[accuracy(result, name) for name in metric_names], result['checkpoint'], result['sha256']])
        writer.writerow(['mean', *[stats[name]['mean'] for name in metric_names], '', ''])
        writer.writerow(['sample_std', *[stats[name]['sample_std'] for name in metric_names], '', ''])
    lines = ['# QAST-EHR 三随机种子标准测试汇总', '', '协议：Seed 713/123/456，随机初始化，完整训练 15 epochs，按验证集选择最佳 checkpoint；测试集仅用于最终一次推理评估。', '', '| 指标 | Seed 713 | Seed 123 | Seed 456 | 均值 ± 样本标准差 |', '|---|---:|---:|---:|---:|']
    for name in metric_names:
        values = [accuracy(result, name) for result in results]
        lines.append(f"| {name} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} | {stats[name]['formatted']} |")
    lines.extend(['', '## Checkpoint 证据', ''])
    for result in results:
        lines.extend([f"- Seed {result['seed']}: `{result['checkpoint']}`", f"  - SHA256: `{result['sha256']}`"])
    (HERE / 'multiseed_test_summary_zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def main() -> None:
    entries = load_validation_entries()
    results = [seed713_existing(entries[713])]
    results.append(load_or_evaluate(123, entries[123]))
    results.append(load_or_evaluate(456, entries[456]))
    save_summary(results)
    print(HERE / 'multiseed_test_summary_zh.md')
if __name__ == '__main__':
    main()
