from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import torch
from box import Box
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.models.net_qast_ehr import QASTEHR
from src.trainutils import get_dloaders, get_items
SUBTASKS = ('Audio/Counting', 'Audio/Comparative', 'Visual/Counting', 'Visual/Location', 'Audio-Visual/Existential', 'Audio-Visual/Counting', 'Audio-Visual/Location', 'Audio-Visual/Comparative', 'Audio-Visual/Temporal')
SUMMARY = ('Audio', 'Visual', 'Audio-Visual', 'Overall')

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda : stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def load_config(path: Path) -> Box:
    spec = importlib.util.spec_from_file_location(f'ablation_{uuid.uuid4().hex}', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import config: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = Box(module.config)
    cfg.mode = 'test'
    cfg.debug = False
    cfg.weight = ''
    return cfg

def metric(correct, totals, key):
    return {'accuracy': 100.0 * int(correct[key]) / int(totals[key]), 'correct': int(correct[key]), 'total': int(totals[key])}

def evaluate(name: str, config_path: Path, checkpoint: Path, training_status: str) -> dict:
    cfg = load_config(config_path)
    model = QASTEHR(**cfg.hyper_params.model)
    state = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict({str(key).removeprefix('module.'): value for (key, value) in state.items()}, strict=True)
    device = torch.device('cuda')
    model.to(device).eval()
    loader = get_dloaders(cfg)['test']
    totals = defaultdict(int)
    correct = defaultdict(int)
    with torch.inference_mode():
        for (index, sample) in enumerate(loader, start=1):
            data = get_items(sample, device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(data)
            hits = output['out'].argmax(dim=-1).eq(data['label'])
            (modalities, question_types) = sample['type']
            for (row, (modality, question_type)) in enumerate(zip(modalities, question_types)):
                modality = str(modality)
                subtask = f'{modality}/{question_type}'
                hit = int(hits[row].item())
                for group in (subtask, modality, 'Overall'):
                    totals[group] += 1
                    correct[group] += hit
            if index % 50 == 0 or index == len(loader):
                print(f'{name}: {index}/{len(loader)}', flush=True)
    result = {'name': name, 'split': 'official test split', 'selection': 'best validation checkpoint available when training was stopped; inference only', 'training_status': training_status, 'subtasks': {key: metric(correct, totals, key) for key in SUBTASKS}, 'summary': {key: metric(correct, totals, key) for key in SUMMARY}, 'av_temporal': metric(correct, totals, 'Audio-Visual/Temporal'), 'checkpoint': str(checkpoint), 'config': str(config_path), 'sha256': sha256(checkpoint), 'evaluated_at': datetime.now(timezone.utc).isoformat()}
    return result

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    completed = json.loads((HERE / 'validation_without_tacl.json').read_text(encoding='utf-8-sig'))
    tacl_checkpoint = Path(completed['checkpoint'])
    tacl_config = Path(completed['config'])
    sacl_checkpoints = sorted(Path('D:\\model\\qast_ehr_ablation_without_sacl_seed713').rglob('best.pt'), key=lambda path: path.stat().st_mtime, reverse=True)
    if not sacl_checkpoints:
        raise RuntimeError('w/o SACL best.pt not found')
    sacl_checkpoint = sacl_checkpoints[0]
    sacl_config = HERE / 'configs' / 'without_sacl_seed713.py'
    results = [evaluate('w/o TACL', tacl_config, tacl_checkpoint, 'completed 15 epochs; best validation checkpoint'), evaluate('w/o SACL', sacl_config, sacl_checkpoint, 'stopped at the start of epoch 15 after epoch 14 completed; best validation checkpoint from epoch 9')]
    payload = {'protocol': 'seed 713; official test split; frozen best validation checkpoint; no test-time training', 'generated_at': datetime.now(timezone.utc).isoformat(), 'results': results}
    (HERE / 'test_results_stopped_training.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    metric_names = [*SUBTASKS, *SUMMARY, 'AV-Temporal']

    def acc(result, key):
        if key in SUBTASKS:
            return result['subtasks'][key]['accuracy']
        if key == 'AV-Temporal':
            return result['av_temporal']['accuracy']
        return result['summary'][key]['accuracy']
    with (HERE / 'test_results_stopped_training.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', *metric_names, 'checkpoint', 'sha256', 'training_status'])
        for result in results:
            writer.writerow([result['name'], *[acc(result, key) for key in metric_names], result['checkpoint'], result['sha256'], result['training_status']])
    lines = ['# TACL/SACL 停止训练后的标准测试结果', '', '测试协议：Seed 713；使用停止时已经保存的最佳验证集 checkpoint；测试集仅执行推理，不参与训练或选模。', '', '| 模型 | Audio | Visual | Audio-Visual | AV-Temporal | Overall |', '|---|---:|---:|---:|---:|---:|']
    for result in results:
        lines.append(f"| {result['name']} | {acc(result, 'Audio'):.2f} | {acc(result, 'Visual'):.2f} | {acc(result, 'Audio-Visual'):.2f} | {acc(result, 'AV-Temporal'):.2f} | {acc(result, 'Overall'):.2f} |")
    lines.extend(['', '## 细分任务', '', '| 模型 | ' + ' | '.join(SUBTASKS) + ' |', '|---|' + '---:|' * len(SUBTASKS)])
    for result in results:
        lines.append('| ' + result['name'] + ' | ' + ' | '.join((f'{acc(result, key):.2f}' for key in SUBTASKS)) + ' |')
    lines.extend(['', '## Checkpoint', ''])
    for result in results:
        lines.extend([f"- {result['name']}: `{result['checkpoint']}`", f"  - SHA256: `{result['sha256']}`", f"  - 训练状态: {result['training_status']}"])
    (HERE / 'test_results_stopped_training_zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(HERE / 'test_results_stopped_training_zh.md')
if __name__ == '__main__':
    main()
