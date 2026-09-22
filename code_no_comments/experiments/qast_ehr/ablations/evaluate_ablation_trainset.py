from __future__ import annotations
import csv
import io
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.qast_ehr.ablations.evaluate_ablation_tests import EXPECTED_ORDER, load_config, require_manifest, sha256
from src.dataset import AVQA_dataset
from src.models.net_qast_ehr import QASTEHR
from src.trainutils import get_items
ABLATION_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = ABLATION_DIR / 'ablation_validation_manifest.json'
TEST_FLAG_PATH = ABLATION_DIR / 'ablation_tests_consumed.flag'
JSON_PATH = ABLATION_DIR / 'ablation_trainset_results.json'
CSV_PATH = ABLATION_DIR / 'ablation_trainset_results.csv'

def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        stream.write(text)
    temporary.replace(path)

def evaluate(config_path: Path, checkpoint: Path) -> dict:
    cfg = load_config(config_path)
    cfg.mode = 'train'
    cfg.debug = False
    cfg.weight = ''
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for BF16 training-set evaluation')
    dataset = AVQA_dataset(cfg, mode='train')
    loader = DataLoader(dataset, batch_size=cfg.data.eval_batch_size, shuffle=False, num_workers=cfg.data.num_workers, pin_memory=True)
    model = QASTEHR(**cfg.hyper_params.model)
    state = torch.load(checkpoint, map_location='cpu')
    if not isinstance(state, Mapping):
        raise RuntimeError(f'checkpoint state must be a mapping: {checkpoint}')
    model.load_state_dict({str(name).removeprefix('module.'): value for (name, value) in state.items()}, strict=True)
    device = torch.device('cuda')
    model.to(device).eval()
    totals: defaultdict[str, int] = defaultdict(int)
    correct: defaultdict[str, int] = defaultdict(int)
    with torch.inference_mode():
        for (batch_index, sample) in enumerate(loader, start=1):
            data = get_items(sample, device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(data)
            matched = output['out'].argmax(dim=-1).eq(data['label'])
            (modalities, question_types) = sample['type']
            for (index, (modality, question_type)) in enumerate(zip(modalities, question_types)):
                modality = str(modality)
                hit = int(matched[index].item())
                totals[modality] += 1
                correct[modality] += hit
                totals['overall'] += 1
                correct['overall'] += hit
                if modality == 'Audio-Visual' and str(question_type) == 'Temporal':
                    totals['av_temporal'] += 1
                    correct['av_temporal'] += hit
            if batch_index % 100 == 0 or batch_index == len(loader):
                print(f'{checkpoint.parent.name}: {batch_index}/{len(loader)}', flush=True)

    def metric(key: str) -> dict[str, int | float]:
        total = totals[key]
        hits = correct[key]
        return {'accuracy': 100.0 * hits / max(total, 1), 'correct': hits, 'total': total}
    return {'audio': metric('Audio'), 'visual': metric('Visual'), 'audio_visual': metric('Audio-Visual'), 'overall': metric('overall'), 'av_temporal': metric('av_temporal'), 'checkpoint': str(checkpoint.resolve()), 'config': str(config_path.resolve()), 'sha256': sha256(checkpoint)}

def save(results: list[dict]) -> None:
    payload = {'split': 'official training split', 'selection': 'best validation checkpoint for each ablation', 'generated_at': datetime.now(timezone.utc).isoformat(), 'results': results}
    atomic_write(JSON_PATH, json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['model', 'audio', 'visual', 'audio_visual', 'overall', 'av_temporal', 'checkpoint', 'sha256'])
    for item in results:
        writer.writerow([item['model'], item['audio']['accuracy'], item['visual']['accuracy'], item['audio_visual']['accuracy'], item['overall']['accuracy'], item['av_temporal']['accuracy'], item['checkpoint'], item['sha256']])
    atomic_write(CSV_PATH, stream.getvalue())

def main() -> None:
    entries = require_manifest(MANIFEST_PATH, TEST_FLAG_PATH)
    by_name = {entry.name: entry for entry in entries}
    results: list[dict] = []
    for name in EXPECTED_ORDER:
        entry = by_name[name]
        print(f'Evaluating {name} on the official training split', flush=True)
        result = evaluate(entry.config, entry.checkpoint)
        result['model'] = name
        results.append(result)
        save(results)
    print(JSON_PATH, flush=True)
    print(CSV_PATH, flush=True)
if __name__ == '__main__':
    main()
