from __future__ import annotations
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
import torch
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.qast_ehr.ablations.evaluate_ablation_tests import load_config
from src.models.net_qast_ehr import QASTEHR
from src.trainutils import get_dloaders, get_items
ABLATION_DIR = Path(__file__).resolve().parent
MANIFEST = ABLATION_DIR / 'ablation_validation_manifest.json'
FULL_RESULT = ROOT / 'experiments/qast_ehr/final_test_results.json'
JSON_OUTPUT = ABLATION_DIR / 'ablation_test_subtask_results.json'
CSV_OUTPUT = ABLATION_DIR / 'ablation_test_subtask_results.csv'
ORDER = ('full', 'without_cmg', 'without_scm', 'without_adaptive_position', 'without_patch_grounder')
SUBTASKS = ('Audio/Counting', 'Audio/Comparative', 'Visual/Counting', 'Visual/Location', 'Audio-Visual/Existential', 'Audio-Visual/Counting', 'Audio-Visual/Location', 'Audio-Visual/Comparative', 'Audio-Visual/Temporal')
SUMMARY_KEYS = ('Audio', 'Visual', 'Audio-Visual', 'Overall')

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda : stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def load_entries() -> list[dict[str, str]]:
    full = json.loads(FULL_RESULT.read_text(encoding='utf-8-sig'))
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8-sig'))
    raw = {item['name']: item for item in manifest['results']}
    entries = [{'name': 'full', 'config': full['config'], 'checkpoint': full['checkpoint'], 'sha256': full['sha256']}]
    for name in ORDER[1:]:
        item = raw[name]
        entries.append({'name': name, 'config': item['config'], 'checkpoint': item['checkpoint'], 'sha256': item['sha256']})
    for entry in entries:
        checkpoint = Path(entry['checkpoint'])
        if not checkpoint.is_file():
            raise RuntimeError(f'checkpoint missing: {checkpoint}')
        if sha256(checkpoint) != entry['sha256']:
            raise RuntimeError(f'checkpoint hash mismatch: {checkpoint}')
    return entries

def evaluate(entry: Mapping[str, str]) -> dict:
    cfg = load_config(Path(entry['config']))
    model = QASTEHR(**cfg.hyper_params.model)
    state = torch.load(entry['checkpoint'], map_location='cpu')
    if not isinstance(state, Mapping):
        raise RuntimeError('checkpoint state must be a mapping')
    model.load_state_dict({str(key).removeprefix('module.'): value for (key, value) in state.items()}, strict=True)
    device = torch.device('cuda')
    model.to(device).eval()
    loader = get_dloaders(cfg)['test']
    totals: defaultdict[str, int] = defaultdict(int)
    correct: defaultdict[str, int] = defaultdict(int)
    with torch.inference_mode():
        for (index, sample) in enumerate(loader, start=1):
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
            if index % 50 == 0 or index == len(loader):
                print(f"{entry['name']}: {index}/{len(loader)}", flush=True)

    def metric(key: str) -> dict[str, int | float]:
        return {'accuracy': 100.0 * correct[key] / totals[key], 'correct': correct[key], 'total': totals[key]}
    return {'model': entry['name'], 'subtasks': {key: metric(key) for key in SUBTASKS}, 'summary': {key: metric(key) for key in SUMMARY_KEYS}, 'checkpoint': entry['checkpoint'], 'sha256': entry['sha256']}

def save(results: list[dict]) -> None:
    payload = {'split': 'official test split', 'selection': 'best validation checkpoint; inference only', 'generated_at': datetime.now(timezone.utc).isoformat(), 'results': results}
    temporary = JSON_OUTPUT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(JSON_OUTPUT)
    temporary = CSV_OUTPUT.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        columns = ['model', *SUBTASKS, *SUMMARY_KEYS, 'checkpoint', 'sha256']
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in results:
            row: dict[str, object] = {'model': result['model']}
            row.update({key: result['subtasks'][key]['accuracy'] for key in SUBTASKS})
            row.update({key: result['summary'][key]['accuracy'] for key in SUMMARY_KEYS})
            row['checkpoint'] = result['checkpoint']
            row['sha256'] = result['sha256']
            writer.writerow(row)
    temporary.replace(CSV_OUTPUT)

def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    results: list[dict] = []
    for entry in load_entries():
        print(f"Evaluating {entry['name']}", flush=True)
        results.append(evaluate(entry))
        save(results)
    print(JSON_OUTPUT, flush=True)
    print(CSV_OUTPUT, flush=True)
if __name__ == '__main__':
    main()
