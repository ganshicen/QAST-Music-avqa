from __future__ import annotations
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.qast_ehr.ablations.evaluate_test_subtasks import SUBTASKS, SUMMARY_KEYS, evaluate, sha256
HERE = Path(__file__).resolve().parent
OLD_MANIFEST = ROOT / 'experiments/qast_ehr/ablations/ablation_validation_manifest.json'
NEW_MANIFEST = HERE / 'validation_manifest.json'
JSON_OUTPUT = HERE / 'test_subtask_results.json'
CSV_OUTPUT = HERE / 'test_subtask_results.csv'
FLAG = HERE / 'test_evaluation_consumed.flag'

def load_entries() -> list[dict[str, str]]:
    old = json.loads(OLD_MANIFEST.read_text(encoding='utf-8-sig'))
    new = json.loads(NEW_MANIFEST.read_text(encoding='utf-8-sig'))
    baseline = next((item for item in old['results'] if item['name'] == 'without_scm'))
    entries = [{'name': 'without_scm_baseline', 'config': baseline['config'], 'checkpoint': baseline['checkpoint'], 'sha256': baseline['sha256']}]
    for item in new['results']:
        entries.append({'name': item['name'], 'config': item['config'], 'checkpoint': item['checkpoint'], 'sha256': item['sha256']})
    for entry in entries:
        checkpoint = Path(entry['checkpoint'])
        if sha256(checkpoint) != entry['sha256']:
            raise RuntimeError(f'checkpoint hash mismatch: {checkpoint}')
    return entries

def save(results: list[dict]) -> None:
    payload = {'split': 'official test split', 'selection': 'best validation checkpoint; inference only', 'baseline': 'without_scm_baseline', 'generated_at': datetime.now(timezone.utc).isoformat(), 'results': results}
    temporary = JSON_OUTPUT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(JSON_OUTPUT)
    temporary = CSV_OUTPUT.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        columns = ['model', *SUBTASKS, *SUMMARY_KEYS, 'delta_overall', 'checkpoint', 'sha256']
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        baseline = results[0]['summary']['Overall']['accuracy']
        for result in results:
            row: dict[str, object] = {'model': result['model']}
            row.update({key: result['subtasks'][key]['accuracy'] for key in SUBTASKS})
            row.update({key: result['summary'][key]['accuracy'] for key in SUMMARY_KEYS})
            row['delta_overall'] = result['summary']['Overall']['accuracy'] - baseline
            row['checkpoint'] = result['checkpoint']
            row['sha256'] = result['sha256']
            writer.writerow(row)
    temporary.replace(CSV_OUTPUT)

def main() -> None:
    if FLAG.exists():
        raise RuntimeError('SCM-baseline test evaluation already consumed')
    results: list[dict] = []
    for entry in load_entries():
        print(f"Evaluating {entry['name']}", flush=True)
        result = evaluate(entry)
        results.append(result)
        save(results)
    FLAG.write_text(datetime.now(timezone.utc).isoformat() + '\n', encoding='utf-8')
    print(JSON_OUTPUT, flush=True)
    print(CSV_OUTPUT, flush=True)
if __name__ == '__main__':
    main()
