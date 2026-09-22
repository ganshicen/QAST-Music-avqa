from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence
import torch
from box import Box
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.qast_ehr.ablations.audit_ablations import audit_ablation, load_config as load_audit_config
from src.models.net_qast_ehr import QASTEHR
from src.trainutils import get_dloaders, get_items
EXPECTED_ORDER = ('without_cmg', 'without_scm', 'without_adaptive_position', 'without_patch_grounder')
EXPECTED_DISABLED_SWITCH = {'without_cmg': 'enable_cmg', 'without_scm': 'enable_scm', 'without_adaptive_position': 'enable_adaptive_position', 'without_patch_grounder': 'enable_patch_grounder'}
BASELINE = {'overall': 77.02924745317121, 'av_temporal': 69.4647201946472}
CSV_COLUMNS = ('model', 'audio', 'visual', 'audio_visual', 'overall', 'av_temporal', 'delta_overall', 'delta_av_temporal', 'checkpoint', 'sha256')

@dataclass(frozen=True)
class ManifestEntry:
    name: str
    config: Path
    checkpoint: Path
    sha256: str

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda : stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _manifest_path(value: object, manifest_path: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f'manifest {field} must be a non-empty path')
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()

def _validate_manifest_shape(manifest: object) -> list[Mapping[str, object]]:
    if not isinstance(manifest, dict):
        raise RuntimeError('ablation validation manifest must be a JSON object')
    order = manifest.get('order')
    if order != list(EXPECTED_ORDER):
        raise RuntimeError('manifest order must exactly equal: ' + ', '.join(EXPECTED_ORDER))
    results = manifest.get('results')
    if not isinstance(results, list):
        raise RuntimeError('manifest results must be a list')
    if len(results) != len(EXPECTED_ORDER):
        raise RuntimeError('manifest results must contain exactly four entries')
    if not all((isinstance(entry, dict) for entry in results)):
        raise RuntimeError('every manifest result must be a JSON object')
    names = [entry.get('name') for entry in results]
    if len(set(names)) != len(names):
        raise RuntimeError('manifest result names must not contain duplicates')
    if set(names) != set(EXPECTED_ORDER):
        raise RuntimeError('manifest results must contain each predetermined ablation exactly once')
    return results

def require_manifest(manifest_path: Path, flag_path: Path) -> list[ManifestEntry]:
    manifest_path = manifest_path.resolve()
    flag_path = flag_path.resolve()
    if flag_path.exists():
        raise RuntimeError('ablation tests have already been consumed')
    if not manifest_path.is_file():
        raise RuntimeError(f'ablation validation manifest is missing: {manifest_path}')
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'cannot read ablation validation manifest: {exc}') from exc
    raw_results = _validate_manifest_shape(manifest)
    by_name = {str(entry['name']): entry for entry in raw_results}
    entries: list[ManifestEntry] = []
    for name in EXPECTED_ORDER:
        raw = by_name[name]
        config = _manifest_path(raw.get('config'), manifest_path, f'{name}.config')
        checkpoint = _manifest_path(raw.get('checkpoint'), manifest_path, f'{name}.checkpoint')
        expected_hash = raw.get('sha256')
        if not config.is_file():
            raise RuntimeError(f'ablation config is missing for {name}: {config}')
        if not checkpoint.is_file():
            raise RuntimeError(f'ablation checkpoint is missing for {name}: {checkpoint}')
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise RuntimeError(f'invalid checkpoint SHA-256 for {name}')
        actual_hash = sha256(checkpoint)
        if actual_hash.lower() != expected_hash.lower():
            raise RuntimeError(f'checkpoint SHA-256 does not match for {name}')
        entries.append(ManifestEntry(name=name, config=config, checkpoint=checkpoint, sha256=actual_hash))
    for entry in entries:
        try:
            config = load_audit_config(entry.config)
            audit = audit_ablation(config)
        except Exception as exc:
            raise RuntimeError(f'ablation audit failed for {entry.name}: {exc}') from exc
        if audit.errors:
            raise RuntimeError(f'ablation audit failed for {entry.name}: ' + '; '.join(audit.errors))
        expected_disabled = EXPECTED_DISABLED_SWITCH[entry.name]
        if audit.disabled != expected_disabled:
            raise RuntimeError(f'{entry.name} must disable only {expected_disabled}; audit found {audit.disabled}')
    return entries

def load_config(path: Path) -> Box:
    module_name = f'ehr_ablation_test_{uuid.uuid4().hex}'
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import config: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = getattr(module, 'config', None)
    if not isinstance(config, dict):
        raise RuntimeError(f'config must be a dict: {path}')
    cfg = Box(config)
    cfg.mode = 'test'
    cfg.debug = False
    cfg.weight = ''
    return cfg

def evaluate(config_path: Path, checkpoint: Path) -> dict:
    cfg = load_config(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for BF16 ablation test evaluation')
    model = QASTEHR(**cfg.hyper_params.model)
    state = torch.load(checkpoint, map_location='cpu')
    if not isinstance(state, Mapping):
        raise RuntimeError(f'checkpoint state must be a mapping: {checkpoint}')
    stripped_state = {str(name).removeprefix('module.'): value for (name, value) in state.items()}
    model.load_state_dict(stripped_state, strict=True)
    device = torch.device('cuda')
    model.to(device).eval()
    loader = get_dloaders(cfg)['test']
    totals: defaultdict[str, int] = defaultdict(int)
    correct: defaultdict[str, int] = defaultdict(int)
    with torch.inference_mode():
        for sample in loader:
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

    def metric(key: str) -> dict[str, int | float]:
        total = totals[key]
        hits = correct[key]
        return {'accuracy': 100.0 * hits / max(total, 1), 'correct': hits, 'total': total}
    return {'audio': metric('Audio'), 'visual': metric('Visual'), 'audio_visual': metric('Audio-Visual'), 'overall': metric('overall'), 'av_temporal': metric('av_temporal'), 'checkpoint': str(checkpoint.resolve()), 'config': str(config_path.resolve()), 'sha256': sha256(checkpoint)}

def _csv_text(results: Sequence[Mapping[str, object]]) -> str:
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator='\n')
    writer.writeheader()
    for result in results:
        writer.writerow({'model': result['model'], 'audio': result['audio']['accuracy'], 'visual': result['visual']['accuracy'], 'audio_visual': result['audio_visual']['accuracy'], 'overall': result['overall']['accuracy'], 'av_temporal': result['av_temporal']['accuracy'], 'delta_overall': result['delta_overall'], 'delta_av_temporal': result['delta_av_temporal'], 'checkpoint': result['checkpoint'], 'sha256': result['sha256']})
    return stream.getvalue()

def _stage_text(path: Path, content: str) -> Path:
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    return temporary

def _write_final_outputs(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'ablation_test_results.json'
    csv_path = output_dir / 'ablation_test_results.csv'
    flag_path = output_dir / 'ablation_tests_consumed.flag'
    contents = {json_path: json.dumps(payload, ensure_ascii=False, indent=2) + '\n', csv_path: _csv_text(payload['results']), flag_path: f"evaluated_at={payload['evaluated_at']}\nmodels={','.join(EXPECTED_ORDER)}\n"}
    staged: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        for (final_path, content) in contents.items():
            staged[final_path] = _stage_text(final_path, content)
        for final_path in (json_path, csv_path, flag_path):
            os.replace(staged[final_path], final_path)
            committed.append(final_path)
    except Exception:
        for final_path in committed:
            final_path.unlink(missing_ok=True)
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)

def _run(manifest_path: Path, output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    flag_path = output_dir / 'ablation_tests_consumed.flag'
    entries = require_manifest(manifest_path.resolve(), flag_path)
    results = []
    for entry in entries:
        measured = evaluate(entry.config, entry.checkpoint)
        row = {'model': entry.name, **measured}
        row['delta_overall'] = row['overall']['accuracy'] - BASELINE['overall']
        row['delta_av_temporal'] = row['av_temporal']['accuracy'] - BASELINE['av_temporal']
        results.append(row)
    payload = {'evaluated_at': datetime.now(timezone.utc).isoformat(), 'baseline': dict(BASELINE), 'results': results}
    _write_final_outputs(output_dir, payload)
    return payload

def main(argv: Sequence[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description='Guarded one-time test evaluation for four QAST-EHR ablations')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = _run(args.manifest, args.output_dir)
    except Exception as exc:
        print(f'ABLATION TEST EVALUATION REFUSED: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
