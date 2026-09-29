from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time

working_root = Path('/kaggle/working')
project_root = working_root / 'CycleDiffusion'
generation_dir = working_root / 'layer8_trinity_full'
evaluation_dir = working_root / 'layer8_trinity_full_eval'
metrics_dir = working_root / 'layer8_trinity_full_metrics'
unit_manifest = working_root / 'trinity_full_manifest.csv'
generation_log = working_root / 'layer8_trinity_full_generation.log'
evaluation_log = working_root / 'layer8_trinity_full_eval.log'
supervisor_log = working_root / 'layer8_full_supervisor.log'
generation_pid_path = working_root / 'layer8_trinity_full_generation_pid.json'
status_path = working_root / 'layer8_trinity_full_status.json'
archive_path = working_root / 'layer8_trinity_full_evidence.tgz'
sha_path = working_root / 'layer8_trinity_full_evidence.sha256'

def emit(message):
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{stamp}] {message}', flush=True)

pid = int(json.loads(generation_pid_path.read_text())['pid'])
emit(f'waiting for generation PID {pid}')
while True:
    manifest_path = generation_dir / 'generation_manifest.json'
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        rows = payload.get('rows', [])
        expected = int(payload.get('config', {}).get('expected_outputs', -1))
        if expected == 480 and len(rows) == 480:
            break
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    journal = generation_dir / 'generation_progress.jsonl'
    completed = sum(1 for line in journal.read_text().splitlines() if line.strip()) if journal.exists() else 0
    if not alive:
        tail = generation_log.read_text(encoding='utf-8', errors='replace')[-5000:] if generation_log.exists() else ''
        raise RuntimeError(f'generation stopped at {completed}/480 without a complete manifest: {tail}')
    emit(f'generation progress {completed}/480')
    time.sleep(30)

emit('generation complete; starting evaluation')
evaluation_command = [
    sys.executable, str(project_root / 'evaluate_trinity_grid.py'),
    '--project-root', str(project_root),
    '--generated-dir', str(generation_dir),
    '--output-dir', str(evaluation_dir),
    '--batch-size', '8',
]
with evaluation_log.open('a', encoding='utf-8') as handle:
    result = subprocess.run(evaluation_command, stdout=handle, stderr=subprocess.STDOUT)
if result.returncode != 0:
    tail = evaluation_log.read_text(encoding='utf-8', errors='replace')[-8000:]
    raise RuntimeError(f'evaluation failed with {result.returncode}: {tail}')
raw_csv = evaluation_dir / 'trinity_eval_raw.csv'
if not raw_csv.exists():
    raise RuntimeError('evaluation completed without trinity_eval_raw.csv')
emit('evaluation complete; starting paired analysis')
analysis_command = [
    sys.executable, str(project_root / 'analyze_trinity.py'),
    '--evaluation-csv', str(raw_csv),
    '--output-dir', str(metrics_dir),
    '--bootstrap-repetitions', '50000',
    '--seed', '20260922',
]
analysis_result = subprocess.run(analysis_command, capture_output=True, text=True)
print(analysis_result.stdout, flush=True)
if analysis_result.returncode != 0:
    raise RuntimeError(analysis_result.stderr)
summary = json.loads((metrics_dir / 'trinity_summary.json').read_text())
if summary.get('evaluated_outputs') != 480 or summary.get('source_clusters_per_variant') != {'base_epoch50': 20}:
    raise RuntimeError(f'unexpected analysis summary: {summary}')

items = [
    (unit_manifest, 'manifest/trinity_full_manifest.csv'),
    (generation_dir / 'generation_config.json', 'generation/generation_config.json'),
    (generation_dir / 'generation_progress.jsonl', 'generation/generation_progress.jsonl'),
    (generation_dir / 'generation_manifest.json', 'generation/generation_manifest.json'),
    (evaluation_dir / 'trinity_eval_progress.jsonl', 'evaluation/trinity_eval_progress.jsonl'),
    (raw_csv, 'evaluation/trinity_eval_raw.csv'),
    (generation_log, 'logs/layer8_trinity_full_generation.log'),
    (evaluation_log, 'logs/layer8_trinity_full_eval.log'),
    (Path(__file__), 'scripts/layer8_full_supervisor.py'),
]
for name in ('build_trinity_manifest.py', 'generate_trinity_grid.py', 'evaluate_trinity_grid.py', 'analyze_trinity.py'):
    items.append((project_root / name, 'scripts/' + name))
with tarfile.open(archive_path, 'w:gz') as bundle:
    for source, archive_name in items:
        if not source.exists():
            raise FileNotFoundError(source)
        bundle.add(source, arcname=archive_name)
    for source in sorted(metrics_dir.iterdir()):
        if source.is_file():
            bundle.add(source, arcname='metrics/' + source.name)
digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
sha_path.write_text(digest + '  ' + archive_path.name + '\n')
status = {
    'status': 'complete',
    'evaluated_outputs': 480,
    'source_clusters': 20,
    'archive': str(archive_path),
    'archive_bytes': archive_path.stat().st_size,
    'sha256': digest,
}
status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + '\n')
emit('TRINITY_FULL_PIPELINE_COMPLETE ' + json.dumps(status, sort_keys=True))
