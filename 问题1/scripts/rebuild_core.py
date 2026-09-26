"""Rebuild the complete core pipeline in a new directory; never overwrite a run."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from common import digest, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source, models, output = (p.resolve() for p in (args.source, args.models, args.output))
    if not source.is_dir() or not (models / 'manifest.json').is_file():
        parser.error('Source directory and fixed model cache manifest must exist')
    if output.exists():
        parser.error('Output must be a new directory; existing runs are never overwritten')
    output.mkdir(parents=True)
    shutil.copytree(root / 'scripts', output / 'scripts', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(root / 'config', output / 'config')
    (output / 'data/logs').mkdir(parents=True)
    steps = [
        ['record_environment.py'],
        ['audit.py', '--source', str(source)],
        ['extract.py', '--models', str(models), '--media', str(output / 'review-media')],
        ['run_automatic_alignment.py', '--models', str(models)],
        ['run_local_evidence.py', '--corroborated'],
        ['build_temporal_index.py'], ['verify_temporal_index.py'],
        ['verify_automatic_alignment.py'], ['build_delivery_summary.py'],
    ]
    manifest = {'status': 'running', 'started_unix_s': time.time(),
        'source': str(source), 'models_manifest_sha256': digest(models / 'manifest.json'),
        'python': sys.version, 'steps': [], 'human_data_used': False,
        'scripts': {p.name: digest(p) for p in (output / 'scripts').glob('*.py')},
        'scope': 'All 100 native features, selective word candidates, physical indices and summary.'}
    env = dict(os.environ, PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    write(output / 'data/core_rebuild.json', manifest)
    for index, step in enumerate(steps):
        cmd = [sys.executable, str(output / 'scripts' / step[0]), *step[1:]]
        log = output / 'data/logs' / f'{index + 1:02d}-{Path(step[0]).stem}.log'
        print('Running', step[0], flush=True)
        start = time.monotonic()
        with log.open('w') as stream:
            result = subprocess.run(cmd, cwd=output, env=env, stdout=stream, stderr=subprocess.STDOUT)
        manifest['steps'].append({'command': cmd, 'returncode': result.returncode,
            'seconds': time.monotonic() - start, 'log': str(log.relative_to(output)),
            'log_sha256': digest(log)})
        if result.returncode:
            manifest['status'] = 'failed'
            write(output / 'data/core_rebuild.json', manifest)
            raise SystemExit(f'Failed {step[0]}; inspect {log}')
        write(output / 'data/core_rebuild.json', manifest)
    manifest.update(status='passed', finished_unix_s=time.time())
    write(output / 'data/core_rebuild.json', manifest)
    print('Complete core rebuild passed:', output, flush=True)


if __name__ == '__main__':
    main()
