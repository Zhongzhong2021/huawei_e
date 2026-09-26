"""Record fresh-environment evidence after a real full rebuild and comparison."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import zipfile
from common import digest, read, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuilt', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--install-log', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = Path(sys.prefix)
    cfg = (env / 'pyvenv.cfg').read_text()
    assert sys.prefix != sys.base_prefix
    assert 'include-system-site-packages = false' in cfg.lower()
    expected = {}
    for line in (root / 'config/environment.txt').read_text().splitlines():
        name, version = line.strip().split('==')
        assert importlib.metadata.version(name) == version, name
        expected[name] = version
    check = subprocess.run([sys.executable, '-m', 'pip', 'check'],
                           capture_output=True, text=True, check=True)
    install_log = args.install_log.read_text()
    assert 'Successfully installed' in install_log
    # Verify this environment really executed every step, not just the comparison.
    run = read(args.rebuilt / 'data/core_rebuild.json')
    assert run['status'] == 'passed' and run['human_data_used'] is False
    for step in run['steps']:
        assert step['command'][0] == sys.executable and step['returncode'] == 0
        assert digest(args.rebuilt / step['log']) == step['log_sha256']
    with zipfile.ZipFile(args.package) as archive:
        for name, sha in run['scripts'].items():
            assert hashlib.sha256(archive.read('q1/scripts/' + name)).hexdigest() == sha
        assert archive.read('q1/config/environment.txt') == (root / 'config/environment.txt').read_bytes()
    comparison = read(root / 'data/delivery_summary/full_rebuild_comparison.json')
    assert comparison['rebuild_manifest_sha256'] == digest(args.rebuilt / 'data/core_rebuild.json')
    assert comparison['passed'] and comparison['samples'] == 100
    assert comparison['native_arrays_equal'] == 1900
    assert comparison['physical_index_arrays_equal'] == 1200
    assert comparison['all_decision_fields_equal']
    assert comparison['all_local_corroboration_decisions_equal']
    evidence = root / 'data/clean_environment'
    evidence.mkdir(exist_ok=True)
    (evidence / 'install.log').write_text(install_log)
    (evidence / 'pyvenv.cfg').write_text(cfg)
    write(evidence / 'verification.json', {
        'status': 'passed', 'isolated_venv': True,
        'fresh_install_log_sha256': digest(evidence / 'install.log'),
        'environment_lock_sha256': digest(root / 'config/environment.txt'),
        'installed_locked_packages': expected, 'pip_check': check.stdout.strip(),
        'input_package_sha256': digest(args.package),
        'rebuild_manifest_sha256': digest(args.rebuilt / 'data/core_rebuild.json'),
        'comparison_sha256': digest(root / 'data/delivery_summary/full_rebuild_comparison.json'),
        'runtime_environment': read(args.rebuilt / 'data/runtime_environment.json'),
        'samples': 100, 'exact_native_arrays': 1900, 'exact_physical_arrays': 1200,
        'all_automatic_decisions_equal': True, 'human_records_required': False,
        'model_files': 'Reused fixed cache after SHA256 validation; no new model download claimed.',
        'scope': 'New isolated Python environment on the same OS, hardware and FFmpeg. Not a cross-platform or semantic-accuracy guarantee.'})
    print(json.dumps({'status': 'passed', 'samples': 100, 'exact_arrays': 3100}))


if __name__ == '__main__':
    main()
