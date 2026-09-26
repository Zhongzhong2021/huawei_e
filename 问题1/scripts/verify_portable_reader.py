"""Copy only required data to a fresh directory and verify reading all samples."""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from common import digest, read, write


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='q1-reader-') as temporary:
        target = Path(temporary)
        (target / 'data').mkdir()
        shutil.copytree(root / 'scripts', target / 'scripts', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(root / 'data/inputs.json', target / 'data/inputs.json')
        shutil.copytree(root / 'config', target / 'config')
        for name in ['features', 'automatic_alignment', 'temporal_index', 'local_evidence_corroborated']:
            shutil.copytree(root / 'data' / name, target / 'data' / name)
        subprocess.run([sys.executable, str(target / 'scripts/build_delivery_summary.py')],
                       cwd=temporary, check=True)
        for name in ['all_100.csv', 'all_100.md']:
            if digest(target / 'data/delivery_summary' / name) != digest(root / 'data/delivery_summary' / name):
                raise AssertionError('Relocation changed summary: ' + name)
        code = '''
import sys
sys.path.insert(0, 'scripts')
from delivery_reader import Dataset
from common import write
ds = Dataset('.')
candidate = missing = 0
for sid in ds.records:
    s = ds.sample(sid)
    for i in range(len(s.automatic['words'])):
        r = s.word(i)
        if r['relation'] is None:
            missing += 1
        else:
            candidate += 1
            assert not r['semantic_verified_by_this_function']
            assert r['relation']['audio']['usable']
    r = s.window(s.source['audio']['start_s'], s.source['audio']['end_s'])
    assert len(r['text_rows']) == 0
    assert r['audio']['usable'] != s.source['zero_audio']
write('read_checks.json', {'candidates': candidate, 'unresolved': missing, 'samples': len(ds.records)})
'''
        subprocess.run([sys.executable, '-c', code], cwd=temporary, check=True)
        counts = read(target / 'read_checks.json')
        write(root / 'data/delivery_summary/portable_read_verification.json', {
            **counts, 'status': 'passed', 'identical_summary_after_relocation': True,
            'source_media_or_model_inference_rerun': False,
            'human_files_or_diagnostic_models_required': False,
            'environment': 'existing Python/NumPy; fresh directory, not a newly installed environment',
            'scope': 'identity, feature/index hashes, all word queries, full-audio physical queries',
            'script_sha256': digest(Path(__file__))})


if __name__ == '__main__':
    main()
