"""Check archive hashes, safe paths, complete IDs, and relocated reads."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile
from common import read, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--extended-tests', action='store_true', help='Also run phonetic tests; requires the module extra dependencies.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    archive_path = args.archive.resolve()
    with tempfile.TemporaryDirectory(prefix='q1-submission-') as directory:
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            assert not any('/source_audit/' in n or 'verify_source_audit_evidence' in n for n in names)
            assert len(names) == len(set(names)), 'Duplicate archive member'
            for name in names:
                p = PurePosixPath(name)
                assert not p.is_absolute() and '..' not in p.parts and p.parts[0] == 'q1'
            manifest = json.loads(archive.read('q1/submission_manifest.json'))
            expected = {'q1/' + p for p in manifest['files']} | {'q1/submission_manifest.json'}
            assert set(names) == expected
            for name, item in manifest['files'].items():
                blob = archive.read('q1/' + name)
                assert len(blob) == item['bytes'] and hashlib.sha256(blob).hexdigest() == item['sha256']
            archive.extractall(directory)
        extracted = Path(directory) / 'q1'
        inputs = read(extracted / 'data/inputs.json')['samples']
        ids = {s['sample_id'] for s in inputs}
        assert len(ids) == len(inputs) == 100
        for folder in ['features', 'temporal_index']:
            assert {p.stem for p in (extracted / 'data' / folder).glob('*.npz')} == ids
        # This subprocess imports the extracted code, with extracted data as its cwd.
        check = '''import sys
sys.path.insert(0, 'scripts')
from delivery_reader import Dataset
from common import write
ds=Dataset('.')
counts={'samples':0,'words':0,'candidates':0}
for sid in ds.records:
    s=ds.sample(sid); counts['samples']+=1
    for i in range(len(s.automatic['words'])):
        r=s.word(i); counts['words']+=1; counts['candidates']+=r['relation'] is not None
    r=s.window(s.source['audio']['start_s'],s.source['audio']['end_s'])
    assert len(r['text_rows'])==0
write('archive_read_result.json',counts)
'''
        subprocess.run([sys.executable, '-c', check], cwd=extracted, check=True)
        attachment_check = subprocess.run([sys.executable, 'scripts/verify_attachment_audit.py'],
                                      cwd=extracted, capture_output=True, text=True)
        assert attachment_check.returncode == 0, attachment_check.stdout + attachment_check.stderr
        tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'],
                               cwd=extracted, capture_output=True, text=True)
        assert tests.returncode == 0, tests.stdout + tests.stderr
        text = (extracted / '问题1正文稿.md').read_text()
        assert sum(line.startswith('| -') for line in text.splitlines()) == 100
        assert (extracted / 'data/figures/delivery_example.png').is_file()
        figure = read(extracted / 'data/figures/delivery_example.json')
        assert len(figure['word_intervals_s']) == len(figure['text_rows']) > 0
        assert figure['human_data_used_for_generation'] is False
        for frame in figure['frames']:
            assert hashlib.sha256((extracted / frame['image']).read_bytes()).hexdigest() == frame['image_sha256']
        assert set(read(extracted / 'data/runtime_environment.json')['tools']) == {'ffmpeg', 'ffprobe'}
        clean = read(extracted / 'data/clean_environment/verification.json')
        assert clean['status'] == 'passed' and clean['isolated_venv']
        assert clean['fresh_install_log_sha256'] == hashlib.sha256(
            (extracted / 'data/clean_environment/install.log').read_bytes()).hexdigest()
        assert clean['environment_lock_sha256'] == hashlib.sha256(
            (extracted / 'config/environment.txt').read_bytes()).hexdigest()
        assert clean['comparison_sha256'] == hashlib.sha256(
            (extracted / 'data/delivery_summary/full_rebuild_comparison.json').read_bytes()).hexdigest()
        quality = subprocess.run([sys.executable, 'scripts/verify_pipeline_quality.py'],
                                 cwd=extracted, capture_output=True, text=True)
        assert quality.returncode == 0, quality.stdout + quality.stderr
        enhanced=subprocess.run([sys.executable,'scripts/verify_multigranular.py'],cwd=extracted,capture_output=True,text=True)
        assert enhanced.returncode==0,enhanced.stdout+enhanced.stderr
        phonetic=subprocess.run([sys.executable,'研究/发音竞争/scripts/verify_research.py'],cwd=extracted,capture_output=True,text=True)
        assert phonetic.returncode==0,phonetic.stdout+phonetic.stderr
        extra=None
        if args.extended_tests:
            extra=subprocess.run([sys.executable,'-m','unittest','discover','-s','研究/发音竞争/scripts','-p','test_*.py'],cwd=extracted,capture_output=True,text=True)
            assert extra.returncode==0,extra.stdout+extra.stderr
        report = {'status': 'passed', 'archive_sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            'archive_bytes': archive_path.stat().st_size, 'files_hash_checked': len(manifest['files']),
            'safe_paths_and_unique_members': True, 'all_100_paper_rows_and_figure_present': True,
            'extracted_read': read(extracted / 'archive_read_result.json'),
            'attachment_audit_verified': True,
            'clean_environment_reproduction_evidence_verified': True,
            'controlled_quality_evidence_verified': True,
            'enhanced_evidence_verified':True,
            'phonetic_and_correspondence_evidence_verified':True,
            'phonetic_check_output':phonetic.stdout.strip(),
            'extended_test_output':extra.stdout+extra.stderr if extra is not None else None,
            'enhanced_read':read(extracted/'data/multigranular_alignment/verification.json'),
            'controlled_quality_check_output': quality.stdout.strip(),
            'attachment_audit_check_output': attachment_check.stdout.strip(),
            'extracted_test_output': tests.stdout + tests.stderr,
            'semantic_accuracy_verified': False, 'team_total_size_verified': False,
            'anonymity_scope': 'Curated files exclude personal review exports; not a completed manual identity audit.'}
        write(root / 'data/submission_verification.json', report)
        print(json.dumps({k: v for k, v in report.items() if k != 'extracted_test_output'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
