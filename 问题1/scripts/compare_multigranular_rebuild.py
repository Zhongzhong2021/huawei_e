"""Verify fresh inference and final decisions, then retain self-contained proof."""
import argparse
import shutil
from pathlib import Path
from common import read, write, digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuilt', type=Path, required=True)
    args = parser.parse_args()
    rebuilt = args.rebuilt.resolve()
    record = read(rebuilt / 'data/multigranular_rebuild.json')
    assert record['status'] == 'complete' and len(record['steps']) == 5
    out = ROOT / 'data/multigranular_alignment'
    reproduced = rebuilt / 'data/multigranular_alignment'
    for name in ['run_whisper_recognition.py', 'prepare_whisper_local.py', 'run_whisper_local_mfa.py',
                 'whisper_support.py', 'multigranular_alignment.py', 'phrase_selection.py',
                 'multigranular_reader.py', 'build_multigranular_alignment.py']:
        assert digest(ROOT / 'scripts' / name) == record['scripts'][name]
    before = read(out / 'acoustic_evidence.json')
    after = read(reproduced / 'acoustic_evidence.json')
    assert read(ROOT / 'data/whisper_recognition/protocol.json') == read(rebuilt / 'data/whisper_recognition/protocol.json')
    original_execution = read(ROOT / 'data/whisper_local/execution.json')
    fresh_execution = read(rebuilt / 'data/whisper_local/execution.json')
    for key in ['version', 'models', 'config_sha256', 'script_sha256']:
        assert original_execution[key] == fresh_execution[key], key
    for key in ['records', 'jobs', 'raw_sha256']:
        assert before[key] == after[key], key
    hashes = {}
    inputs = read(ROOT / 'data/inputs.json')
    assert read(rebuilt / 'data/inputs.json') == inputs
    for item in inputs['samples']:
        sid = item['sample_id']
        original = read(ROOT / 'data/whisper_recognition' / (sid + '.json'))
        fresh = read(rebuilt / 'data/whisper_recognition' / (sid + '.json'))
        original.pop('elapsed_s'); fresh.pop('elapsed_s')
        assert original == fresh, sid
        assert (out / (sid + '.json')).read_bytes() == (reproduced / (sid + '.json')).read_bytes(), sid
        hashes[sid] = digest(reproduced / (sid + '.json'))
    logs = out / 'reproduction_logs'; logs.mkdir(exist_ok=True)
    for step in record['steps']:
        assert step['exit_code'] == 0 and digest(rebuilt / step['log']) == step['log_sha256']
        shutil.copy2(rebuilt / step['log'], logs / step['log'])
    write(out / 'reproduction.json', {
        'status': 'passed', 'source_rebuild': str(rebuilt), 'samples': len(hashes),
        'recognition_words_and_segments_equal': True, 'all_multigranular_decisions_equal': True,
        'all_mapped_phonetic_evidence_and_raw_hashes_equal': True,
        'fresh_model_inference': True, 'fresh_local_phonetic_alignments': len(after['jobs']),
        'inference_protocols_and_model_hashes_equal': True, 'human_data_used': False,
        'hardware_identity_checked_by_this_script': False,
        'rebuild_record': record, 'relation_manifest_sha256': digest(out / 'manifest.json'),
        'reproduced_relation_sha256': hashes, 'comparison_script_sha256': digest(__file__)})
    print({'status': 'passed', 'samples': len(hashes), 'fresh_phonetic_tasks': len(after['jobs'])})


if __name__ == '__main__':
    main()
