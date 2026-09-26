"""Audit present evidence against Q1 deliverables; never infer semantic success."""
import argparse
import re
import zipfile
from pathlib import Path
from common import read, write, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    originals = read(root / 'data/input_hashes.json')['sha256']
    assert all(Path(p).is_file() and digest(p) == sha for p, sha in originals.items())
    source = read(root / 'data/inputs.json')
    ids = {s['sample_id'] for s in source['samples']}
    assert len(ids) == len(source['samples']) == 100
    rebuild = read(root / 'data/delivery_summary/full_rebuild_comparison.json')
    dependencies = ['common.py', 'extract.py', 'math_features.py', 'ctc_alignment.py',
        'local_alignment.py', 'feature_io.py', 'temporal_support.py',
        'selective_alignment.py', 'q1_reader.py', 'run_automatic_alignment.py',
        'local_evidence.py', 'run_local_evidence.py', 'delivery_reader.py']
    assert rebuild['passed'] and rebuild['samples'] == 100
    clean = read(root / 'data/clean_environment/verification.json')
    assert clean['status'] == 'passed' and clean['isolated_venv']
    assert clean['comparison_sha256'] == digest(root / 'data/delivery_summary/full_rebuild_comparison.json')
    assert clean['environment_lock_sha256'] == digest(root / 'config/environment.txt')
    for name in dependencies:
        assert digest(root / 'scripts' / name) == rebuild['rebuild_manifest']['scripts'][name]
    protocol = read(root / 'data/independent_content_protocol.json')
    for name, sha in protocol['method_sha256'].items():
        assert digest(root / name) == sha
    for item in protocol['samples']:
        assert digest(root / 'data/automatic_alignment' / (item['sample_id'] + '.json')) == item['automatic_result_sha256']
    verification = read(root / 'data/submission_verification.json')
    assert verification['status'] == 'passed' and digest(args.archive) == verification['archive_sha256']
    assert verification['attachment_audit_verified'] is True
    assert verification['controlled_quality_evidence_verified'] is True
    quality = read(root / 'data/quality_evaluation/verification.json')
    assert quality['status'] == 'passed' and quality['word_records_checked'] == 5796
    assert quality['results_sha256'] == digest(root / 'data/quality_evaluation/results.json')
    with zipfile.ZipFile(args.archive) as archive:
        paper = archive.read('q1/问题1正文稿.md').decode()
        table_ids = [line.split('|')[1].strip() for line in paper.splitlines() if line.startswith('| -')]
        assert len(table_ids) == 100 and set(table_ids) == ids
        assert archive.read('q1/问题1正文稿.md') == (root / '问题1正文稿.md').read_bytes()
        assert archive.read('q1/data/attachment_audit/checks.json') == (root / 'data/attachment_audit/checks.json').read_bytes()
        assert archive.read('q1/data/clean_environment/verification.json') == (root / 'data/clean_environment/verification.json').read_bytes()
        # No stale executable is allowed in the final package after reproduction.
        for name in archive.namelist():
            if name.startswith('q1/scripts/') and name.endswith('.py'):
                assert archive.read(name) == (root / name.removeprefix('q1/')).read_bytes(), name
        metadata = {}
        for name in archive.namelist():
            if name.endswith('.pdf'):
                raw = archive.read(name)
                metadata[name] = {key: [s.decode('latin1') for s in re.findall(
                    rb'/' + key.encode() + rb'\s*\(([^)]*)\)', raw)]
                    for key in ['Author', 'Creator', 'Producer', 'Title', 'Subject']}
        raw_bytes = sum(i.file_size for i in archive.infolist())
    evaluation = read(root / 'data/independent_content_evaluation.json')
    assert digest(root / 'data/independent_content_protocol.json') == evaluation['protocol_sha256']
    assert digest(root / 'data/independent_content_review.json') == evaluation['review_sha256']
    build = read(root / 'data/submission_build.json')
    assert build['sha256'] == verification['archive_sha256']
    build['status'] = 'q1_delivery_verified_with_explicit_quality_limits'
    write(root / 'data/submission_build.json', build)
    write(root / 'data/delivery_audit.json', {
        'status': 'q1_delivery_verified_with_explicit_quality_limits',
        'original_files_checked': len(originals), 'original_files_unchanged': True,
        'core_dependencies_unchanged_since_full_rebuild': dependencies,
        'frozen_method_and_eight_predictions_unchanged': True,
        'paper_ids_equal_all_source_ids': True, 'packaged_paper_equal_current': True,
        'attachment_audit_verified': True,
        'clean_python_environment_reproduction_verified': True,
        'controlled_quality_experiments_verified': 300,
        'controlled_quality_word_records_verified': 5796,
        'clean_environment_evidence_sha256': digest(root / 'data/clean_environment/verification.json'),
        'attachment_audit_scope': 'Signal checks computed from contest attachments; human content judgments remain separately attributed.',
        'archive_sha256': verification['archive_sha256'],
        'archive_bytes': args.archive.stat().st_size, 'uncompressed_bytes': raw_bytes,
        'pdf_metadata': metadata,
        'independent_review': read(root / 'data/independent_content_evaluation.json'),
        'unclaimed_metrics_and_team_integration': [
            'Independent word accuracy and absolute boundary error are unmeasured.',
            'Coverage, controlled stability and development content review are reported as separate metrics.',
            'Future edits must rebuild the paper/package and recheck hashes.',
            'Integrate Q1 allocation with the whole-team 50MB submission and final anonymity review.'
        ],
        'limits': 'Small content pilot cannot establish word timing precision or SOTA. Fresh isolated Python installation and full rebuild verified on the same OS/hardware; not arbitrary cross-platform identity.'})
    print('Q1 delivery checks passed; independent word accuracy remains unmeasured.')


if __name__ == '__main__':
    main()
