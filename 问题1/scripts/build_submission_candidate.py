"""Build an explicit core delivery candidate, excluding model/media caches."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from common import write

ROOT = Path(__file__).resolve().parents[1]
CORE_SCRIPTS = '''common math_features ctc_alignment local_alignment selective_alignment
audit extract feature_io temporal_support q1_reader delivery_reader local_evidence run_local_evidence evaluate_local_evidence run_automatic_alignment
build_temporal_index verify_temporal_index verify_automatic_alignment build_delivery_summary
verify_portable_reader rebuild_core compare_core_rebuild fetch_core_models build_paper record_environment build_delivery_figure verify_clean_reproduction evaluate_pipeline_quality build_quality_report verify_pipeline_quality
evaluate_independent_content build_attachment_audit verify_attachment_audit build_submission_candidate verify_submission_candidate
fetch_whisper_model fetch_acoustic_models run_whisper_recognition whisper_support prepare_whisper_local run_whisper_local_mfa
acoustic_mapping partial_acoustic_mapping phrase_selection evaluate_phrase_selection multigranular_alignment multigranular_reader build_multigranular_alignment
evaluate_multigranular_alignment verify_multigranular build_multigranular_report rebuild_multigranular compare_multigranular_rebuild'''.split()
TESTS = ['core', 'ctc_alignment', 'local_alignment', 'selective_alignment', 'temporal_support', 'q1_reader', 'independent_content', 'local_evidence', 'delivery_reader','multigranular_alignment', 'acoustic_mapping', 'partial_acoustic_mapping', 'phrase_selection']


def contents():
    files = {}
    def add(relative):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files[relative] = path.read_bytes()
    for name in CORE_SCRIPTS:
        add(f'scripts/{name}.py')
    for name in TESTS:
        add(f'tests/test_{name}.py')
    for name in ['method.json', 'automatic_alignment.json', 'environment.txt',
                 'mfa_environment_explicit.txt','whisper_alignment.json','whisper_environment.txt']:
        add(f'config/{name}')
    for name in ['问题1正文稿.md', 'scripts/paper_body.txt', '复现运行说明.md']:
        add(name)
    for directory in ['features', 'automatic_alignment', 'temporal_index', 'delivery_summary', 'figures', 'local_evidence_corroborated', 'attachment_audit', 'clean_environment', 'quality_evaluation','whisper_recognition','whisper_local','multigranular_alignment', 'acoustic_mapping', 'partial_acoustic_mapping', 'phrase_selection']:
        for p in sorted((ROOT / 'data' / directory).rglob('*')):
            if p.is_file():
                if directory=='figures':
                    figure=json.loads((ROOT/'data/figures/delivery_example.json').read_text())
                    allowed={'data/figures/delivery_example.'+ext for ext in ['json','png','pdf']}
                    allowed.update(f['image'] for f in figure['frames'])
                    if str(p.relative_to(ROOT)) not in allowed:continue
                add(str(p.relative_to(ROOT)))
    for name in ['inputs.json', 'models.json', 'audio_support_correction.json', 'runtime_environment.json', 'face_fields.json', 'requirements_source.json',
                 'input_hashes.json', 'automatic_evaluation.json', 'automatic_verification.json',
                 'independent_content_protocol.json', 'independent_content_review.json',
                 'independent_content_evaluation.json', 'independent_content_ablation_diagnostic.json']:
        add('data/' + name)
    add('data/acoustic_comparison/models.json')
    for path in sorted((ROOT/'研究/发音竞争').rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            add(str(path.relative_to(ROOT)))
    add('README.md')
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = contents()
    manifest = {'schema': 'q1-submission-candidate-v1', 'status': 'q1_deliverable_with_explicit_candidate_uncertainty',
        'sample_count': 100, 'semantic_accuracy_verified': False,
        'scope': 'Full native features and core reproducible pipeline, with current evaluation evidence and attachment-only signal checks.',
        'not_included': ['original contest media', 'pretrained model weights', 'review audio/UI'],
        'files': {name: {'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()}
                  for name, blob in sorted(files.items())}}
    files['submission_manifest.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    output = args.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.partial')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, blob in sorted(files.items()):
            info = zipfile.ZipInfo('q1/' + name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, blob, compresslevel=9)
    if temporary.stat().st_size > 50_000_000:
        raise ValueError('Candidate alone exceeds 50 million bytes')
    temporary.replace(output)
    write(ROOT / 'data/submission_build.json', {'archive': str(output), 'bytes': output.stat().st_size,
        'sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'files': len(files),
        'uncompressed_bytes': sum(map(len, files.values())), 'status': 'candidate_pending_archive_verification',
        'team_total_size_verified': False, 'human_anonymity_review_complete': False})
    print(f'{output}: {output.stat().st_size:,} bytes, {len(files)} files')


if __name__ == '__main__':
    main()
