"""Compare all regenerated numeric arrays and semantic decision fields."""
import argparse
from pathlib import Path
import numpy as np
from common import read, digest, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuilt', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rebuilt = args.rebuilt.resolve()
    run = read(rebuilt / 'data/core_rebuild.json')
    if run['status'] != 'passed':
        raise ValueError('Rebuild must have completed successfully')
    original_inputs = read(root / 'data/inputs.json')
    new_inputs = read(rebuilt / 'data/inputs.json')
    assert original_inputs['label_sha256'] == new_inputs['label_sha256']
    assert original_inputs['samples'] == new_inputs['samples'], 'Original records or clocks changed'
    results = []
    for sample in original_inputs['samples']:
        sid = sample['sample_id']
        row = {'sample_id': sid}
        for directory in ['features', 'temporal_index']:
            with np.load(root / 'data' / directory / (sid + '.npz'), allow_pickle=False) as a, \
                 np.load(rebuilt / 'data' / directory / (sid + '.npz'), allow_pickle=False) as b:
                assert set(a.files) == set(b.files)
                for key in a.files:
                    assert a[key].dtype == b[key].dtype and np.array_equal(a[key], b[key]), (sid, directory, key)
                row[directory + '_equal_arrays'] = len(a.files)
        a = read(root / 'data/features' / (sid + '.json'))
        b = read(rebuilt / 'data/features' / (sid + '.json'))
        # A review-file location changes when moving the run. Its PCM digest must not.
        a.pop('review_wav'); b.pop('review_wav')
        assert a == b, ('metadata', sid)
        a = read(root / 'data/automatic_alignment' / (sid + '.json'))
        b = read(rebuilt / 'data/automatic_alignment' / (sid + '.json'))
        for key in ['words', 'recognition', 'sample_status', 'informative_anchors',
                    'informative_reference_tokens', 'informative_coverage', 'edit_cost',
                    'optimal_endpoints', 'context_envelopes', 'automatic_eligible_words']:
            assert a[key] == b[key], ('automatic_decisions', sid, key)
        a = read(root / 'data/local_evidence_corroborated' / (sid + '.json'))
        b = read(rebuilt / 'data/local_evidence_corroborated' / (sid + '.json'))
        for key in ['words', 'local_support_runs', 'corroborating_run_pairs',
                    'decision_policy', 'automatic_eligible_words', 'new_local_candidate_words']:
            assert a[key] == b[key], ('local_corroboration', sid, key)
        results.append(row)
    assert digest(root / 'data/delivery_summary/all_100.csv') == digest(rebuilt / 'data/delivery_summary/all_100.csv')
    write(root / 'data/delivery_summary/full_rebuild_comparison.json', {
        'passed': True, 'samples': len(results), 'per_sample': results,
        'native_arrays_equal': sum(r['features_equal_arrays'] for r in results),
        'physical_index_arrays_equal': sum(r['temporal_index_equal_arrays'] for r in results),
        'all_decision_fields_equal': True, 'all_source_records_and_labels_equal': True,
        'all_local_corroboration_decisions_equal': True,
        'metadata_ignored_fields': ['review_wav absolute path only'],
        'scope': 'Full 100 fresh inference in a new directory; exact array and decision comparison. Installation provenance is recorded separately; this does not establish semantic accuracy or portability to arbitrary hardware.',
        'rebuild_manifest': run, 'rebuild_manifest_sha256': digest(rebuilt / 'data/core_rebuild.json'),
        'comparison_script_sha256': digest(Path(__file__))})
    print('All 100 source records, native/index arrays and automatic decisions match exactly.')


if __name__ == '__main__':
    main()
