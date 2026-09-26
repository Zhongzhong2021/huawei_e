"""Evaluate content judgments against frozen outputs without altering predictions."""
import argparse
from collections import Counter
from pathlib import Path
from common import read, digest, write


def evaluate(review, protocol):
    if review.get('schema') != 'q1-independent-content-review-v1':
        raise ValueError('Unexpected review schema')
    expected = {s['sample_id']: s for s in protocol['samples']}
    if len(expected) != len(protocol['samples']):
        raise ValueError('Duplicate protocol sample')
    records, counts, groups = {}, Counter(), {}
    allowed = {'', 'complete', 'partial', 'absent', 'uncertain'}
    for record in review['samples']:
        sid = record['sample_id']
        if sid not in expected or sid in records:
            raise ValueError('Unknown or duplicate sample')
        if record['source_sha256'] != expected[sid]['source_sha256']:
            raise ValueError('Reference source mismatch')
        value = record.get('content_relation', '')
        heard = record.get('listened_full_audio', False)
        if value not in allowed or type(heard) is not bool:
            raise ValueError('Invalid judgment or full-audio flag')
        if value in {'complete', 'partial', 'absent'} and not heard:
            raise ValueError('Definite judgment requires full audio review')
        records[sid] = record
    for sid, prediction in expected.items():
        value = records.get(sid, {}).get('content_relation', '') or 'unfilled'
        counts[value] += 1
        if value not in {'complete', 'partial', 'absent'}:
            continue
        group = groups.setdefault(value, {'samples': 0, 'original_words': 0,
            'candidate_words': 0, 'samples_with_any_candidate': 0})
        group['samples'] += 1
        group['original_words'] += prediction['reference_words']
        group['candidate_words'] += prediction['candidate_words']
        group['samples_with_any_candidate'] += prediction['candidate_words'] > 0
    for group in groups.values():
        group['candidate_coverage'] = (group['candidate_words'] / group['original_words']
                                       if group['original_words'] else None)
    return {'status': 'partial_pilot' if counts['unfilled'] else 'pilot_reviewed',
        'counts': dict(counts), 'content_groups': groups, 'expected_samples': len(expected),
        'absent_samples_with_any_candidate': groups.get('absent', {}).get('samples_with_any_candidate', 0),
        'word_boundary_accuracy': None, 'word_precision': None,
        'automatic_results_modified': False,
        'interpretation': 'Coverage is availability, not accuracy. Absent groups remain separate; uncertain and unfilled are not negatives.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / 'data/independent_content_protocol.json'
    protocol = read(path)
    for name, expected in protocol['method_sha256'].items():
        if digest(root / name) != expected:
            raise ValueError('Frozen method changed; do not label this an independent fixed-method evaluation')
    for s in protocol['samples']:
        if digest(root / 'data/automatic_alignment' / (s['sample_id'] + '.json')) != s['automatic_result_sha256']:
            raise ValueError('Frozen prediction changed')
    if args.output.exists() or args.output.resolve() == args.review.resolve():
        raise ValueError('Use a new output file; review and existing reports are preserved')
    result = evaluate(read(args.review), protocol)
    result.update(protocol_sha256=digest(path), review_sha256=digest(args.review),
                  method_and_prediction_hashes_checked=True)
    write(args.output, result)
    print(result['status'], result['counts'])


if __name__ == '__main__':
    main()
