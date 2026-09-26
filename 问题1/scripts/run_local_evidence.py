"""Generate local-evidence experiment for every original sample, without reviews."""
from pathlib import Path
from common import read, write, digest
import argparse
from local_evidence import local_candidates, corroborated_local_candidates

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corroborated', action='store_true')
    args = parser.parse_args()
    output = ROOT / ('data/local_evidence_corroborated' if args.corroborated else 'data/local_evidence')
    decide = corroborated_local_candidates if args.corroborated else local_candidates
    output.mkdir(exist_ok=False)
    config = read(ROOT / 'config/automatic_alignment.json')
    rows = []
    for source in read(ROOT / 'data/inputs.json')['samples']:
        sid = source['sample_id']
        path = ROOT / 'data/automatic_alignment' / (sid + '.json')
        baseline = read(path)
        assert baseline['source_sha256'] == source['sha256']
        result = decide(baseline, config)
        result.update(baseline_result_sha256=digest(path), human_data_used_for_generation=False)
        write(output / (sid + '.json'), result)
        rows.append({'sample_id': sid, 'words': len(result['words']),
            'baseline_candidates': baseline['automatic_eligible_words'],
            'candidate_words': result['automatic_eligible_words'],
            'new_local_candidates': result['new_local_candidate_words']})
    write(output / 'manifest.json', {'samples': rows, 'total_samples': len(rows),
        'candidate_words': sum(r['candidate_words'] for r in rows),
        'added_words': sum(r['new_local_candidates'] for r in rows),
        'human_data_used_for_generation': False,
        'design_informed_by_development_feedback': True,
        'config_sha256': digest(ROOT / 'config/automatic_alignment.json'),
        'scripts': {name: digest(ROOT / 'scripts' / name) for name in ['local_evidence.py', 'run_local_evidence.py']},
        'status': 'experiment_not_default_reader_policy'})
    print('Samples:', len(rows), 'added candidates:', sum(r['new_local_candidates'] for r in rows))


if __name__ == '__main__':
    main()
