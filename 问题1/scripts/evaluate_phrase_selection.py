"""Ablate phrase decisions on identical inference evidence, without review input."""
import copy
import hashlib
import json
from pathlib import Path
from collections import Counter
from common import read, write, digest
from multigranular_alignment import combine
from phrase_selection import compatible

ROOT = Path(__file__).resolve().parents[1]


def word_digest(words):
    return hashlib.sha256(json.dumps(words, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def metrics(result):
    phrases = result['phrases']
    ordered = sorted(phrases, key=lambda p: p['word_indices'][0])
    return {'word_candidates': sum(w['automatic_eligible'] for w in result['words']),
            'phrase_only_words': len(result['phrase_only_words']), 'phrases': len(phrases),
            'incompatible_pairs': sum(not compatible(a, b) for i, a in enumerate(ordered) for b in ordered[i+1:]),
            'group_word_memberships': sum(len(p['word_indices']) for p in phrases)}


def evaluate():
    out = ROOT / 'data/phrase_selection'
    reference = read(out / 'reference.json')
    cfg = read(ROOT / 'config/whisper_alignment.json')
    evidence_path = ROOT / 'data/multigranular_alignment/acoustic_evidence.json'
    assert reference['acoustic_evidence_sha256'] == digest(evidence_path)
    assert reference['config_sha256'] == digest(ROOT / 'config/whisper_alignment.json')
    sources = {s['sample_id']: s for s in read(ROOT / 'data/inputs.json')['samples']}
    old_records = {s['sample_id']: s for s in reference['samples']}
    assert set(old_records) == set(sources)
    policies = ['early_greedy', 'deferred_greedy', 'optimal']
    totals = {mode: Counter() for mode in policies}
    rows = []; controls = Counter()
    for e in read(evidence_path)['records']:
        sid = e['sample_id']; s = sources[sid]
        words = read(ROOT / 'data/features' / (sid + '.json'))['words']
        baseline = read(ROOT / 'data/local_evidence_corroborated' / (sid + '.json'))
        bounds = [s['audio']['start_s'], s['audio']['end_s']]
        args = (words, baseline, e['support'], e['runs'], cfg, bounds, s['zero_audio'])
        results = {mode: combine(*args, phrase_policy=mode) for mode in policies}
        old = old_records[sid]
        assert word_digest(results['early_greedy']['words']) == old['words_sha256']
        assert results['early_greedy']['phrases'] == old['phrases']
        assert results['early_greedy']['phrase_only_words'] == old['phrase_only_words']
        assert all(r['words'] == results['optimal']['words'] for r in results.values())
        actual = read(ROOT / 'data/multigranular_alignment' / (sid + '.json'))
        assert all(actual[k] == v for k, v in results['optimal'].items())
        sample_metrics = {mode: metrics(result) for mode, result in results.items()}
        for mode, values in sample_metrics.items(): totals[mode].update(values)
        assert sample_metrics['optimal']['incompatible_pairs'] == 0
        for p in results['optimal']['phrases']:
            assert not p['internal_word_times_assigned']
        # Ordering of independent evidence records cannot change selected relations.
        permuted = combine(words, baseline, e['support'], list(reversed(e['runs'])), cfg, bounds, s['zero_audio'])
        assert permuted['words'] == actual['words'] and permuted['phrases'] == actual['phrases']
        controls['run_order_permutations'] += 1
        for condition in ['missing_context', 'outside_audio']:
            runs = copy.deepcopy(e['runs'])
            for run in runs:
                if condition == 'missing_context': run['contexts'][1] = []
                else:
                    for context in run['contexts']:
                        for word in context:
                            if word['interval_s'] is not None:
                                word['interval_s'] = [v + bounds[1] - bounds[0] + 1 for v in word['interval_s']]
            result = combine(words, baseline, e['support'], runs, cfg, bounds, s['zero_audio'])
            assert result['added_words'] == 0 and not result['phrases']
            controls[condition + '_no_additions'] += 1
        rows.append({'sample_id': sid, 'metrics': sample_metrics,
                     'gained_coarse_word_indices': sorted(set(actual['phrase_only_words']) - set(old['phrase_only_words'])),
                     'lost_coarse_word_indices': sorted(set(old['phrase_only_words']) - set(actual['phrase_only_words']))})
    return {
        'status': 'passed', 'samples': len(rows), 'words': sum(len(read(ROOT/'data/features'/(sid+'.json'))['words']) for sid in sources),
        'totals': {k: dict(v) for k, v in totals.items()}, 'samples_with_added_coverage': sum(bool(r['gained_coarse_word_indices']) for r in rows),
        'gained_coarse_words': sum(len(r['gained_coarse_word_indices']) for r in rows),
        'lost_coarse_words': sum(len(r['lost_coarse_word_indices']) for r in rows),
        'controls': dict(controls), 'sample_results': rows,
        'reference_sha256': digest(out/'reference.json'), 'acoustic_evidence_sha256': digest(evidence_path),
        'current_manifest_sha256': digest(ROOT/'data/multigranular_alignment/manifest.json'),
        'scripts_sha256': {n: digest(ROOT/'scripts'/n) for n in ['phrase_selection.py', 'multigranular_alignment.py', 'evaluate_phrase_selection.py']},
        'word_decisions_unchanged': True, 'previous_policy_exactly_reproduced': True,
        'human_data_used': False, 'independent_accuracy_measured': False,
        'scope': 'Fixed-evidence selection ablation and structural controls. Not semantic accuracy or true boundary error.'}


def main():
    report = evaluate()
    write(ROOT / 'data/phrase_selection/evaluation.json', report)
    print(report['totals'])
    print(report['controls'])


if __name__ == '__main__':
    main()
