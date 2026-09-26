"""Build an optional, conflict-checked phrase sidecar without changing base results."""
import math
import os
import sys
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parents[1]
Q = Path(os.environ.get('Q1_PROJECT', R.parents[1]))
sys.path.insert(0, str(Q / 'scripts'))
from common import read, write, digest
from phrase_selection import select_phrases

EPS = 1e-7  # Floating-point comparison only; no additional timing tolerance.


def relation_consistent(candidate, existing):
    """Disjoint index spans must be ordered; nested spans must contain in time."""
    a, b = candidate['word_indices'], existing['word_indices']
    x, y = candidate['interval_s'], existing['interval_s']
    if a[-1] < b[0]:
        return x[1] <= y[0] + EPS
    if b[-1] < a[0]:
        return y[1] <= x[0] + EPS
    if set(b) <= set(a):
        return x[0] <= y[0] + EPS and y[1] <= x[1] + EPS
    if set(a) <= set(b):
        return y[0] <= x[0] + EPS and x[1] <= y[1] + EPS
    return False


def screen(candidate, base, source):
    indices = candidate['word_indices']
    interval = candidate['interval_s']
    if (not indices or indices != list(range(indices[0], indices[-1] + 1))
            or indices[0] < 0 or indices[-1] >= len(base['words'])):
        return 'invalid_indices', []
    if (len(interval) != 2 or not all(math.isfinite(x) for x in interval)
            or not source.get('audio') or source['zero_audio']
            or not source['audio']['start_s'] - EPS <= interval[0] < interval[1]
            or interval[1] > source['audio']['end_s'] + EPS):
        return 'unavailable_audio_range', []
    if candidate['reference_words'] != [base['words'][i]['text'] for i in indices]:
        return 'reference_mismatch', []
    if any(base['words'][i]['status'] == 'cross_model_time_conflict' for i in indices):
        return 'existing_time_conflict', []
    covered = {i for i, w in enumerate(base['words']) if w['automatic_eligible']}
    covered.update(i for p in base['phrases'] for i in p['word_indices'])
    novel = sorted(set(indices) - covered)
    if not novel:
        return 'already_covered', []
    relations = [{'word_indices': [i], 'interval_s': w['interval_s']}
                 for i, w in enumerate(base['words']) if w['automatic_eligible']]
    relations += base['phrases']
    if not all(relation_consistent(candidate, p) for p in relations):
        return 'base_relation_conflict', novel
    return 'eligible', novel


def build():
    identity = R / 'data/contextual/identity.json'
    inputs = Q / 'data/inputs.json'
    sources = read(inputs)['samples']
    rows = read(identity)['rows']
    records, counts, base_hashes = [], Counter(), {}
    new_words = 0
    for source in sources:
        sid = source['sample_id']
        path = Q / 'data/multigranular_alignment' / (sid + '.json')
        base = read(path)
        assert base['source_sha256'] == source['sha256']
        base_hashes[sid] = digest(path)
        decisions, eligible = [], []
        for row_index, p in enumerate(rows):
            if p['sample_id'] != sid:
                continue
            # Require both expected windows, not vacuous all([]).
            supported = (p['accepted'] and len(p['evidence']) == 2
                         and [e['context_s'] for e in p['evidence']] == [0.0, 0.1]
                         and all(e['accepted'] for e in p['evidence']))
            status, novel = screen(p, base, source) if supported else ('acoustic_checks_failed', [])
            decision = {'evidence_row': row_index, 'word_indices': p['word_indices'],
                        'reference_words': p['reference_words'], 'interval_s': p['interval_s'],
                        'new_word_indices': novel, 'status': status,
                        'internal_word_times_assigned': False}
            decisions.append(decision)
            if status == 'eligible':
                eligible.append(decision)
        covered = {i for i, w in enumerate(base['words']) if w['automatic_eligible']}
        covered.update(i for p in base['phrases'] for i in p['word_indices'])
        masks = [{'automatic_eligible': i in covered} for i in range(len(base['words']))]
        selected = select_phrases(eligible, masks)
        selected_rows = {p['evidence_row'] for p in selected}
        for p in decisions:
            if p['status'] == 'eligible':
                p['status'] = ('supplemental_phrase_candidate' if p['evidence_row'] in selected_rows
                               else 'joint_selection_excluded')
            counts[p['status']] += 1
        new_words += len({i for p in selected for i in p['new_word_indices']})
        records.append({'sample_id': sid, 'source_sha256': source['sha256'],
                        'phrases': selected, 'decisions': decisions})
    return {'schema': 'q1-phonetic-enhancement-v1', 'mode': 'contextual',
            'status': 'optional_candidate_sidecar', 'human_data_used': False,
            'base_outputs_modified': False, 'independent_accuracy_measured': False,
            'interval_semantics': 'ASR anchor window supported as a whole; not refined phonetic boundaries.',
            'input_sha256': digest(inputs), 'evidence_sha256': digest(identity),
            'protocol_sha256': digest(R / 'protocol.json'), 'base_sha256': base_hashes,
            'code_sha256': digest(__file__),
            'selector_sha256': digest(Q / 'scripts/phrase_selection.py'),
            'summary': {'samples': len(records), 'candidate_decisions': sum(counts.values()),
                        'statuses': dict(sorted(counts.items())),
                        'potential_new_words': new_words, 'new_words_promoted': 0},
            'samples': records}


if __name__ == '__main__':
    result = build()
    write(R / 'data/enhancement.json', result)
    print(result['summary'])
