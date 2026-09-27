"""Choose a maximum-coverage chain of evidence-qualified phrase intervals.

Compatibility requires disjoint reference indices AND ordered audio intervals.
Scores are lexicographic: covered unresolved words, negative total group length,
negative group count. Only candidates supplied by acoustic checks are selectable.
"""
import json


def compatible(left, right):
    return (left['word_indices'][-1] < right['word_indices'][0]
            and left['interval_s'][1] <= right['interval_s'][0] + 1e-7)


def objective(phrases, words):
    return (sum(not words[i]['automatic_eligible'] for p in phrases for i in p['word_indices']),
            -sum(len(p['word_indices']) for p in phrases), -len(phrases))


def select_phrases(candidates, words, *, greedy=False):
    if greedy:
        # Controlled ablation: the former shortest-first rule.
        selected, covered = [], set()
        for p in sorted(candidates, key=lambda p: (len(p['word_indices']), p['word_indices'][0])):
            unknown = {i for i in p['word_indices'] if not words[i]['automatic_eligible']}
            if unknown - covered:
                selected.append(p)
                covered.update(unknown)
        return selected
    # Canonical ordering also fixes ties independently of candidate enumeration.
    candidates = sorted(candidates, key=lambda p: (
        p['word_indices'][-1], p['word_indices'][0], tuple(p['interval_s']),
        json.dumps(p, sort_keys=True, separators=(',', ':'))))
    best = []
    for k, p in enumerate(candidates):
        weight = objective([p], words)
        options = [(weight, (k,))]
        for j, q in enumerate(candidates[:k]):
            if compatible(q, p):
                score, path = best[j]
                options.append((tuple(a + b for a, b in zip(score, weight)), path + (k,)))
        best.append(max(options, key=lambda x: x[0]))
    score, path = max([((0, 0, 0), ())] + best, key=lambda x: x[0])
    return [candidates[k] for k in path]
