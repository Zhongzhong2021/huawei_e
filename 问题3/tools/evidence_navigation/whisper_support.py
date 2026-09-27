"""Unique lexical support from unprompted ASR; no sample identity or review input."""
import math
from math_features import words as tokenize
from selective_alignment import optimal_relations


def lexical_support(reference_words, recognized, cfg, start_s, end_s):
    tokens = []
    for index, entry in enumerate(recognized):
        parts = [p for word in tokenize(entry['text']) for p in word['parts']]
        for part in parts:
            tokens.append({'text': part, 'raw_row': index, 'interval_s': entry['interval_s']})
    ref = [p for word in reference_words for p in word['parts']]
    dp = optimal_relations(ref, [t['text'] for t in tokens], cfg)
    output, cursor = [], 0
    for i, word in enumerate(reference_words):
        links = dp['reference_tokens'][cursor:cursor+len(word['parts'])]
        cursor += len(word['parts'])
        exact = bool(links) and all(p['stable'] and p['exact'] for p in links)
        positions = [p['hypothesis_options'][0] for p in links] if exact else []
        exact = exact and positions == list(range(positions[0], positions[0]+len(positions)))
        interval = [tokens[positions[0]]['interval_s'][0], tokens[positions[-1]]['interval_s'][1]] if exact else None
        valid = bool(interval and all(math.isfinite(t) for t in interval) and start_s <= interval[0] < interval[1] <= end_s+1e-6)
        # A raw ASR token may span multiple normalized words. Its internal times
        # cannot be divided into invented word boundaries.
        atomic = bool(exact and (positions[0] == 0 or tokens[positions[0]-1]['raw_row'] != tokens[positions[0]]['raw_row'])
                      and (positions[-1]+1 == len(tokens) or tokens[positions[-1]+1]['raw_row'] != tokens[positions[-1]]['raw_row']))
        output.append({'word_index': i, 'exact_unique': bool(exact), 'valid_time': valid,
                       'atomic_word_time': atomic, 'token_rows': positions,
                       'raw_rows': sorted({tokens[j]['raw_row'] for j in positions}),
                       'interval_s': interval if valid else None})
    anchors = lambda run: sorted({p for i in run for p in reference_words[i]['parts']
                        if len(p) >= cfg['informative_min_chars'] and p not in cfg['function_words']})
    runs, run = [], []
    for i, row in enumerate(output):
        valid = row['exact_unique'] and row['valid_time']
        contiguous = (valid and run and row['token_rows'][0] == output[run[-1]]['token_rows'][-1]+1
                      and row['interval_s'][0] >= output[run[-1]]['interval_s'][0]
                      and row['interval_s'][1] >= output[run[-1]]['interval_s'][1])
        if run and not contiguous:
            runs.append(run); run = []
        if valid: run.append(i)
    if run: runs.append(run)
    # Retain every support candidate, but expose only runs meeting the inherited
    # informative-anchor requirement as inputs to acoustic verification.
    accepted = []
    for run in runs:
        if len(anchors(run)) < cfg['minimum_informative_anchors']:
            continue
        # Boundaries must coincide with whole ASR token boundaries.
        if not output[run[0]]['atomic_word_time'] or not output[run[-1]]['atomic_word_time']:
            continue
        accepted.append({'word_indices': run, 'anchors': anchors(run),
                         'interval_s': [output[run[0]]['interval_s'][0], output[run[-1]]['interval_s'][1]]})
    vocabulary = {p for w in reference_words for p in w['parts']
                  if len(p)>=cfg['informative_min_chars'] and p not in cfg['function_words']}
    matched = {p for i,w in enumerate(reference_words) if output[i]['exact_unique'] for p in w['parts']
               if p in vocabulary}
    coverage = len(matched)/len(vocabulary) if vocabulary else 0.0
    supported = len(matched)>=cfg['minimum_informative_anchors'] and coverage>=cfg['minimum_informative_coverage']
    pairs = [[i,j] for i in range(len(accepted)) for j in range(i+1,len(accepted))
             if set(accepted[i]['anchors']).isdisjoint(accepted[j]['anchors'])]
    permitted = {i for pair in pairs for i in pair}
    if not supported:
        accepted = [run for i,run in enumerate(accepted) if i in permitted]
    return {'words': output, 'runs': accepted, 'edit_cost': dp['edit_cost'],
            'informative_coverage':coverage, 'sample_supported':supported,
            'exact_unique_words': sum(r['exact_unique'] for r in output)}
