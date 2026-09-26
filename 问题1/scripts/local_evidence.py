"""Add coherent local exact-match runs without lowering the whole-text threshold.

Inputs contain automatic evidence only. No sample IDs, labels, review records,
dictionary exceptions, or human timestamps take part in the decision.
"""
import copy


def local_candidates(result, config):
    output = copy.deepcopy(result)
    words = output['words']
    runs, current, previous_hypothesis = [], [], None
    for i, word in enumerate(words):
        parts = word['recognition_options']
        indices = [p['hypothesis_options'][0] for p in parts
                   if p['exact'] and p['stable'] and len(p['hypothesis_options']) == 1
                   and not p['unmatched_possible']]
        valid = (bool(parts) and len(indices) == len(parts)
                 and indices == list(range(indices[0], indices[0] + len(indices)))
                 and word['diagnostic_interval_s'] is not None
                 and word['boundary_disagreement_s'] is not None
                 and word['boundary_disagreement_s'] <= config['boundary_agreement_s'])
        continuation = (valid and current and indices[0] == previous_hypothesis + 1
                        and words[current[-1]]['diagnostic_interval_s'][1]
                        <= word['diagnostic_interval_s'][0] + 1e-9)
        if current and not continuation:
            runs.append(current); current = []
        if valid:
            current.append(i); previous_hypothesis = indices[-1]
        else:
            previous_hypothesis = None
    if current:
        runs.append(current)
    support = []
    if output['sample_status'] != 'no_signal':
        for run in runs:
            hypothesis_indices = [p['hypothesis_options'][0] for i in run
                                  for p in words[i]['recognition_options']]
            anchors = {output['recognition'][j]['text'] for j in hypothesis_indices
                       if len(output['recognition'][j]['text']) >= config['informative_min_chars']
                       and output['recognition'][j]['text'] not in config['function_words']}
            if len(anchors) < config['minimum_informative_anchors']:
                continue
            run_id = len(support)
            support.append({'word_indices': run, 'hypothesis_indices': hypothesis_indices,
                            'distinct_informative_anchors': sorted(anchors),
                            'meaning': 'Local lexical/acoustic consistency, not calibrated correctness.'})
            for i in run:
                if not words[i]['automatic_eligible']:
                    words[i].update(automatic_eligible=True,
                        interval_s=words[i]['diagnostic_interval_s'],
                        status='local_coherent_candidate', local_support_run=run_id)
    output['local_support_runs'] = support
    output['automatic_eligible_words'] = sum(w['automatic_eligible'] for w in words)
    output['new_local_candidate_words'] = sum(w['status'] == 'local_coherent_candidate' for w in words)
    output['decision_policy'] = 'frozen_selective_baseline_plus_coherent_local_runs_v1'
    output['whole_text_content_confirmed'] = False
    # Navigation envelopes describe the prior policy and are not alignment claims.
    output['context_envelopes_policy'] = 'original_frozen_baseline_navigation_only'
    return output


def corroborated_local_candidates(result, config):
    """For whole-text rejection, require two runs with disjoint anchor vocabularies.

    Runs are disjoint and internally contiguous in both original and recognized
    token order. This is contextual corroboration, not independent ASR evidence.
    """
    output = local_candidates(result, config)
    runs = output['local_support_runs']
    pairs = [[i, j] for i in range(len(runs)) for j in range(i + 1, len(runs))
             if set(runs[i]['distinct_informative_anchors']).isdisjoint(
                 runs[j]['distinct_informative_anchors'])]
    corroborated = {index for pair in pairs for index in pair}
    if result['sample_status'] != 'supported':
        for i, word in enumerate(output['words']):
            if word['status'] == 'local_coherent_candidate' and word['local_support_run'] not in corroborated:
                output['words'][i] = copy.deepcopy(result['words'][i])
    output['corroborating_run_pairs'] = pairs
    output['automatic_eligible_words'] = sum(w['automatic_eligible'] for w in output['words'])
    output['new_local_candidate_words'] = sum(w['status'] == 'local_coherent_candidate' for w in output['words'])
    output['decision_policy'] = 'frozen_selective_baseline_plus_corroborated_local_runs_v2'
    return output
