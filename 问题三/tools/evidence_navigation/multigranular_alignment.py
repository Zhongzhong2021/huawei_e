"""Cross-model temporal corroboration and explicit coarse relations.

Reference identity, evidence and decision logic are separate. No model confidence
is interpreted as correctness probability; no interval is assigned by interpolation.
"""
import copy
import math
from phrase_selection import select_phrases


def valid(interval, support):
    return bool(interval and all(math.isfinite(v) for v in interval)
                and support[0] <= interval[0] < interval[1] <= support[1]+1e-7)


def distance(a,b):
    return max(abs(a[k]-b[k]) for k in (0,1))


def combine(reference, baseline, support, local_runs, cfg, signal_support, no_signal=False, *, phrase_policy="optimal"):
    """Returns an additional policy; frozen baseline is never modified in place.

    local_runs contain two context outputs in a fixed order, 0.5 then 1 second.
    At least two distinct informative anchors are required per ASR run. The final
    reader reports this as model-supported candidate evidence, not human truth.
    """
    if phrase_policy not in {"optimal", "deferred_greedy", "early_greedy"}:
        raise ValueError("Unknown phrase selection policy")
    output = copy.deepcopy(baseline)
    proposed, phrases, diagnostics = {}, [], []
    tolerance = cfg['boundary_agreement_s']
    if no_signal:
        assert not any(w['automatic_eligible'] for w in output['words'])
        return {'words':output['words'],'phrases':[], 'diagnostics':[], 'added_words':0,
                'phrase_only_words':[], 'baseline_candidates':0,'conflicting_baseline_words':0}
    for run in local_runs:
        indices = run['word_indices']
        a,b = run['contexts']
        if not a or not b or len(a)!=len(indices) or len(b)!=len(indices):
            diagnostics.append({'run_index':run['run_index'],'status':'incomplete_acoustic_output'})
            continue
        if len(run['anchors']) < cfg['minimum_distinct_anchors']:
            continue
        for j,i in enumerate(indices):
            w = support['words'][i]
            ta,tb,tw = a[j]['interval_s'],b[j]['interval_s'],w['interval_s']
            acoustic_ok = valid(ta,signal_support) and valid(tb,signal_support)
            evidence = {'run_index':run['run_index'], 'word_index':i,
                        'asr_interval_s':tw,'mfa_context_05_s':ta,'mfa_context_10_s':tb}
            ok = bool(w['exact_unique'] and w['atomic_word_time'] and acoustic_ok
                      and valid(tw,signal_support) and distance(ta,tb)<=tolerance+1e-9
                      and distance(ta,tw)<=tolerance+1e-9)
            if baseline['words'][i]['automatic_eligible']:
                old = baseline['words'][i]['interval_s']
                evidence['status'] = ('baseline_agreement' if ok and distance(ta,old)<=tolerance+1e-9
                                      else 'baseline_cross_model_disagreement' if ok else 'baseline_without_new_support')
                if evidence['status']=='baseline_cross_model_disagreement':
                    output['words'][i].update(automatic_eligible=False,interval_s=None,
                        status='cross_model_time_conflict',competing_intervals_s=[old,ta])
            else:
                evidence['status'] = 'word_proposal' if ok else 'word_evidence_insufficient'
                if ok: proposed[i] = ta
            diagnostics.append(evidence)
        # A phrase must have continuous exact ASR identity, no dropped words,
        # valid phonetic paths, and corroborated outer boundaries. Internal
        # word timestamps are not emitted for this coarse relation.
        for length in range(cfg['minimum_phrase_words'], min(cfg['maximum_phrase_words'],len(indices))+1):
            for start in range(len(indices)-length+1):
                js = list(range(start,start+length)); target = [indices[j] for j in js]
                if phrase_policy == 'early_greedy' and all(output['words'][i]['automatic_eligible'] or i in proposed for i in target):
                    continue
                wa,wb = [a[j]['interval_s'] for j in js],[b[j]['interval_s'] for j in js]
                if not all(valid(t,signal_support) for t in wa+wb): continue
                if any(x[1]>y[0]+1e-7 for times in [wa,wb] for x,y in zip(times,times[1:])): continue
                first,last = support['words'][target[0]],support['words'][target[-1]]
                if not first['atomic_word_time'] or not last['atomic_word_time']: continue
                asr = [first['interval_s'][0],last['interval_s'][1]]
                ta,tb = [wa[0][0],wa[-1][1]],[wb[0][0],wb[-1][1]]
                if distance(ta,tb)>tolerance+1e-9 or distance(ta,asr)>tolerance+1e-9: continue
                phrases.append({'word_indices':target,'interval_s':ta,'run_index':run['run_index'],
                                'asr_interval_s':asr,'second_context_interval_s':tb,
                                'status':'acoustically_supported_phrase_candidate',
                                'internal_word_times_assigned':False})
    # Keep unchallenged baseline timestamps. Contradictory supported locations
    # remain explicit alternatives; new proposals cannot cross accepted words.
    combined = {i:w['interval_s'] for i,w in enumerate(output['words']) if w['automatic_eligible']}
    combined.update(proposed)
    bad = set()
    active = sorted(combined)
    for left,right in zip(active,active[1:]):
        if combined[left][1] > combined[right][0]+1e-7:
            bad.update(i for i in [left,right] if i in proposed)
    for i,t in proposed.items():
        if i in bad:
            diagnostics.append({'word_index':i,'status':'new_word_time_conflict'})
        else:
            output['words'][i].update(automatic_eligible=True,interval_s=t,
                status='whisper_local_phonetic_candidate',human_verified=False)
    eligible = []
    # Word decisions are final before any coarse candidate is discarded.
    for phrase in phrases:
        target = phrase['word_indices']
        if any(output['words'][i]['status']=='cross_model_time_conflict' for i in target):continue
        unknown = {i for i in target if not output['words'][i]['automatic_eligible']}
        if not unknown: continue
        t = phrase['interval_s']
        # Also enforce consistency with accepted words inside and outside the group.
        conflict=False
        for i,w in enumerate(output['words']):
            if not w['automatic_eligible']: continue
            a,b=w['interval_s']
            if ((i<target[0] and b>t[0]+1e-7) or (i>target[-1] and a<t[1]-1e-7)
                or (i in target and (a<t[0]-tolerance or b>t[1]+tolerance))):conflict=True;break
        if conflict: continue
        eligible.append(phrase)
    kept = select_phrases(eligible, output['words'], greedy=phrase_policy != 'optimal')
    covered = {i for p in kept for i in p['word_indices'] if not output['words'][i]['automatic_eligible']}
    return {'words':output['words'],'phrases':kept,'diagnostics':diagnostics,
            'added_words':sum(w['status']=='whisper_local_phonetic_candidate' for w in output['words']),
            'conflicting_baseline_words':sum(w['status']=='cross_model_time_conflict' for w in output['words']),
            'phrase_only_words':sorted(covered),
            'baseline_candidates':sum(w['automatic_eligible'] for w in baseline['words'])}
