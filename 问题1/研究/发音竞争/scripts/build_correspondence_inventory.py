"""Account for every original word and preserve synchronous visual observations."""
import sys
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from build_enhancement import Q,R
from common import read,write,digest


def word_state(word,phrase_rows,source,sample_status):
    if source['zero_audio']:return 'no_audio_signal'
    if word['automatic_eligible']:return 'word_candidate'
    if phrase_rows:return 'phrase_candidate'
    if word['status']=='cross_model_time_conflict':return 'time_conflict'
    if word['status']=='boundary_disagreement':return 'unstable_boundary'
    if word['status']=='approximate_lexical_match':return 'approximate_text_only'
    if word['status']=='unmatched_or_ambiguous':return 'unmatched_or_ambiguous'
    if word['status']=='sample_support_insufficient':
        return {'no_informative_support':'no_informative_anchor',
                'insufficient_evidence':'insufficient_sample_support'}[sample_status]
    raise ValueError('Unaccounted word status: '+word['status'])


def build():
    inputs=read(Q/'data/inputs.json');enhance=read(R/'data/enhancement.json')
    supplements={s['sample_id']:s for s in enhance['samples']}
    omissions=read(R/'data/local_omissions.json');lookup=defaultdict(list)
    for n,row in enumerate(omissions['rows']):lookup[(row['sample_id'],row['word_index'])].append(n)
    states=Counter();visual=Counter();samples=[];source_hashes={};tested=set();negative=set();pending=set()
    for source in inputs['samples']:
        sid=source['sample_id'];base_path=Q/'data/multigranular_alignment'/(sid+'.json')
        automatic_path=Q/'data/automatic_alignment'/(sid+'.json')
        feature_path=Q/'data/features'/(sid+'.npz')
        base=read(base_path);automatic=read(automatic_path)
        assert base['source_sha256']==automatic['source_sha256']==source['sha256']
        source_hashes[sid]={'base':digest(base_path),'automatic':digest(automatic_path),'features':digest(feature_path)}
        rows=[]
        for i,w in enumerate(base['words']):
            phrases=[j for j,p in enumerate(base['phrases']) if i in p['word_indices']]
            extra=[j for j,p in enumerate(supplements[sid]['phrases']) if i in p['new_word_indices']]
            tests=lookup[(sid,i)];supported=[j for j in tests if omissions['rows'][j]['local_omission_supported']]
            state=word_state(w,phrases,source,automatic['sample_status']);states[state]+=1
            if state not in {'word_candidate','phrase_candidate'}:pending.add((sid,i))
            if tests:tested.add((sid,i))
            if supported:negative.add((sid,i))
            rows.append({'word_index':i,'text':w['text'],'state':state,'original_status':w['status'],
                         'word_interval_s':w['interval_s'] if w['automatic_eligible'] else None,
                         'phrase_rows':phrases,'supplemental_phrase_rows':extra,
                         'local_omission_test_rows':tests,'local_omission_support_rows':supported,
                         'global_absence':'no_audio_signal' if source['zero_audio'] else 'not_established',
                         'evidence_conflict':bool(supported and (w['automatic_eligible'] or phrases or extra))})
        with np.load(feature_path) as a:
            v={'sampled_frames':len(a['scene']),'scene_dimensions':int(a['scene'].shape[1]),
               'frames_with_face_detections':int((a['face_counts']>0).sum()),
               'frames_without_face_detections':int((a['face_counts']==0).sum()),
               'face_events':len(a['faces']),'face_dimensions':int(a['faces'].shape[1]),
               'multiple_face_frames':int((a['face_counts']>1).sum()),
               'speaker_identity_assigned':False,'relation':'synchronous_observation_only',
               'no_detection_means_face_absent':False}
        for key in ['sampled_frames','frames_with_face_detections','frames_without_face_detections','face_events','multiple_face_frames']:visual[key]+=v[key]
        visual['samples_without_face_detections']+=int(v['face_events']==0)
        samples.append({'sample_id':sid,'source_sha256':source['sha256'],'sample_status':automatic['sample_status'],
                        'words':rows,'visual':v})
    summary={'samples':len(samples),'words':sum(states.values()),'states':dict(sorted(states.items())),
             'unresolved_in_base':len(pending),'local_omission_tested_words':len(tested),
             'local_omission_supported_words':len(negative),'unresolved_with_local_omission_tests':len(pending&tested),
             'unresolved_with_local_omission_support':len(pending&negative),'visual':dict(visual),
             'nonzero_audio_words_declared_globally_absent':0}
    return {'schema':'q1-complete-correspondence-inventory-v1','human_data_used':False,
            'primary_states_describe_available_evidence_not_ground_truth':True,
            'visual_features_preserved':True,'base_results_modified':False,
            'script_sha256':digest(__file__),'inputs_sha256':digest(Q/'data/inputs.json'),
            'enhancement_sha256':digest(R/'data/enhancement.json'),'omission_sha256':digest(R/'data/local_omissions.json'),
            'upstream_sha256':source_hashes,'summary':summary,'samples':samples}

if __name__=='__main__':
    result=build();write(R/'data/correspondence_inventory.json',result);print(result['summary'])
