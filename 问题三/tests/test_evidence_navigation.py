"""Check evidence identity, conservative localization and actual video clocks."""
import hashlib,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/evidence_navigation'))
from build import locate
from multigranular_alignment import combine

def read(p):return json.loads(p.read_text())
def records(p):return [json.loads(l) for l in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def test_punctuation_and_disjoint_words_do_not_get_invented_times():
    ref=[{'char_span':[0,3]},{'char_span':[4,7]}]
    result={'words':[{'automatic_eligible':True,'interval_s':[0,1]},{'automatic_eligible':True,'interval_s':[2,3]}],'phrases':[]}
    assert locate([[3,4]],ref,result)['interval_s'] is None
    assert locate([[0,7]],ref,result)['interval_s'] is None
    result['phrases']=[{'word_indices':[0,1],'interval_s':[0,3]}]
    value=locate([[4,6]],ref,{**result,'words':[{'automatic_eligible':False}]*2})
    assert value['status']=='phrase_candidate' and value['internal_word_times_assigned'] is False

def test_acoustic_disagreement_is_not_accepted_as_word():
    base={'words':[{'automatic_eligible':False,'interval_s':None,'status':'unresolved'}]}
    support={'words':[{'exact_unique':True,'atomic_word_time':True,'interval_s':[1,1.4]}]}
    run={'word_indices':[0],'anchors':['ALPHA','BETA'],'run_index':0,'contexts':[[{'interval_s':[1,1.4]}],[{'interval_s':[2,2.4]}]]}
    cfg={'boundary_agreement_s':.2,'minimum_distinct_anchors':2,'minimum_phrase_words':2,'maximum_phrase_words':8}
    out=combine([{}],base,support,[run],cfg,[0,5])
    assert not out['words'][0]['automatic_eligible'] and not out['phrases']
    assert not base['words'][0]['automatic_eligible']

def test_navigation_preserves_every_original_attribution_and_frame_clock():
    folder=ROOT/'results/evidence_navigation';nav=records(folder/'navigation.jsonl');exp=records(ROOT/'results/uncertainty/special_explanations.jsonl');inputs={s['sample_id']:s for s in read(folder/'inputs.json')['samples']}
    assert len(nav)==len(exp)==20
    count=0
    for n,e in zip(nav,exp):
        assert n['id']==e['id'];source=inputs[n['id']]
        for mod in 'TAV':
            assert len(n['top_evidence'][mod])==len(e['top_evidence'][mod])
            for actual,old in zip(n['top_evidence'][mod],e['top_evidence'][mod]):
                count+=1
                for k,v in old.items():assert actual[k]==v
                t=actual['interval_s']
                if t:assert source['audio']['start_s']<=t[0]<t[1]<=source['audio']['end_s']
                if 'video_frame' in actual:
                    frame=actual['video_frame'];pts=source['video_frame_pts_s'][frame['frame_index']]
                    assert pts==frame['pts_s'] and t[0]<=pts<t[1]
                    assert sha(folder/frame['path'])==frame['sha256']
                    assert not frame['original_feature_frame_verified']
        assert not n['source_feature_timestamps_available']
    summary=read(folder/'summary.json');assert count==summary['top_evidence_count']
    for p,h in summary['frozen_result_sha256'].items():assert sha(ROOT/p)==h

def test_reused_q1_policy_is_byte_identical_to_recorded_version():
    folder=ROOT/'tools/evidence_navigation'
    for p,h in read(folder/'reuse.json')['unmodified_helpers_sha256'].items():assert sha(folder/p)==h

def test_content_flags_are_diagnostics_and_preserve_media_identity():
    from check_correspondence import coverage
    assert coverage('alpha beta gamma','alpha beta gamma') == 1
    assert coverage('alpha beta gamma','delta epsilon') == 0
    folder=ROOT/'results/evidence_navigation'
    checks=read(folder/'media_consistency.json')['samples']
    inputs={r['sample_id']:r for r in read(folder/'inputs.json')['samples']}
    nav={r['id']:r for r in records(folder/'navigation.jsonl')}
    assert len(checks)==20
    for r in checks:
        assert r['video_sha256']==inputs[r['id']]['sha256']==nav[r['id']]['source_sha256']
        assert not r['content_verified']
        low=all(v<.5 for v in r['ordered_lexical_coverage'].values())
        assert low==(r['content_status']=='low_cross_recognizer_support')
