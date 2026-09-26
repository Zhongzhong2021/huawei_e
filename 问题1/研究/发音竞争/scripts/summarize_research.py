"""Report experiments and provisional coverage; never rewrite stable predictions."""
import os,sys,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];Q=Path(os.environ.get('Q1_PROJECT',R.parents[1]))
sys.path.insert(0,str(Q/'scripts'))
from common import read,write,digest

def main():
    stable=read(R/'data/base_reference.json');sources={s['sample_id']:s for s in read(Q/'data/inputs.json')['samples']}
    modes={};candidates=[]
    for mode in ['canonical','contextual','lexicon']:
        out=R/'data'/mode;summary=read(out/'summary.json');injected=read(out/'injected.json');union=set();new=set();positive=[]
        for p in read(out/'identity.json')['rows']:
            if not p['accepted']:continue
            sid=p['sample_id'];base=read(Q/'data/multigranular_alignment'/(sid+'.json'))
            existing={i for i,w in enumerate(base['words']) if w['automatic_eligible']}|set(base['phrase_only_words'])
            union.update((sid,i) for i in p['word_indices']);novel=[i for i in p['word_indices'] if i not in existing]
            new.update((sid,i) for i in novel);positive.append(p)
            if mode=='contextual' and novel:
                candidates.append({'sample_id':sid,'source_sha256':sources[sid]['sha256'],'source_path':sources[sid]['path'],
                    'word_indices':p['word_indices'],'new_word_indices':novel,'reference_words':p['reference_words'],
                    'candidate_interval_s':p['interval_s'],'evidence':p['evidence'],
                    'status':'research_candidate_pending_independent_evaluation','human_review':{'content':'','timing':'','note':''},
                    'stable_results_modified':False,'internal_word_times_assigned':False})
        modes[mode]={'accepted_groups':len(positive),'union_covered_words':len(union),'provisional_new_words_vs_stable':len(new),
                     'shifted_accepted':summary['shifted']['accepted'],'shifted_trials':summary['shifted']['proposals'],
                     'corrupted_accepted':summary['text_corruption']['accepted'],'corrupted_trials':summary['text_corruption']['proposals'],
                     'injected_accepted':injected['accepted'],'injected_trials':injected['proposals']}
    candidates.sort(key=lambda p:(p['source_sha256'],p['word_indices']))
    write(R/'data/review_candidates.json',{'schema':'q1-independent-phonetic-candidate-evaluation-v1','use':'Independent evaluation only. Never read by proposal, inference or scoring code.','candidates':candidates})
    write(R/'data/research_summary.json',{'status':'research_complete_not_promoted','modes':modes,'protocol_sha256':digest(R/'protocol.json'),
          'stable_archive_sha256':stable['archive_sha256'],'independent_accuracy_measured':False,
          'stable_outputs_changed':False,'new_review_groups':len(candidates),
          'decision':'Contextual pronunciation is the preferred next evaluation branch; no automatic merge into stable predictions. Fixed control behavior does not establish natural speech precision.',
          'development_limits':'Per-phone open-score normalization was introduced during development after whole-sequence scoring rejected all proposals. Shift and corruption controls informed the design; injected different-video controls were specified after freezing v3. No threshold sweep was performed.'})
    print(modes);print('Pending evaluation groups:',len(candidates))
if __name__=='__main__':main()
