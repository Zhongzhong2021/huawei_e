"""Full-corpus ablation and constructed lexical stress controls, without reviews."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from common import read, write, digest
from whisper_support import lexical_support
from math_features import words as tokenize
from selective_alignment import align_sample
from local_evidence import corroborated_local_candidates
from multigranular_reader import Dataset

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--recognition',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stress',action='store_true')
    args=p.parse_args()
    ds=Dataset(ROOT)
    cfg=read(ROOT/'config/automatic_alignment.json')
    baseline_exact=0;new_exact=0;asr_swap=0;total=0;word_count=0;coarse_only=0
    sample_rows=[];diagnostics=Counter();sources=ds.records
    metadata={};recognition={}
    for sid in sources:
        sample=ds.sample(sid);item=sources[sid]
        new=read(args.recognition/(sid+'.json'));old=read(ROOT/'data/automatic_alignment'/(sid+'.json'))
        assert new['source_sha256']==item['sha256']
        metadata[sid]=sample.metadata;recognition[sid]=new
        support=lexical_support(sample.metadata['words'],new['words'],cfg,item['audio']['start_s'],item['audio']['end_s'])
        norm=[]
        for r in new['words']:
            for w in tokenize(r['text']):
                norm.extend({'text':part,'interval_s':r['interval_s']} for part in w['parts'])
        swap=align_sample(sample.metadata['words'],norm,sample.arrays['word_intervals'],cfg,no_signal=item['zero_audio'])
        swap['recognition']=norm
        swap=corroborated_local_candidates(swap,cfg)
        old_exact=sum(bool(w['recognition_options']) and all(p['exact'] for p in w['recognition_options']) for w in old['words'])
        baseline_exact+=old_exact;new_exact+=support['exact_unique_words'];asr_swap+=swap['automatic_eligible_words']
        result=sample.automatic
        n=sum(w['automatic_eligible'] for w in result['words'])
        word_count+=n;coarse_only+=len(result['phrase_only_words']);total+=len(result['words'])
        diagnostics.update(d['status'] for d in result['diagnostics'])
        for i,w in enumerate(result['words']):
            relation=sample.word(i)
            assert (relation['relation'] is not None)==w['automatic_eligible']
            if relation['relation'] is not None:assert len(relation['relation']['audio']['rows'])>0
        for i in range(len(result['phrases'])):
            view=sample.phrase(i)
            assert view['internal_word_times_assigned'] is False
            assert len(view['relation']['audio']['rows'])>0
        if item['zero_audio']:assert n==0 and not result['phrases']
        sample_rows.append({'sample_id':sid,'words':len(result['words']),
            'baseline_candidates':result['baseline_candidates'],'ctc_unique_exact':old_exact,
            'whisper_unique_exact':support['exact_unique_words'],'asr_swap_same_rule_candidates':swap['automatic_eligible_words'],
            'enhanced_word_candidates':n,'phrase_only_words':len(result['phrase_only_words']),
            'phrase_candidates':len(result['phrases'])})
    stress=[]
    if args.stress:
        for sid,item in sources.items():
            if item['zero_audio']:continue
            for ref_id,other in sources.items():
                if item['video_id']==other['video_id']:continue
                support=lexical_support(metadata[ref_id]['words'],recognition[sid]['words'],cfg,item['audio']['start_s'],item['audio']['end_s'])
                if support['runs']:
                    stress.append({'audio_sample_id':sid,'reference_sample_id':ref_id,
                                   'supported_runs':support['runs'],'informative_coverage':support['informative_coverage']})
        comparisons=sum(not a['zero_audio'] and a['video_id']!=b['video_id'] for a in sources.values() for b in sources.values())
    else:comparisons=None
    report={'samples':len(sample_rows),'words':total,
        'ablation':{'baseline_unique_exact_lexical_words':baseline_exact,'whisper_unique_exact_lexical_words':new_exact,
                    'baseline_candidates':sum(r['baseline_candidates'] for r in sample_rows),
                    'whisper_substitution_with_original_rule_candidates':asr_swap,
                    'enhanced_word_candidates':word_count,'phrase_only_words':coarse_only,
                    'word_or_phrase_covered_words':word_count+coarse_only},
        'diagnostic_counts':dict(diagnostics),'sample_results':sample_rows,
        'stress':{'comparisons':comparisons,'pairs_with_lexical_proposals':len(stress) if args.stress else None,
                  'cases':stress,'scope':'Constructed different-video lexical screen. Proposals are not accepted acoustic relations; shared real phrases may occur.'},
        'inference_readers_verified':True,'human_data_used':False,'independent_accuracy_measured':False,
        'recognition_manifest_sha256':digest(args.recognition/'manifest.json'),
        'enhanced_manifest_sha256':digest(ROOT/'data/multigranular_alignment/manifest.json')}
    write(args.output,report)
    print({k:v for k,v in report.items() if k not in ['sample_results','stress']})
    print({'stress_comparisons':comparisons,'stress_lexical_proposals':len(stress)})

if __name__=='__main__':main()
