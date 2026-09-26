"""Collect local phonetic outputs and deterministically construct the new policy."""
import argparse
from pathlib import Path
from collections import Counter
from common import read, write, digest
from partial_acoustic_mapping import map_partial_word_tier
from multigranular_alignment import combine

ROOT = Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--local',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    prep=read(args.local/'preparation.json')
    execution=read(args.local/'execution.json')
    assert execution['returncode']==0 and execution['preparation_sha256']==digest(args.local/'preparation.json')
    cfg=read(ROOT/'config/whisper_alignment.json')
    inputs=read(ROOT/'data/inputs.json')
    source={s['sample_id']:s for s in inputs['samples']}
    args.output.mkdir(parents=True,exist_ok=False)
    raw={p.stem:p for p in (args.local/'raw').rglob('*.json')}
    jobs={};raw_hashes={}
    for job in prep['jobs']:
        rows=[]; status='no_tool_output'
        if job['file_id'] in raw:
            path=raw[job['file_id']]; tiers=read(path)['tiers']
            entries=[v['entries'] for k,v in tiers.items() if k=='words' or k.endswith(' - words')]
            assert len(entries)==1
            rows,status=map_partial_word_tier(job['source_words'],entries[0],job['source_offset_s'],job['duration_s'])
            raw_hashes[str(path.relative_to(args.local))]=digest(path)
        jobs[job['file_id']]={**job,'mapped_words':rows,'status':status}
    summary=[]; records=[]
    for s in prep['samples']:
        sid=s['sample_id'];item=source[sid]
        baseline=read(ROOT/'data/local_evidence_corroborated'/(sid+'.json'))
        meta=read(ROOT/'data/features'/(sid+'.json'))
        assert item['sha256']==s['source_sha256']==baseline['source_sha256']
        assert digest(ROOT/'data/features'/(sid+'.json'))==s['metadata_sha256']
        runs=[]
        for index,run in enumerate(s['support']['runs']):
            pair=sorted([jobs[j] for j in s['jobs'] if jobs[j]['run_index']==index],key=lambda j:j['context_s'])
            assert [j['context_s'] for j in pair]==[0.5,1.0]
            runs.append({**run,'run_index':index,'contexts':[j['mapped_words'] for j in pair],
                         'job_ids':[j['file_id'] for j in pair],
                         'distinct_crops':len({tuple(j['sample_span_16k']) for j in pair})})
        result=combine(meta['words'],baseline,s['support'],runs,cfg,
                       [item['audio']['start_s'],item['audio']['end_s']],item['zero_audio'])
        result.update(sample_id=sid,source_sha256=item['sha256'],human_data_used=False,
                      baseline_result_sha256=digest(ROOT/'data/local_evidence_corroborated'/(sid+'.json')),
                      decision_policy='unprompted_asr_local_phonetic_multigranular_ordered_selection_v3')
        write(args.output/(sid+'.json'),result)
        records.append({'sample_id':sid,'source_sha256':item['sha256'],'support':s['support'],'runs':runs})
        counts=Counter(d['status'] for d in result['diagnostics'])
        summary.append({'sample_id':sid,'words':len(meta['words']), 'baseline_candidates':result['baseline_candidates'],
                        'word_candidates':sum(w['automatic_eligible'] for w in result['words']),
                        'added_words':result['added_words'],'phrase_candidates':len(result['phrases']),
                        'conflicting_baseline_words':result['conflicting_baseline_words'],
                        'phrase_only_words':len(result['phrase_only_words']),
                        'unique_exact_asr_words':s['support']['exact_unique_words'],
                        'supported_runs':len(runs),'diagnostic_counts':dict(counts)})
    write(args.output/'acoustic_evidence.json',{'records':records,'jobs':list(jobs.values()),'raw_sha256':raw_hashes,
         'preparation_sha256':digest(args.local/'preparation.json'),'execution_sha256':digest(args.local/'execution.json')})
    totals={key:sum(s[key] for s in summary) for key in ['words','baseline_candidates','word_candidates','added_words','conflicting_baseline_words','phrase_candidates','phrase_only_words','unique_exact_asr_words','supported_runs']}
    report={'totals':totals,'samples':summary,'human_data_used':False,'scope':'Coverage and model corroboration; no ground-truth accuracy claim.'}
    write(args.output/'summary.json',report)
    write(args.output/'manifest.json',{'schema':'q1-multigranular-v1','complete':len(summary)==100,
          'config_sha256':digest(ROOT/'config/whisper_alignment.json'),
          'scripts':{n:digest(ROOT/'scripts'/n) for n in ['multigranular_alignment.py','phrase_selection.py','whisper_support.py','multigranular_reader.py','build_multigranular_alignment.py']},
          'evidence_sha256':digest(args.output/'acoustic_evidence.json'),
          'samples':[{ 'sample_id':s['sample_id'],'sha256':digest(args.output/(s['sample_id']+'.json'))} for s in summary]})
    print(totals)

if __name__=='__main__':main()
