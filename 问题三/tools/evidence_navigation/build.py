"""Create conservative evidence navigation without changing feature attribution."""
import argparse,copy,json,subprocess
from collections import Counter
from pathlib import Path
from common import read,write,digest
from partial_acoustic_mapping import map_partial_word_tier
from multigranular_alignment import combine
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def locate(spans,reference,result):
    indices=[i for i,w in enumerate(reference) if any(max(a,w['char_span'][0])<min(b,w['char_span'][1]) for a,b in spans)]
    if not indices:return {'status':'no_lexical_span','interval_s':None,'word_indices':[]}
    if len(indices)==1 and result['words'][indices[0]]['automatic_eligible']:
        return {'status':'word_candidate','interval_s':result['words'][indices[0]]['interval_s'],'word_indices':indices}
    for phrase in result['phrases']:
        if set(indices)<=set(phrase['word_indices']):
            return {'status':'phrase_candidate','interval_s':phrase['interval_s'],'word_indices':indices,'phrase_word_indices':phrase['word_indices'],'internal_word_times_assigned':False}
    # No hull across separated or unsupported words.
    return {'status':'insufficient_support','interval_s':None,'word_indices':indices}

def main():
 p=argparse.ArgumentParser();p.add_argument('--local',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);a=p.parse_args()
 prep=read(a.local/'preparation.json');inputs=read(a.inputs);execution=read(a.local/'execution.json');assert execution['returncode']==0 and execution['preparation_sha256']==digest(a.local/'preparation.json')
 assert prep['inputs_sha256']==digest(a.inputs)
 assert inputs['mapping_sha256']==digest(ROOT/'results/media_mapping.jsonl')
 cfg=read(HERE/'config/whisper_alignment.json');raw={p.stem:p for p in (a.local/'raw').rglob('*.json')};jobs={};hashes={}
 for job in prep['jobs']:
  rows=[];status='no_tool_output'
  if job['file_id'] in raw:
   path=raw[job['file_id']];tiers=read(path)['tiers'];entries=[v['entries'] for k,v in tiers.items() if k=='words' or k.endswith(' - words')];assert len(entries)==1
   rows,status=map_partial_word_tier(job['source_words'],entries[0],job['source_offset_s'],job['duration_s']);hashes[str(path.relative_to(a.local))]=digest(path)
  jobs[job['file_id']]={**job,'mapped_words':rows,'status':status}
 media={m['id']:m for m in (json.loads(l) for l in (ROOT/'results/media_mapping.jsonl').read_text().splitlines())}
 explanations=[json.loads(l) for l in (ROOT/'results/uncertainty/special_explanations.jsonl').read_text().splitlines()]
 assert len(prep['samples'])==len(explanations)==20
 records=[];policies=[];counts=Counter();reasons=Counter();old_delta=[];totals=Counter();frame_cache={}
 for s,e in zip(prep['samples'],explanations):
  sid=s['sample_id'];assert e['id']==sid
  runs=[]
  for i,run in enumerate(s['support']['runs']):
   pair=sorted([jobs[j] for j in s['jobs'] if jobs[j]['run_index']==i],key=lambda j:j['context_s']);assert [j['context_s'] for j in pair]==[.5,1.]
   runs.append({**run,'run_index':i,'contexts':[j['mapped_words'] for j in pair]})
  baseline={'words':[{'text':w['text'],'char_span':w['char_span'],'automatic_eligible':False,'interval_s':None,'status':'unresolved'} for w in s['words']]}
  result=combine(s['words'],baseline,s['support'],runs,cfg,[s['audio']['start_s'],s['audio']['end_s']],s['zero_audio'])
  policies.append({'id':sid,**result,'support':s['support'],'runs':runs})
  totals['reference_words']+=len(s['words']);totals['word_candidates']+=sum(w['automatic_eligible'] for w in result['words']);totals['phrase_candidates']+=len(result['phrases']);totals['phrase_only_words']+=len(result['phrase_only_words'])
  entries=[]
  for old in media[sid]['entries']:
   nav=locate(old['char_spans'],s['words'],result);entry={**copy.deepcopy(old),'navigation':nav}
   # All old estimates are retained solely as a comparison, never accepted as a baseline.
   entry['previous_alignment_score']=entry.pop('alignment_score',None)
   entry['previous_interval_s']=[old['start_seconds'],old['end_seconds']] if old.get('start_seconds') is not None else None
   t=nav['interval_s'];entry.update(start_seconds=t[0] if t else None,end_seconds=t[1] if t else None,speech_time_status=nav['status'],feature_time_status='source_feature_timestamp_unavailable',time_granularity=nav['status'])
   entries.append(entry)
  lookup={j:z for z in entries for j in z['source_indices']};top={}
  for mod in 'TAV':
   top[mod]=[]
   for item in e['top_evidence'][mod]:
    selected=[lookup[j] for j in item['source_indices']];spans=[span for z in selected for span in z['char_spans']];nav=locate(spans,s['words'],result);counts[nav['status']]+=1
    row={**copy.deepcopy(item),'char_spans':spans,'text':' / '.join(dict.fromkeys(z['text'] for z in selected)),**nav,'feature_time_status':'source_feature_timestamp_unavailable'}
    if nav['status']=='insufficient_support':
     ix=nav['word_indices'];sw=[s['support']['words'][i] for i in ix]
     if not all(w['exact_unique'] for w in sw):reason='lexical_identity_not_unique_exact'
     elif not any(set(ix)<=set(r['word_indices']) for r in runs):reason='run_anchor_support_insufficient'
     elif not all(w['atomic_word_time'] for w in sw):reason='asr_internal_boundary_unavailable'
     elif any(d['status']=='new_word_time_conflict' and d.get('word_index') in ix for d in result['diagnostics']):reason='temporal_order_conflict'
     else:reason='acoustic_boundary_support_insufficient'
     row['unresolved_reason']=reason;reasons[reason]+=1
    t=nav['interval_s'];old=[z['previous_interval_s'] for z in selected]
    if t and all(old):
     delta=max(abs(t[0]-min(v[0] for v in old)),abs(t[1]-max(v[1] for v in old)));row['previous_boundary_delta_s']=delta
     if nav['status']=='word_candidate':old_delta.append(delta)
    if t:
     video=Path(inputs['source_root'])/s['path'];assert digest(video)==s['sha256']
     candidates=[(i,pts) for i,pts in enumerate(s['video_frame_pts_s']) if t[0]<=pts<t[1]]
     if candidates:
      index,pts=min(candidates,key=lambda v:abs(v[1]-sum(t)/2));name=f'{sid}_{index:05d}.png';dest=a.output/'frames'/name
      if name not in frame_cache:
       dest.parent.mkdir(parents=True,exist_ok=True)
       subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(video),'-vf',f'select=eq(n\\,{index})','-frames:v','1',str(dest)],check=True)
       frame_cache[name]=digest(dest)
      row['video_frame']={'path':'frames/'+name,'frame_index':index,'pts_s':pts,'sha256':frame_cache[name],'role':'synchronous_scene_navigation','speaker_identity_verified':False,'original_feature_frame_verified':False}
    top[mod].append(row)
  records.append({'id':sid,'raw_text':s['raw_text'],'entries':entries,'top_evidence':top,'source_sha256':s['sha256'],'source_feature_timestamps_available':False})
 a.output.mkdir(parents=True,exist_ok=True)
 (a.output/'navigation.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
 write(a.output/'alignment.json',{'samples':policies,'raw_sha256':hashes,'execution_sha256':digest(a.local/'execution.json')})
 summary={'samples':len(records),'samples_with_key_navigation':sum(any(i['interval_s'] for items in r['top_evidence'].values() for i in items) for r in records),'totals':dict(totals),'top_evidence':dict(counts),'unresolved_reasons':dict(reasons),'top_evidence_count':sum(counts.values()),'unique_frames':len({i['video_frame']['path'] for r in records for items in r['top_evidence'].values() for i in items if i.get('video_frame')}),'word_top_items_compared_with_previous':len(old_delta),'word_top_items_previous_boundary_difference_over_02s':sum(d>.2+1e-9 for d in old_delta),'maximum_previous_boundary_difference_s':max(old_delta,default=None),'human_data_used':False,'labels_used':False,'scope':'Model corroboration and navigation coverage; no ground-truth timing accuracy or prediction improvement measurement.','frozen_result_sha256':{str(path.relative_to(ROOT)):digest(path) for path in [ROOT/'results/uncertainty/special_explanations.jsonl',*[ROOT/'results'/m/'special_predictions.csv' for m in ['uncertainty','selfmm','tetfn']]]},'inputs_sha256':digest(a.inputs),'mapping_sha256':digest(ROOT/'results/media_mapping.jsonl')}
 write(a.output/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
