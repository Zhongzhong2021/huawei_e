import sys,json,re,csv,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
W=Path(__file__).resolve().parent;ROOT=W.parents[1];sys.path.insert(0,str(W/'python_libs'));import h5py
F=ROOT/'outputs/问题一_建模解答_v7/features';O=W/'public_comparison';O.mkdir(exist_ok=True)
def tok(s):return re.findall(r"[a-z0-9]+(?:'[a-z]+)*",s.lower().replace('’',"'").replace('‘',"'"))
def match(ref,hyp):
 m,n=len(ref),len(hyp);d=np.zeros((m+1,n+1),dtype=np.int32);d[:,0]=np.arange(m+1)
 for i in range(1,m+1):
  for j in range(1,n+1):d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(ref[i-1]!=hyp[j-1]))
 j=int(np.argmin(d[-1]));cost=int(d[-1,j]);end=j;i=m;pairs=[]
 while i:
  if j and d[i,j]==d[i-1,j-1]+(ref[i-1]!=hyp[j-1]):
   if ref[i-1]==hyp[j-1]:pairs.append((i-1,j-1))
   i-=1;j-=1
  elif d[i,j]==d[i-1,j]+1:i-=1
  else:j-=1
 return cost,list(reversed(pairs)),j,end
records=[];details=[];metadata={}
with h5py.File(W/'public/CMU_MOSEI_TimestampedWords.csd','r') as h,h5py.File(W/'public/CMU_MOSEI_Labels.csd','r') as labels:
 for title,group in [('words',h['words']),('segment_intervals',labels['All Labels'])]:
  metadata[title]={k:str(group['metadata'][k][()]) for k in group['metadata']}
 for p in sorted((F/'samples').glob('*.json'),key=lambda p:json.loads(p.read_text(encoding='utf-8'))['table_row_0based']):
  m=json.loads(p.read_text(encoding='utf-8'));z=np.load(p.with_suffix('.npz'));vid,clip=m['sample_id'].split('$_$');ref=[w['word'] for w in m['words']]
  row=dict(sample_id=m['sample_id'],public_video_found=vid in h['words/data'],local_words=len(ref),current_supported_words=int(z['alignment_supported'].sum()))
  if not row['public_video_found']:records.append(row);continue
  data=h['words/data'][vid];intervals=data['intervals'][:];tokens=[];indices=[];raw=[x[0].decode('utf-8') for x in data['features'][:]]
  for j,s in enumerate(raw):
   if s.lower() in ['sp','sil','<sil>']:continue
   for t in tok(s):tokens.append(t);indices.append(j)
  cost,pairs,lo,hi=match(ref,tokens);pubwindow=intervals[np.asarray(indices[lo:hi])];matched=[]
  for i,j in pairs:
   ix=indices[j];pt=intervals[ix];local=[float(z['word_start_quantiles'][i,1]),float(z['word_end_quantiles'][i,1])]
   matched.append(dict(word_index=i,word=ref[i],public_word=raw[ix],public_index=ix,public_start=float(pt[0]),public_end=float(pt[1]),local_start=local[0],local_end=local[1],local_supported=bool(z['alignment_supported'][i])))
  supported=[x for x in matched if x['local_supported']]
  row.update(public_token_edit_distance=cost,public_exact_token_matches=len(pairs),public_text_span_start_s=float(pubwindow[0,0]),public_text_span_end_s=float(pubwindow[-1,1]),matched_supported_words=len(supported))
  if len(supported)>=3:
   offset=float(np.median([.5*(x['public_start']+x['public_end']-x['local_start']-x['local_end']) for x in supported]));res=[max(abs(x['public_start']-offset-x['local_start']),abs(x['public_end']-offset-x['local_end'])) for x in supported]
   row.update(fitted_source_offset_s=offset,shift_adjusted_boundary_difference_median_s=float(np.median(res)),shift_adjusted_boundary_difference_p95_s=float(np.quantile(res,.95)))
  segs=labels['All Labels/data'][vid]['intervals'][:] if vid in labels['All Labels/data'] else np.empty((0,2))
  span=[float(pubwindow[0,0]),float(pubwindow[-1,1])];overlap=np.maximum(0,np.minimum(segs[:,1],span[1])-np.maximum(segs[:,0],span[0])) if len(segs) else np.array([])
  candidates=[dict(row_index=int(j),start=float(segs[j,0]),end=float(segs[j,1])) for j in np.flatnonzero(overlap>0)]
  row['public_overlapping_segments']=len(candidates);row['local_duration_s']=m['duration_s'];records.append(row);details.append(dict(**row,official_text=m['text_original'],matched_words=matched,public_window_words=tokens[lo:hi],public_segment_candidates=candidates))
summary=dict(samples=len(records),public_video_matches=sum(r['public_video_found'] for r in records),unique_videos=len(set(r['sample_id'].split('$_$')[0] for r in records)),exact_text_sequences=sum(r.get('public_token_edit_distance')==0 for r in records),total_exact_token_matches=sum(r.get('public_exact_token_matches',0) for r in records),local_words=sum(r['local_words'] for r in records),unsupported_clips_found=sum(r['public_video_found'] and not r['current_supported_words'] for r in records),unsupported_clips_exact_text=sum(r.get('public_token_edit_distance')==0 and not r['current_supported_words'] for r in records),clips_with_shift_comparison=sum('fitted_source_offset_s' in r for r in records),notes=['Words file is a version-pinned community mirror listed as CMU-MOSEI; mirror hash verified, not verified against unavailable official host bytes.','No sentiment/emotion label feature values read; Labels.csd used only for interval records.','Clip ID is not assumed to equal Labels.csd row index; joins use video ID and transcript subsequences.','Fitted translation is estimated from local supported words; its residual is a descriptive cross-pipeline comparison, not independent accuracy or proof for unsupported clips.'])
(O/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');(O/'details.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf-8');(O/'public_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
keys=list(dict.fromkeys(k for r in records for k in r))
with (O/'全100条公开结果对照.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
print(json.dumps(summary,ensure_ascii=False));print(json.dumps(details[0],ensure_ascii=False))
