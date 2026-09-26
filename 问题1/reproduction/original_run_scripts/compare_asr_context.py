import sys,json,csv,re
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W/'python_libs'));import h5py
from public_utils import tok,match
P=Path('C:/WorkArea/Huawei-Model');details=json.loads((W/'public_comparison/details.json').read_text(encoding='utf-8'));rows=[]
with h5py.File(W/'public/CMU_MOSEI_TimestampedWords.csd','r') as f:
 for m in details:
  sid=m['sample_id'];vid,clip=sid.split('$_$');a=json.loads((P/'results/q1_next/asr'/f'{vid}__{clip}.json').read_text(encoding='utf-8'))
  if m['current_supported_words'] or a['language']!='en' or not a['segments']:continue
  d=f['words/data'][vid];ints=d['intervals'][:];raw=[x[0].decode() for x in d['features'][:]];hyp=[];ix=[]
  for j,s in enumerate(raw):
   if s=='sp':continue
   for t in tok(s):hyp.append(t);ix.append(j)
  ref=tok(a['transcript']);cost,pairs,lo,hi=match(ref,hyp)
  if hi==lo:
   rows.append(dict(sample_id=sid,given_text=m['official_text'],asr_text=a['transcript'],asr_token_count=len(ref),asr_exact_matches=0,asr_edit_distance=cost,public_interval_gap_s=None));continue
  chosen=ints[np.array(ix[lo:hi],dtype=int)]
  start,end=float(chosen[0,0]),float(chosen[-1,1]);given=[m['public_text_span_start_s'],m['public_text_span_end_s']]
  gap=max(0,start-given[1],given[0]-end)
  rows.append(dict(sample_id=sid,given_text=m['official_text'],asr_text=a['transcript'],asr_token_count=len(ref),asr_exact_matches=len(pairs),asr_edit_distance=cost,asr_public_start_s=start,asr_public_end_s=end,given_public_start_s=given[0],given_public_end_s=given[1],public_interval_gap_s=gap,local_duration_s=m['local_duration_s']))
(W/'public_comparison/asr_context.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
with (W/'public_comparison/异常片段上下文对照.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
print(json.dumps(rows,ensure_ascii=False))
