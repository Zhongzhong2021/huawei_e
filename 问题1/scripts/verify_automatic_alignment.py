import json,hashlib
from pathlib import Path
import numpy as np
from feature_io import load
from selective_alignment import automatic_rows
ROOT=Path(__file__).resolve().parents[1]
def main():
 manifest=json.loads((ROOT/'data/automatic_alignment/manifest.json').read_text());inputs=json.loads((ROOT/'data/inputs.json').read_text());native={s['sample_id']:s for s in inputs['samples']};count=0;asr_equal=0
 for summary in manifest['samples']:
  sid=summary['sample_id'];r=json.loads((ROOT/'data/automatic_alignment'/(sid+'.json')).read_text());sample=load(ROOT/'data/features',sid);a,m=sample
  assert len(r['words'])==len(m['words']) and r['human_data_used'] is False
  assert ''.join(h['text'] for h in r['recognition'])==''.join(m['alignment']['ctc_text'].split());asr_equal+=1
  for i,w in enumerate(r['words']):
   rows=automatic_rows(sample,r,i,native[sid]['video']);assert rows['human_verified'] is False
   if w['automatic_eligible']:
    assert rows['audio'];assert all(0<=j<len(a['scene']) and weight>0 for j,weight in rows['scene']);count+=1
   else:assert w['interval_s'] is None and rows['audio']==rows['scene']==[]
  for c in r['context_envelopes']:assert not c['automatic_eligible'] and c['status']=='context_only_not_alignment' and c['interval_s'][1]>c['interval_s'][0]
 for name,expected in manifest['scripts'].items():assert hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()==expected
 result={'passed':True,'samples':len(manifest['samples']),'eligible_word_reads_checked':count,'decoded_transcripts_equal_previous_inference':asr_equal,'human_references_used_for_generation':False,'script_hashes_checked':True}
 (ROOT/'data/automatic_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
