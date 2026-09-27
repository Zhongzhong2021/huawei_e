"""Build two fixed-context acoustic checks for independently recognized runs."""
import argparse, math, subprocess
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from common import read,write,digest
from math_features import words
from whisper_support import lexical_support
HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--asr',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args()
 cfg=read(HERE/'config/automatic_alignment.json');inputs=read(a.inputs);corpus=a.work/'corpus';corpus.mkdir(parents=True,exist_ok=False)
 jobs=[];samples=[]
 for item in inputs['samples']:
  sid=item['sample_id'];source=Path(inputs['source_root'])/item['path'];assert digest(source)==item['sha256']
  rec=read(a.asr/(sid+'.json'));assert rec['source_sha256']==item['sha256']
  refs=words(item['raw_text']);start,end=item['audio']['start_s'],item['audio']['end_s'];support=lexical_support(refs,rec['words'],cfg,start,end)
  raw=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(source),'-map','0:a:0','-ac','1','-ar','16000','-f','f32le','-']);wave=np.frombuffer(raw,dtype='<f4')
  sample={**item,'words':refs,'support':support,'jobs':[]}
  for ri,run in enumerate(support['runs']):
   ix=run['word_indices'];first=support['words'][ix[0]]['interval_s'][0];last=support['words'][ix[-1]]['interval_s'][1]
   for context in [.5,1.]:
    lo=max(0,math.floor((first-start-context)*16000));hi=min(len(wave),math.ceil((last-start+context)*16000));fid=f'w{len(jobs):05d}'
    wavfile.write(corpus/(fid+'.wav'),16000,wave[lo:hi]);selected=[refs[i] for i in ix];(corpus/(fid+'.lab')).write_text(' '.join(p.lower() for w in selected for p in w['parts']))
    jobs.append({'file_id':fid,'sample_id':sid,'run_index':ri,'context_s':context,'word_indices':ix,'source_words':selected,'sample_span_16k':[lo,hi],'source_offset_s':start+lo/16000,'duration_s':(hi-lo)/16000,'wav_sha256':digest(corpus/(fid+'.wav')),'text_sha256':digest(corpus/(fid+'.lab'))});sample['jobs'].append(fid)
  samples.append(sample)
 write(a.output/'preparation.json',{'samples':samples,'jobs':jobs,'inputs_sha256':digest(a.inputs),'recognition_manifest_sha256':digest(a.asr/'manifest.json'),'config_sha256':digest(HERE/'config/automatic_alignment.json')});print({'samples':len(samples),'jobs':len(jobs)})
if __name__=='__main__':main()
