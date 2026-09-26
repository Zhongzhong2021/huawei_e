"""Automatic experiment. Reads only fixed config, original media, native metadata and models."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import torch,torchaudio
from transformers import Wav2Vec2Processor,Wav2Vec2ForCTC
from common import run
from math_features import cells
from selective_alignment import align_sample
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def decode_words(ids,logp,vocab,blank,frame_cells):
    inverse={v:k for k,v in vocab.items()};chunks=[];current=[];start=0
    def flush():
        if current:
            chunks.append({'text':''.join(c[0] for c in current),'interval_s':[float(frame_cells[current[0][1],0]),float(frame_cells[current[-1][2]-1,1])],'mean_emission_probability':float(np.mean([c[3] for c in current]))});current.clear()
    while start<len(ids):
        end=start+1
        while end<len(ids) and ids[end]==ids[start]:end+=1
        token=inverse[int(ids[start])]
        if int(ids[start])!=blank:
            if token=='|':flush()
            elif len(token)==1:current.append((token,start,end,float(logp[start:end,int(ids[start])].exp().mean())))
        start=end
    flush();return chunks

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',type=Path,default=Path('/workspace/q1-models'));args=ap.parse_args()
    out=ROOT/'data/automatic_alignment';out.mkdir(exist_ok=False)
    cfg=read(ROOT/'config/automatic_alignment.json');inputs=read(ROOT/'data/inputs.json');torch.set_num_threads(2);torch.manual_seed(0)
    for item in read(args.models/'manifest.json'):
        if item['path'].startswith('ctc/'):assert sha(args.models/item['path'])==item['sha256']
    processor=Wav2Vec2Processor.from_pretrained(args.models/'ctc',local_files_only=True);model=Wav2Vec2ForCTC.from_pretrained(args.models/'ctc',local_files_only=True).eval();stride=field=1
    for k,s in zip(model.config.conv_kernel,model.config.conv_stride):field+=(k-1)*stride;stride*=s
    summary=[]
    with torch.inference_mode():
        for item in inputs['samples']:
            sid=item['sample_id'];tic=time.monotonic();meta=read(ROOT/'data/features'/(sid+'.json'));path=Path(inputs['source_root'])/item['path'];assert sha(path)==item['sha256']
            pcm=np.frombuffer(run(['ffmpeg','-nostdin','-v','error','-i',str(path),'-map','0:a:0','-f','f32le','-']),'<f4').reshape(-1,item['audio']['channels']);wave=torchaudio.functional.resample(torch.from_numpy(pcm.mean(1).copy()),int(item['audio']['sample_rate']),16000).numpy();hyp=[]
            if np.any(wave):
                batch=processor(wave,sampling_rate=16000,return_tensors='pt');lp=model(batch.input_values).logits[0].log_softmax(-1);ids=lp.argmax(-1).numpy();clock=cells(item['audio']['start_s']+(np.arange(len(ids))*stride+(field-1)/2)/16000,item['audio']['start_s'],item['audio']['start_s']+len(wave)/16000);hyp=decode_words(ids,lp,processor.tokenizer.get_vocab(),model.config.pad_token_id,clock)
            with np.load(ROOT/'data/features'/(sid+'.npz')) as a:baseline=a['word_intervals'].copy();mask=a['word_candidate_mask'].copy()
            result=align_sample(meta['words'],hyp,baseline,cfg,no_signal=not bool(np.any(wave)))
            result.update(sample_id=sid,source_sha256=item['sha256'],config_sha256=sha(ROOT/'config/automatic_alignment.json'),recognition=hyp,baseline_candidate_words=int(mask.sum()),native_features_modified=False,human_data_used=False)
            (out/(sid+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');summary.append({'sample_id':sid,'sample_status':result['sample_status'],'automatic_eligible_words':result['automatic_eligible_words'],'words':len(meta['words']),'baseline_candidate_words':int(mask.sum())});print(sid,result['sample_status'],result['automatic_eligible_words'],round(time.monotonic()-tic,2),flush=True)
    manifest={'config_sha256':sha(ROOT/'config/automatic_alignment.json'),'scripts':{p:sha(ROOT/'scripts'/p) for p in ['selective_alignment.py','run_automatic_alignment.py']},'processed_samples':len(summary),'model_forward_samples':sum(r['sample_status']!='no_signal' for r in summary),'models_manifest_sha256':sha(args.models/'manifest.json'),'human_data_used':False,'samples':summary};(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
