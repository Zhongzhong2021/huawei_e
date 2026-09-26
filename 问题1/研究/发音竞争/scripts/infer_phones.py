"""Direct audio-to-phone posterior extraction over all original samples."""
from pathlib import Path
import os
import sys,json,hashlib,subprocess,time
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC,Wav2Vec2FeatureExtractor
sys.path.insert(0,str(Path(os.environ.get('Q1_PROJECT',Path(__file__).resolve().parents[3]))/'scripts'))
from common import read,write,digest
from math_features import cells
R=Path(__file__).resolve().parents[1];Q=Path(os.environ.get('Q1_PROJECT',Path(__file__).resolve().parents[3]));M=Path(os.environ.get('Q1_PHONE_MODEL','/workspace/q1-phone-model'))
def main():
    torch.set_num_threads(2);torch.manual_seed(0)
    out=Path(os.environ.get('Q1_PHONE_POSTERIORS','/workspace/q1-phone-posteriors'));out.mkdir(exist_ok=True)
    manifest=read(M/'manifest.json')
    for item in manifest['files']:assert digest(M/item['path'])==item['sha256']
    extractor=Wav2Vec2FeatureExtractor.from_pretrained(M,local_files_only=True)
    model=Wav2Vec2ForCTC.from_pretrained(M,local_files_only=True).eval()
    stride=field=1
    for k,s in zip(model.config.conv_kernel,model.config.conv_stride):field+=(k-1)*stride;stride*=s
    inputs=read(Q/'data/inputs.json');records=[];vocab=read(M/'vocab.json');inverse={i:p for p,i in vocab.items()}
    with torch.inference_mode():
        for item in inputs['samples']:
            sid=item['sample_id'];path=Path(inputs['source_root'])/item['path'];assert digest(path)==item['sha256'];tic=time.monotonic()
            raw=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(path),'-map','0:a:0','-ac','1','-ar','16000','-f','f32le','-'])
            wave=np.frombuffer(raw,dtype='<f4').copy()
            if np.any(wave):
                batch=extractor(wave,sampling_rate=16000,return_tensors='pt')
                logp=model(**batch).logits[0].log_softmax(-1).numpy()
                clock=cells(item['audio']['start_s']+(np.arange(len(logp))*stride+(field-1)/2)/16000,item['audio']['start_s'],item['audio']['end_s'])
                ids=logp.argmax(-1);decoded=[int(i) for t,i in enumerate(ids) if i!=model.config.pad_token_id and (t==0 or i!=ids[t-1])]
            else:logp=np.empty((0,len(vocab)),np.float32);clock=np.empty((0,2));decoded=[]
            destination=out/(sid+'.npz');np.savez_compressed(destination,logp=logp,frame_cells=clock)
            records.append({'sample_id':sid,'source_sha256':item['sha256'],'posterior_sha256':digest(destination),'frames':len(logp),'decoded_phones':[inverse[i] for i in decoded],'elapsed_s':time.monotonic()-tic})
            print(sid,len(logp),round(time.monotonic()-tic,2),flush=True)
            write(out/'manifest.json',{'complete':len(records)==100,'model':manifest,'input_sha256':digest(Q/'data/inputs.json'),'script_sha256':digest(__file__),'conv_stride':stride,'conv_receptive_field':field,'blank_id':model.config.pad_token_id,'dtype':'float32','device':'cpu','records':records,'human_data_used':False})
if __name__=='__main__':main()
