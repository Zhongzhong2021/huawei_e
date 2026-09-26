"""AI-assisted Q1 study: actual waveform interventions, no emotion labels used."""
import json,time,hashlib,csv,itertools,subprocess,os
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch,torchaudio
from q1n_posterior import forward_backward
OUT=Path('results/q1_deep_v6/alignment')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def cells(n,K,origin,mapping):
    if mapping=='stretch':edges=np.linspace(0,n/16000,K+1)
    else:
        assert K==(n-400)//320+1,(K,n)
        centers=(np.arange(K)*320+199.5)/16000
        edges=np.r_[0,(centers[:-1]+centers[1:])/2,n/16000]
    return np.c_[edges[:-1],edges[1:]]+origin
def alignment(e,words,dictionary,separator):
    K,V=e.shape;ext=np.c_[e,np.zeros(K)];pad=np.full((1,V+1),-np.inf);pad[0,-1]=0
    target=[V];bounds=[]
    for i,w in enumerate(words):
        if i and separator is not None:target.append(dictionary[separator])
        start=len(target);target.extend(dictionary[c] for c in w);bounds.append((start,len(target)-1))
    target.append(V)
    en,ex,mem,logz,err=forward_backward(np.r_[pad,ext,pad],target,bounds)
    assert err<1e-6
    starts=np.array([[np.searchsorted(np.cumsum(a),q)-1 for q in [.05,.5,.95]] for a in en]);ends=np.array([[np.searchsorted(np.cumsum(a),q)-1 for q in [.05,.5,.95]] for a in ex])
    assert starts.min()>=0 and ends.max()<K
    return starts,ends,mem[:,1:-1],err
def decoded(e,labels):
    ids=[k for k,_ in itertools.groupby(e.argmax(1)) if k!=0]
    return ''.join(labels[k] for k in ids).replace('|',' ').lower()
def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(6);torch.manual_seed(20260924)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    files=sorted(Path('results/q1/samples').glob('*.json'),key=lambda p:json.loads(p.read_text())['table_row_0based']);assert len(files)==100
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),n=100,
        models=['MMS_FA','WAV2VEC2_ASR_BASE_960H'],conditions=['original','prepend_silence_0.6s','gain_0.5','white_noise_20dB'],seed=20260924,
        masks='Fixed previously established1464-word evidence mask, not selected on intervention results.',
        measures='Absolute start/end median displacement after removing known shift; no true alignment accuracy. Compare stretch versus convolution-anchor Voronoi time cells.',
        selection='No automatic replacement based on stability alone. Keep all samples, all words, baseline outputs. No emotion labels, training, or threshold optimization.',
        inputs={p.stem:dict(meta=sha(p),source=json.loads(p.read_text())['source_sha256']) for p in files},
        versions=dict(torch=torch.__version__,torchaudio=torchaudio.__version__))
    lock=OUT/'protocol.json'
    if lock.exists():assert json.loads(lock.read_text())['source_sha256']==sha(__file__),'Use a new version for changed code'
    else:dump(lock,protocol)
    start=time.perf_counter();allrows=[];modelmeta={};history=[]
    for name in protocol['models']:
        bundle=getattr(torchaudio.pipelines,name); model=bundle.get_model(with_star=False) if name=='MMS_FA' else bundle.get_model()
        model=model.eval().cuda();labels=bundle.get_labels(star=None) if name=='MMS_FA' else bundle.get_labels()
        dictionary={c.lower():i for i,c in enumerate(labels)};sep=None if name=='MMS_FA' else '|'
        modelmeta[name]=dict(parameters=sum(p.numel() for p in model.parameters()),labels=labels,bundle_path=bundle._path)
        for p in files:
            out=OUT/name/p.stem;out.mkdir(exist_ok=True,parents=True);finish=out/'complete.json'
            if finish.exists():history.append(json.loads(finish.read_text()));continue
            t=time.perf_counter();meta=json.loads(p.read_text());source=Path('/data')/meta['source_path'];assert sha(source)==meta['source_sha256']
            wave=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',str(source),'-map','0:a:0','-ac','1','-ar','16000','-f','f32le','pipe:1']),dtype='<f4').copy()
            words=[w['word'] for w in meta['words']];supported=np.load(Path('results/q1_next/enhanced')/p.with_suffix('.npz').name)['alignment_supported']
            seed=int(hashlib.sha256(p.stem.encode()).hexdigest()[:8],16)^20260924;rng=np.random.default_rng(seed)
            noise=rng.normal(size=len(wave));noise*=np.sqrt(np.mean(wave.astype(float)**2)/100)/max(np.sqrt(np.mean(noise**2)),1e-12)
            waves=[wave,np.r_[np.zeros(9600,np.float32),wave],wave*.5,(wave+noise).astype(np.float32)];records=[];ref={}
            for condition,wav,shift in zip(protocol['conditions'],waves,[0,.6,0,0]):
                with torch.inference_mode():emission=model(torch.from_numpy(wav)[None].cuda())[0][0].log_softmax(-1).cpu().numpy()
                si,ei,mem,err=alignment(emission,words,dictionary,sep);data={}
                for mapping in ['stretch','conv_anchor']:
                    cc=cells(len(wav),len(emission),meta['audio_start_s'],mapping);st=cc[si,0];en=cc[ei,1];data[mapping+'_start']=st;data[mapping+'_end']=en
                    if condition=='original':ref[mapping]=(st,en)
                    else:
                        ds=st[:,1]-shift-ref[mapping][0][:,1];de=en[:,1]-shift-ref[mapping][1][:,1]
                        for j,w in enumerate(words):records.append(dict(sample=p.stem,video_id=p.stem.split('__')[0],model=name,condition=condition,mapping=mapping,word_index=j,word=w,supported=bool(supported[j]),start_error_s=float(abs(ds[j])),end_error_s=float(abs(de[j])),max_boundary_error_s=float(max(abs(ds[j]),abs(de[j])))))
                if condition=='original':data.update(membership=mem.astype(np.float32),ctc_cells_conv=cells(len(wav),len(emission),meta['audio_start_s'],'conv_anchor'))
                np.savez_compressed(out/(condition+'.npz'),**data)
                dump(out/(condition+'.json'),dict(normalization_error=err,frames=len(emission),n_audio_samples=len(wav),transcript=decoded(emission,labels)))
            with (out/'word_interventions.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
            row=dict(model=name,sample=p.stem,words=len(words),supported=int(supported.sum()),seconds=time.perf_counter()-t);dump(finish,row);history.append(row)
            if len(history)%10==0:print('FINISHED',len(history),'/',200,'last_seconds',row['seconds'],flush=True)
        del model;torch.cuda.empty_cache()
    modelmeta['weight_files']={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in Path('/cache/torch/hub/checkpoints').glob('*.pth') if 'wav2vec' in p.name}
    modelmeta['mms_sha256']=sha('/cache/torch/hub/checkpoints/model.pt');dump(OUT/'models.json',modelmeta)
    for name in protocol['models']:
        for p in files:
            with (OUT/name/p.stem/'word_interventions.csv').open(encoding='utf-8-sig') as f:
                for r in csv.DictReader(f):
                    for k in ['start_error_s','end_error_s','max_boundary_error_s']:r[k]=float(r[k])
                    r['supported']=r['supported']=='True';allrows.append(r)
    summaries=[]
    for name in protocol['models']:
        for cond in protocol['conditions'][1:]:
            for mapping in ['stretch','conv_anchor']:
                for group in ['all','supported']:
                    rr=[r for r in allrows if r['model']==name and r['condition']==cond and r['mapping']==mapping and (group=='all' or r['supported'])]
                    v=np.array([r['max_boundary_error_s'] for r in rr]);summaries.append(dict(model=name,condition=cond,mapping=mapping,group=group,n=len(rr),median=float(np.median(v)),p90=float(np.quantile(v,.9)),p95=float(np.quantile(v,.95)),over100ms=int((v>.1).sum()),over200ms=int((v>.2).sum())))
    result=dict(samples=100,models=2,waveform_forward_passes=800,summaries=summaries,elapsed_seconds=time.perf_counter()-start,sample_compute_seconds=sum(r['seconds'] for r in history),limits='Paired perturbation consistency, not true boundary error or emotional prediction quality. Shared audio and related CTC model families do not provide independent truth.')
    dump(OUT/'results.json',result);print('COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
