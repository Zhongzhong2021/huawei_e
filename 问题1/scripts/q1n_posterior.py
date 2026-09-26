# 本程序及代码是在人工智能工具辅助下完成的。
# 工具：OpenAI Codex；模型：GPT-6 Astra (gpt-6-astra)；开发机构：OpenAI；模型发布日期：2026-09-03。
# AI 来源说明于 2026-09-24 补录，未改变算法；参见 AI辅助使用记录与队内审阅要求.md。
"""CTC conditional boundary distributions and posterior expected temporal pooling.

These distributions condition on supplied text, so they cannot validate that text.
Independent acoustic/lexical evidence must gate downstream use.
"""
import argparse,itertools,json,time
from pathlib import Path
import numpy as np

def forward_backward(e,target,bounds):
    """Log emissions T,V; labels nonblank; bounds inclusive target indices per word."""
    labels=np.zeros(2*len(target)+1,dtype=int);labels[1::2]=target
    T,S=len(e),len(labels); E=e[:,labels].astype(np.float64)
    skip=np.zeros(S,bool);skip[2:]=(labels[2:]!=0)&(labels[2:]!=labels[:-2])
    a=np.full((T,S),-np.inf);a[0,:2]=E[0,:2]
    for t in range(1,T):
        prev=a[t-1]; one=np.r_[-np.inf,prev[:-1]];two=np.r_[[-np.inf]*2,prev[:-2]]
        a[t]=E[t]+np.logaddexp(np.logaddexp(prev,one),np.where(skip,two,-np.inf))
    z=np.logaddexp(a[-1,-1],a[-1,-2]);assert np.isfinite(z),'no valid CTC path'
    b=np.full((T,S),-np.inf);b[-1,-2:]=0
    for t in range(T-2,-1,-1):
        nxt=E[t+1]+b[t+1];one=np.r_[nxt[1:],-np.inf];two=np.r_[nxt[2:],[-np.inf]*2]
        b[t]=np.logaddexp(np.logaddexp(nxt,one),np.where(np.r_[skip[2:],False,False],two,-np.inf))
    entries=[];exits=[];errors=[]
    for lo,hi in bounds:
        s=2*lo+1;q=2*hi+1
        inp=a[:-1,s-1].copy()
        if skip[s]: inp=np.logaddexp(inp,a[:-1,s-2])
        en=np.r_[np.exp(E[0,s]+b[0,s]-z) if s==1 else 0,np.exp(inp+E[1:,s]+b[1:,s]-z)]
        out=E[1:,q+1]+b[1:,q+1]
        if q+2<S and skip[q+2]:out=np.logaddexp(out,E[1:,q+2]+b[1:,q+2])
        ex=np.r_[np.exp(a[:-1,q]+out-z),np.exp(a[-1,q]-z) if q==S-2 else 0]
        errors.extend([abs(en.sum()-1),abs(ex.sum()-1)])
        en/=en.sum();ex/=ex.sum();entries.append(en);exits.append(ex)
    entries=np.array(entries);exits=np.array(exits)
    # A word covers frame k iff entered by k and not yet exited before k.
    membership=np.cumsum(entries,axis=1)-np.c_[np.zeros(len(bounds)),np.cumsum(exits,axis=1)[:,:-1]]
    assert membership.min()>-1e-7 and membership.max()<1+1e-7
    return entries,exits,np.clip(membership,0,1),float(z),max(errors)

def selftest():
    rng=np.random.default_rng(20260924); errors=[]
    # Independent exhaustive state-path enumeration; covers repeated labels and optional blanks.
    for target in ([1,2],[1,1],[1,2,1]):
        raw=rng.uniform(.1,1,(7,3)); e=np.log(raw/raw.sum(1,keepdims=True)); bounds=[(i,i) for i in range(len(target))]
        en,ex,mem,z,err=forward_backward(e,target,bounds)
        labels=np.zeros(2*len(target)+1,int);labels[1::2]=target;S=len(labels)
        paths=[]
        def visit(path):
            if len(path)==len(e):
                if path[-1] in (S-2,S-1):paths.append(path)
                return
            s=path[-1]
            for q in (s,s+1,s+2):
                if q<S and (q!=s+2 or (labels[q]!=0 and labels[q]!=labels[s])):visit(path+[q])
        visit([0]);visit([1]); weights=np.array([np.exp(sum(e[t,labels[s]] for t,s in enumerate(p))) for p in paths]);total=weights.sum();weights/=total
        ee=np.zeros_like(en);xx=np.zeros_like(ex);mm=np.zeros_like(mem)
        for p,w in zip(paths,weights):
            for j in range(len(target)):
                indices=np.flatnonzero(np.array(p)==2*j+1);lo,hi=indices[0],indices[-1]
                ee[j,lo]+=w;xx[j,hi]+=w;mm[j,lo:hi+1]+=w
        delta=max(np.max(abs(en-ee)),np.max(abs(ex-xx)),np.max(abs(mem-mm)),abs(np.exp(z)-total),err)
        assert delta<1e-10;errors.append(dict(target=target,paths=len(paths),max_error=float(delta)))
    return errors

def soft_pool(membership,ctc_cells,cells,x,valid,std=False):
    overlap=np.maximum(0,np.minimum(ctc_cells[:,None,1],cells[None,:,1])-np.maximum(ctc_cells[:,None,0],cells[None,:,0]))
    weights=(membership@overlap)*valid[None,:];den=weights.sum(1)
    norm=weights/np.maximum(den[:,None],1e-12);mean=norm@x
    y=np.c_[mean,np.sqrt(np.maximum(norm@(x*x)-mean*mean,0))] if std else mean
    expected_duration=membership@(ctc_cells[:,1]-ctc_cells[:,0])
    present=den>=np.maximum(.005,.05*expected_duration)
    y[~present]=0
    return y.astype(np.float32),present,den

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--limit',type=int);ap.add_argument('--force',action='store_true');args=ap.parse_args()
    out=Path('results/q1_next');out.mkdir(exist_ok=True)
    tests=selftest();(out/'posterior_algorithm_test.json').write_text(json.dumps(tests,indent=2))
    if args.selftest:print(tests);return
    import torch,torchaudio
    from q1_extract import command,sha,dump
    torch.set_num_threads(8);torch.manual_seed(0)
    bundle=torchaudio.pipelines.MMS_FA; model=bundle.get_model(with_star=False).eval().cuda();dictionary=bundle.get_dict()
    dump(out/'posterior_config.json',dict(model='torchaudio.pipelines.MMS_FA',model_sha256=sha('/cache/torch/hub/checkpoints/model.pt'),
          sample_rate=16000,star_log_weight=0,virtual_boundary_frames=2,quantiles=[.05,.5,.95],
          dynamic_program_dtype='float64',stored_membership_dtype='float32',stored_candidate_dtype='float32',
          occupancy_tail_cutoff=1e-4,min_observed_effective_seconds=.005,min_observed_fraction=.05,
          torch_version=torch.__version__,torchaudio_version=torchaudio.__version__,seed=0,script_sha256=sha(__file__)))
    dest=out/'posterior';dest.mkdir(exist_ok=True)
    files=sorted(Path('results/q1/samples').glob('*.json'),key=lambda p:json.loads(p.read_text())['table_row_0based'])
    for p in files[:args.limit]:
        if (dest/p.name).exists() and not args.force:continue
        started=time.time();meta=json.loads(p.read_text());source=Path('/data')/meta['source_path'];assert sha(source)==meta['source_sha256']
        wave=np.frombuffer(command(['ffmpeg','-v','error','-i',source,'-map','0:a:0','-ac','1','-ar','16000','-f','f32le','pipe:1']),dtype='<f4').copy()
        with torch.inference_mode():e=model(torch.from_numpy(wave)[None].cuda())[0][0].cpu().numpy()
        K,V=e.shape; delta=len(wave)/16000/K; origin=meta['audio_start_s']
        ext=np.c_[e,np.zeros(K)];pad=np.full((1,V+1),-np.inf);pad[0,-1]=0;e=np.r_[pad,ext,pad]
        target=[V];bounds=[]
        for w in meta['words']:
            start=len(target);target.extend(dictionary[c] for c in w['word']);bounds.append((start,len(target)-1))
        target.append(V)
        en,ex,membership,z,error=forward_backward(e,target,bounds);membership=membership[:,1:-1]
        # Prevent tiny posterior tails from borrowing a distant face observation.
        membership[membership<1e-4]=0
        membership=membership.astype(np.float32).astype(np.float64)
        assert error<1e-6
        def quantiles(distribution,shift):
            return np.array([[(np.searchsorted(np.cumsum(d),q)+shift)*delta+origin for q in (.05,.5,.95)] for d in distribution])
        starts=quantiles(en,-1);ends=quantiles(ex,0)
        ctc_cells=np.c_[np.arange(K)*delta+origin,(np.arange(K)+1)*delta+origin]
        native=np.load(p.with_suffix('.npz'),allow_pickle=False)
        result=dict(start_quantiles=starts,end_quantiles=ends,ctc_cells=ctc_cells,word_time_membership=membership.astype(np.float32))
        for name,std in [('audio',True),('vision',False),('scene',False)]:
            cells=native['audio_cells' if name=='audio' else 'vision_cells']
            x=native[name+'_native'].astype(np.float64)
            valid=np.ones(len(x),bool) if name!='vision' else native['vision_valid']
            y,mask,den=soft_pool(membership,ctc_cells,cells,x,valid,std)
            result[name+'_soft_candidate']=y;result[name+'_soft_presence']=mask;result[name+'_soft_duration']=den.astype(np.float32)
        for x in result.values():assert np.isfinite(x).all()
        np.savez_compressed(dest/p.with_suffix('.npz').name,**result)
        widths=np.maximum(starts[:,2]-starts[:,0],ends[:,2]-ends[:,0])
        d=dict(sample_id=meta['sample_id'],source_sha256=meta['source_sha256'],baseline_feature_sha256=sha(p.with_suffix('.npz')),
               word_count=len(bounds),conditional_boundary_width_median_s=float(np.median(widths)),
               wide_boundary_words=int((widths>.2).sum()),normalization_max_error=error,
               start_quantiles=starts.tolist(),end_quantiles=ends.tolist(),
               conditioning='official text and optional boundary context; not calibrated timing confidence',elapsed_seconds=time.time()-started)
        dump(dest/p.name,d);print(json.dumps({k:v for k,v in d.items() if k not in ('start_quantiles','end_quantiles')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
