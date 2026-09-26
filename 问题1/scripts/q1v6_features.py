"""AI-assisted Q1 v6: feature-wise validity, voiced pitch, word/native/clock views."""
import argparse,json,time,hashlib,csv
from pathlib import Path
import numpy as np
F0=45
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def overlap(a,b):return np.maximum(0,np.minimum(a[:,None,1],b[None,:,1])-np.maximum(a[:,None,0],b[None,:,0]))
def read_npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def audio_pool(w,x,voiced):
    x=x.astype(np.float64);w=w.astype(np.float64);mass=w.sum(1);vmass=w@voiced.astype(float)
    norm=w/np.maximum(mass[:,None],1e-12);mean=norm@x;var=np.maximum(norm@(x*x)-mean*mean,0)
    vw=w*voiced[None,:];vnorm=vw/np.maximum(vmass[:,None],1e-12)
    mean[:,F0]=vnorm@x[:,F0];var[:,F0]=np.maximum(vnorm@(x[:,F0]**2)-mean[:,F0]**2,0)
    logpitch=np.zeros(len(x));logpitch[voiced]=12*np.log2(x[voiced,F0]/27.5)
    lm=vnorm@logpitch;ls=np.sqrt(np.maximum(vnorm@(logpitch**2)-lm**2,0));fraction=vmass/np.maximum(mass,1e-12)
    out=np.c_[mean,np.sqrt(var),lm,ls,fraction];valid=np.repeat((mass>=.005)[:,None],97,axis=1)
    valid[:,[45,92,94,95]]&=(vmass>=.005)[:,None];out[~valid]=0
    return out.astype(np.float32),valid,mass,vmass
def visual_pool(w,x,valid):
    weights=w*valid[None,:];mass=weights.sum(1);out=weights@x.astype(float)/np.maximum(mass[:,None],1e-12);mask=mass>=.005;out[~mask]=0
    return out.astype(np.float32),mask,mass
def clock_view(z,duration,step=.1):
    assert step>0
    edge=np.r_[np.arange(0,duration,step),duration];grid=np.c_[edge[:-1],edge[1:]]
    audio,ad,am,vm=audio_pool(overlap(grid,z['audio_cells']),z['audio_native'],z['audio_voiced'])
    visual,vv,vs=visual_pool(overlap(grid,z['vision_cells']),z['expression_native'],z['expression_valid'])
    scene,sv,ss=visual_pool(overlap(grid,z['vision_cells']),z['scene_native'],np.ones(len(z['vision_cells']),bool))
    w=overlap(grid,z['ctc_cells'])@z['word_time_membership'].astype(float).T;w*=z['alignment_supported'][None,:];mass=w.sum(1)
    text=w@z['text'].astype(float)/np.maximum(mass[:,None],1e-12);tm=mass>=.005;text[~tm]=0
    return dict(cells=grid,text=text.astype(np.float32),text_mask=tm,text_effective_seconds=mass,
        audio=audio,audio_dim_mask=ad,audio_mask=ad[:,0],audio_effective_seconds=am,voiced_effective_seconds=vm,
        expression=visual,expression_mask=vv,expression_effective_seconds=vs,scene=scene,scene_mask=sv,scene_effective_seconds=ss)
def load_sample(root,stem,view='word',step=.1):
    root=Path(root);p=root/'samples'/f'{stem}.npz';z=read_npz(p);meta=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'));assert sha(p)==meta['feature_sha256']
    if view=='native':return z,meta
    if view=='clock':return clock_view(z,meta['duration_s'],step),meta
    assert view=='word'
    return {k:z[k] for k in ['text','text_mask','word_audio','word_audio_dim_mask','word_expression','word_expression_mask','word_scene','word_scene_mask','alignment_supported','word_start_quantiles','word_end_quantiles']},meta
def collate(samples):
    n=np.array([len(s['text']) for s in samples],dtype=np.int32);L=int(n.max());out={'lengths':n,'sequence_mask':np.arange(L)[None,:]<n[:,None]}
    for key in samples[0]:
        a=samples[0][key]
        if isinstance(a,np.ndarray) and a.ndim and a.shape[0]==n[0]:
            out[key]=np.zeros((len(samples),L,*a.shape[1:]),dtype=a.dtype)
            for i,s in enumerate(samples):out[key][i,:n[i]]=s[key]
    return out
def selftest():
    # Equal voiced/unvoiced duration, fixed200Hz; old zero-filled mean is100Hz.
    x=np.zeros((4,47));x[:2,F0]=200;voiced=np.array([1,1,0,0],bool)
    out,mask,_,_=audio_pool(np.ones((1,4))*.1,x,voiced)
    assert abs(out[0,F0]-200)<1e-6 and out[0,92]==0 and abs(out[0,96]-.5)<1e-6
    empty,em,_,_=audio_pool(np.ones((1,4))*.1,np.zeros_like(x),np.zeros(4,bool));assert not em[0,45] and em[0,0] and empty[0,45]==0
    # Row weights represent time support; irregular intersections conserve observed duration.
    a=np.array([[0,.12],[.12,.31]]);b=np.array([[0,.1],[.1,.2],[.2,.31]])
    assert np.allclose(overlap(a,b).sum(0),np.diff(b).ravel())
    return dict(constant200Hz_half_voiced=dict(old_mean_hz=100,new_mean_hz=float(out[0,45]),new_std_hz=float(out[0,92]),voiced_fraction=float(out[0,96])),all_unvoiced_pitch_mask=False,other_acoustic_features_retained=True,irregular_overlap_conservation=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',default='.');ap.add_argument('--output',default='results/q1_deep_v6/features_v2');args=ap.parse_args()
    P=Path(args.project);O=Path(args.output);assert not O.exists(),'Use a new version; do not overwrite';(O/'samples').mkdir(parents=True);start=time.perf_counter()
    config=json.loads((P/'results/q1/run_config.json').read_text(encoding='utf-8'));names=config['acoustic_native_columns']
    clockfile=P/'results/q1_deep_v6/media_clock/results.json';clocks={r['stem']:r for r in json.loads(clockfile.read_text(encoding='utf-8'))['rows']}
    columns=['mean_'+s for s in names]+['std_'+s for s in names]+['mean_voiced_F0_semitone_ref27.5Hz','std_voiced_F0_semitone_ref27.5Hz','voiced_fraction']
    columns[45]='mean_voiced_F0_hz';columns[92]='std_voiced_F0_hz'
    protocol=dict(source_sha256=sha(__file__),schema='q1-v6-observed-clock',media_clock_sha256=sha(clockfile),n=100,emotion_labels_used=False,native_audio_dim=47,word_audio_dim=97,text_dim=768,expression_dim=52,scene_dim=512,
        audio_columns=columns,word_rule='CTC posterior occupancy with convolution-anchor support cells; original1464-word support mask unchanged; F0 functionals on voiced frames only.',
        grid_rule='100ms physical clock, independently retain available AV; map supported text with posterior overlap. Native time arrays preserved.',
        thresholds=dict(min_effective_seconds=.005,min_word_observed_fraction=.05,occupancy_tail_cutoff=.0001),
        meanings='Time support cells are pooling weights, not exposure duration or calibrated true word boundaries. Contextual BERT and transformer states use broader context.',
        non_changes='No new emotion data, no corrected source transcripts/labels, no active speaker claims, no empirical alignment accuracy claim.',
        previous_extraction_config=config,selftests=selftest())
    dump(O/'protocol.json',protocol);rows=[];wordrows=[];membdiff=[];identity=[];totalgrid=0
    files=sorted((P/'results/q1/samples').glob('*.json'),key=lambda p:json.loads(p.read_text())['table_row_0based']);assert len(files)==100
    for p in files:
        meta=json.loads(p.read_text());b=read_npz(p.with_suffix('.npz'));po=read_npz(P/'results/q1_next/posterior'/p.with_suffix('.npz').name);en=read_npz(P/'results/q1_next/enhanced'/p.with_suffix('.npz').name);face=read_npz(P/'results/q1_refinement_v1/faces'/p.with_suffix('.npz').name)
        original=read_npz(P/'results/q1_deep_v6/alignment/MMS_FA'/p.stem/'original.npz');r=original['membership'].copy();r[r<.0001]=0;ct=original['ctc_cells_conv'];membdiff.append(float(np.max(abs(r-po['word_time_membership']))));supported=en['alignment_supported']
        wa=r.astype(float)@overlap(ct,b['audio_cells']);av,adm,am,vm=audio_pool(wa,b['audio_native'],b['audio_voiced']);expected=r.astype(float)@np.diff(ct).ravel();minmass=np.maximum(.005,.05*expected)
        adm&=(supported&(am>=minmass))[:,None];av[~adm]=0
        wv=r.astype(float)@overlap(ct,face['vision_expression_cells']);ev,em,ed=visual_pool(wv,face['vision_expression_native'],face['vision_expression_valid']);sv,sm,sd=visual_pool(wv,b['scene_native'],np.ones(len(b['scene_native']),bool))
        em&=supported&(ed>=minmass);sm&=supported&(sd>=minmass);ev[~em]=0;sv[~sm]=0
        z={k:b[k] for k in ['text','text_mask','audio_native','audio_time','audio_cells','audio_voiced','bert_token_ids','bert_word_membership','bert_char_offsets','scene_native']}
        z.update(expression_native=face['vision_expression_native'],expression_valid=face['vision_expression_valid'],vision_time=face['vision_expression_time'],vision_cells=face['vision_expression_cells'],vision_frame_index=face['vision_expression_frame_index'],vision_bbox=face['vision_expression_bbox'],recovery_route=face['recovery_route'],alignment_supported=supported,ctc_cells=ct,word_time_membership=r,
            word_start_quantiles=original['conv_anchor_start'],word_end_quantiles=original['conv_anchor_end'],word_audio=av,word_audio_dim_mask=adm,word_audio_effective_seconds=am.astype(np.float32),word_voiced_effective_seconds=vm.astype(np.float32),word_expression=ev,word_expression_mask=em,word_scene=sv,word_scene_mask=sm,
            ctc_word_score=b['alignment_score'],asr_exact_match=en['asr_exact_match'],asr_time_gap_s=en['asr_time_gap_s'],asr_word_probability=en['asr_word_probability'])
        assert np.all(b['audio_native'][~b['audio_voiced'],F0]==0)
        clock=clocks[p.stem];duration=clock['observed_end_s'];grid=clock_view(z,duration);totalgrid+=len(grid['cells']);grid_dt=np.diff(grid['cells']).ravel()
        for v in z.values():assert np.isfinite(v).all()
        assert len(z['text'])==len(meta['words']) and np.all(z['word_audio'][~adm]==0)
        # Exact decomposition of the old-style zero-filled pitch mean under same time weights.
        old=wa@b['audio_native'][:,F0]/np.maximum(am,1e-12);fraction=vm/np.maximum(am,1e-12)
        new=np.divide(wa@(b['audio_native'][:,F0]*b['audio_voiced']),vm,out=np.zeros(len(vm)),where=vm>0)
        identity.append(float(np.max(abs(old-new*fraction))))
        for j,w in enumerate(meta['words']):wordrows.append(dict(sample_id=meta['sample_id'],word_index=j,word=w['word'],supported=bool(supported[j]),old_zero_filled_F0_hz=float(old[j]),new_conditional_F0_hz=float(new[j]),voiced_fraction=float(fraction[j]),pitch_valid=bool(adm[j,45]),start_median_s=float(z['word_start_quantiles'][j,1]),end_median_s=float(z['word_end_quantiles'][j,1])))
        dst=O/'samples'/p.with_suffix('.npz').name;np.savez_compressed(dst,**z)
        words=[{k:v for k,v in w.items() if k in ['word','source_text','char_start','char_end','normalization','bert_positions','bert_tokens']} for w in meta['words']]
        metadata=dict(sample_id=meta['sample_id'],table_row_0based=meta['table_row_0based'],source_path=meta['source_path'],source_sha256=meta['source_sha256'],text_original=meta['text_original'],words=words,duration_s=duration,container_declared_duration_s=meta['duration_s'],media_clock=clock,decoded_audio_duration_s=meta['decoded_audio_duration_s'],audio_start_s=meta['audio_start_s'],feature_sha256=sha(dst),
            schema='q1-v6-observed-clock',dependencies={str(q.relative_to(P)).replace('\\','/'):sha(q) for q in [p,p.with_suffix('.npz'),P/'results/q1_next/enhanced'/p.with_suffix('.npz').name,P/'results/q1_refinement_v1/faces'/p.with_suffix('.npz').name,P/'results/q1_deep_v6/alignment/MMS_FA'/p.stem/'original.npz',clockfile]})
        dump(dst.with_suffix('.json'),metadata)
        rows.append(dict(sample_id=meta['sample_id'],table_row=meta['table_row_0based'],duration_s=duration,container_declared_duration_s=meta['duration_s'],audio_duration_s=meta['decoded_audio_duration_s'],words=len(words),supported_words=int(supported.sum()),text_dim=768,audio_native_dim=47,audio_word_dim=97,expression_dim=52,scene_dim=512,audio_native_rows=len(b['audio_native']),video_native_rows=len(z['vision_time']),face_valid_rows=int(z['expression_valid'].sum()),word_audio_valid=int(adm[:,0].sum()),word_pitch_valid=int(adm[:,45].sum()),word_expression_valid=int(em.sum()),clock_step_s=.1,clock_rows=len(grid['cells']),clock_audio_valid=int(grid['audio_mask'].sum()),clock_expression_valid=int(grid['expression_mask'].sum()),clock_text_valid=int(grid['text_mask'].sum()),clock_audio_seconds=float(grid_dt@grid['audio_mask']),clock_expression_seconds=float(grid_dt@grid['expression_mask']),feature_file='samples/'+dst.name,sha256=metadata['feature_sha256']))
    for name,data in [('全100条特征汇总.csv',rows),('全词音高与时间对照.csv',wordrows)]:
        with (O/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
    valid=[r for r in wordrows if r['pitch_valid']];noalign=[r for r in rows if r['supported_words']==0]
    result=dict(samples=100,words=sum(r['words'] for r in rows),supported_words=sum(r['supported_words'] for r in rows),valid_pitch_words=len(valid),word_audio_valid=sum(r['word_audio_valid'] for r in rows),word_expression_valid=sum(r['word_expression_valid'] for r in rows),clock_rows=totalgrid,
        median_F0_zero_fill_depression_hz=float(np.median([r['new_conditional_F0_hz']-r['old_zero_filled_F0_hz'] for r in valid])),median_pitch_voiced_fraction=float(np.median([r['voiced_fraction'] for r in valid])),
        clips_without_supported_alignment=len(noalign),such_clips_with_clock_audio=sum(r['clock_audio_valid']>0 for r in noalign),such_clips_with_clock_expression=sum(r['clock_expression_valid']>0 for r in noalign),
        pitch_identity_error=max(identity),regenerated_membership_max_delta=max(membdiff),feature_npz_bytes=sum(p.stat().st_size for p in (O/'samples').glob('*.npz')),seconds=time.perf_counter()-start)
    dump(O/'results.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
