"""Independent whole-corpus checks and grouped perturbation summaries."""
import json,csv,hashlib,time,argparse
from pathlib import Path
import numpy as np
from q1v6_features import load_sample,collate
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',default='.');args=ap.parse_args();P=Path(args.project);base=P/'results/q1_deep_v6';O=base/'validation';O.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    O=base/'validation_v2';O.mkdir(parents=True,exist_ok=True)
    files=sorted((base/'features_v2/samples').glob('*.npz'));assert len(files)==100;rows=[];checks=0;poolerrs=[];grids={};wordbatch=[];clockbatch=[]
    for p in files:
        z,m=load_sample(base/'features_v2',p.stem,'native');legacy=np.load(P/'results/q1/samples'/p.name);face=np.load(P/'results/q1_refinement_v1/faces'/p.name)
        assert len(z['text'])==len(m['words']);assert all(sha(P/q)==v for q,v in m['dependencies'].items());checks+=len(m['dependencies'])
        for key in ['audio_native','audio_voiced','audio_time','audio_cells','text','bert_token_ids','bert_word_membership','bert_char_offsets','scene_native']:assert np.array_equal(z[key],legacy[key]);checks+=1
        assert np.array_equal(z['expression_native'],face['vision_expression_native']);checks+=1
        for k in ['audio_time','vision_time']:assert np.all(np.diff(z[k])>0);checks+=1
        for k in ['audio_cells','vision_cells','ctc_cells']:assert np.all(np.diff(z[k])>0);assert np.allclose(z[k][:-1,1],z[k][1:,0]);checks+=2
        for key,mask in [('word_audio','word_audio_dim_mask'),('word_expression','word_expression_mask'),('word_scene','word_scene_mask')]:assert np.all(z[key][~z[mask]]==0);checks+=1
        assert np.all(z['word_start_quantiles'][:,0]<=z['word_start_quantiles'][:,1]) and np.all(z['word_end_quantiles'][:,1]<=z['word_end_quantiles'][:,2]);checks+=1
        # Independent scalar quadrature for three words and both conditional pitch moments.
        for j in sorted(set([0,len(z['text'])//2,len(z['text'])-1])):
            weight=[]
            for a,b in z['audio_cells']:
                weight.append(sum(float(g)*max(0,min(b,d)-max(a,c)) for g,(c,d) in zip(z['word_time_membership'][j],z['ctc_cells'])))
            w=np.array(weight);wv=w*z['audio_voiced'];x=z['audio_native'][:,45].astype(float);den=wv.sum()
            mean=float(wv@x/den) if den>0 else 0.;sd=np.sqrt(max(float(wv@(x*x)/den)-mean*mean,0)) if den>0 else 0
            for col,value in [(45,mean),(92,sd),(96,float(den/max(w.sum(),1e-12)))]:
                expected=value if z['word_audio_dim_mask'][j,col] else 0.;poolerrs.append(abs(expected-float(z['word_audio'][j,col]))/max(1,abs(expected)));checks+=1
        # Physical coverage independent of clock granularity; no text gate enters AV pooling.
        coverage={}
        for step in [.05,.1,.2]:
            g,_=load_sample(base/'features_v2',p.stem,'clock',step);assert np.allclose(g['cells'][1:,0],g['cells'][:-1,1]);assert abs(g['cells'][-1,1]-m['duration_s'])<1e-8
            coverage[str(step)]={key:float(g[key].sum()) for key in ['audio_effective_seconds','expression_effective_seconds','scene_effective_seconds']}
            assert np.isfinite(g['audio']).all() and np.all(g['audio'][~g['audio_dim_mask']]==0)
            if step==.1:clockbatch.append(g)
        for key in coverage['0.1']:assert max(abs(coverage[s][key]-coverage['0.1'][key]) for s in coverage)<1e-7
        grids[p.stem]=coverage;wordbatch.append(load_sample(base/'features_v2',p.stem,'word')[0]);rows.append(dict(stem=p.stem,sample_id=m['sample_id'],checks=checks))
    for samples in [wordbatch,clockbatch]:
        b=collate(samples);assert len(b['lengths'])==100
        for key,v in b.items():
            if v.ndim>=2 and v.shape[:2]==b['sequence_mask'].shape:assert np.all(v[~b['sequence_mask']]==0)
    assert max(poolerrs)<1e-5
    allrows=[]
    for name in ['MMS_FA','WAV2VEC2_ASR_BASE_960H']:
        for p in files:
            with (base/'alignment'/name/p.stem/'word_interventions.csv').open(encoding='utf-8-sig') as f:
                for r in csv.DictReader(f):r['supported']=r['supported']=='True';r['max_boundary_error_s']=float(r['max_boundary_error_s']);allrows.append(r)
    summaries=[]
    for name in ['MMS_FA','WAV2VEC2_ASR_BASE_960H']:
        for cond in ['prepend_silence_0.6s','gain_0.5','white_noise_20dB']:
            for group in ['all','supported','unsupported']:
                rr=[r for r in allrows if r['model']==name and r['condition']==cond and r['mapping']=='conv_anchor' and (group=='all' or r['supported']==(group=='supported'))]
                vals=np.array([r['max_boundary_error_s'] for r in rr]);summaries.append(dict(model=name,condition=cond,group=group,n=len(vals),median=float(np.median(vals)),p95=float(np.quantile(vals,.95)),over100ms=int((vals>.1+1e-8).sum()),over200ms=int((vals>.2+1e-8).sum())))
    # Paired video-level bootstrap of model differences; word-weighted statistic.
    groups=sorted(set(r['video_id'] for r in allrows));rng=np.random.default_rng(62026);mult=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=2000);groupci=[]
    for cond in ['prepend_silence_0.6s','gain_0.5','white_noise_20dB']:
        terms=[]
        for model in ['MMS_FA','WAV2VEC2_ASR_BASE_960H']:
            counts=np.zeros(len(groups));total=np.zeros(len(groups))
            for r in allrows:
                if r['model']==model and r['condition']==cond and r['mapping']=='conv_anchor' and r['supported']:
                    i=groups.index(r['video_id']);total[i]+=1;counts[i]+=r['max_boundary_error_s']>.1+1e-8
            terms.append((counts,total))
        delta=(mult@terms[1][0]/(mult@terms[1][1]))-(mult@terms[0][0]/(mult@terms[0][1]));point=terms[1][0].sum()/terms[1][1].sum()-terms[0][0].sum()/terms[0][1].sum()
        groupci.append(dict(condition=cond,estimand='Wav2vec minus MMS fraction >100ms on fixed supported words',estimate=float(point),percentile95=np.quantile(delta,[.025,.975]).tolist()))
    diff=[];pairrows=[]
    for p in files:
        a=np.load(base/'alignment/MMS_FA'/p.stem/'original.npz');b=np.load(base/'alignment/WAV2VEC2_ASR_BASE_960H'/p.stem/'original.npz');z=np.load(p)
        delta=np.maximum(abs(a['conv_anchor_start'][:,1]-b['conv_anchor_start'][:,1]),abs(a['conv_anchor_end'][:,1]-b['conv_anchor_end'][:,1]));diff.extend(delta[z['alignment_supported']]);pairrows.append(dict(stem=p.stem,supported=int(z['alignment_supported'].sum()),model_boundary_disagreement_median_s=float(np.median(delta)),over200ms_all=int((delta>.2+1e-8).sum())))
    result=dict(samples=100,source_hash_and_structural_checks=checks,independent_pitch_pool_scaled_max_error=max(poolerrs),all_native_features_unchanged=True,word_and_clock_collate_padding_passed=True,
        clock_steps_checked=[.05,.1,.2],clock_effective_duration_conserved=True,perturbation_summaries=summaries,video_groups=len(groups),bootstrap_reps=2000,bootstrap_seed=62026,paired_group_intervals=groupci,
        supported_cross_model_boundary_disagreement=dict(n=len(diff),median=float(np.median(diff)),p95=float(np.quantile(diff,.95)),over200ms=int((np.array(diff)>.2+1e-8).sum())),
        threshold_numeric_tolerance_seconds=1e-8,limits='Conditional descriptions of fixed source videos and fixed supported words; thresholds are not calibrated accuracy guarantees. Do not count numerical100ms equality as >100ms.',seconds=time.perf_counter()-start)
    dump(O/'results.json',result);dump(O/'grid_duration_checks.json',grids);dump(O/'cross_model_by_sample.json',pairrows);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
