import os
import json,csv
from pathlib import Path
import numpy as np
W=Path(os.environ.get('Q1_EXTERNAL_DIR',str(Path(__file__).resolve().parent.parent)));R=Path(os.environ.get('Q1_WORKSPACE',str(W.parent)));F=Path(os.environ.get('Q1_FEATURES',str(W.parent/'features/samples')))
public=json.loads((W/'public_words.json').read_text(encoding='utf-8'));anchors={r['sample_id']:r for r in json.loads((W/'waveform_anchors.json').read_text())}
rows=[];samples=[]
for p in sorted(F.glob('*.json'),key=lambda p:json.loads(p.read_text(encoding='utf-8'))['table_row_0based']):
    m=json.loads(p.read_text(encoding='utf-8'));sid=m['sample_id'];vid=sid.split('$_$')[0];a=anchors[sid];z=np.load(p.with_suffix('.npz'));pw=public[vid];tokens=[x['word'] for x in pw];ref=[x['word'] for x in m['words']]
    starts=[j for j in range(len(tokens)-len(ref)+1) if tokens[j:j+len(ref)]==ref]
    sr=dict(sample_id=sid,source_video=vid,exact_occurrences=len(starts),anchor_accepted=a['accepted'],local_words=len(ref),supported=int(z['alignment_supported'].sum()),eligible_words=0,outside_words=0)
    if len(starts)==1 and a['accepted']:
        for i,x in enumerate(pw[starts[0]:starts[0]+len(ref)]):
            ps=x['start']-a['source_offset_s'];pe=x['end']-a['source_offset_s'];eligible=pe>ps and ps>=m['audio_start_s'] and pe<=m['audio_start_s']+a['audio_length_s']
            sr['eligible_words']+=int(eligible);sr['outside_words']+=int(not eligible)
            row=dict(sample_id=sid,source_video=vid,word_index=i,word=ref[i],public_index=x['index'],public_start_local_s=ps,public_end_local_s=pe,eligible=eligible,supported=bool(z['alignment_supported'][i]))
            methods={'MMS_FA':(z['word_start_quantiles'][:,1],z['word_end_quantiles'][:,1]),'uniform':(np.arange(len(ref))*a['audio_length_s']/len(ref),np.arange(1,len(ref)+1)*a['audio_length_s']/len(ref))}
            q=Path(os.environ.get('Q1_PROJECT','C:/WorkArea/Huawei-Model'))/'results/q1_deep_v6/alignment/WAV2VEC2_ASR_BASE_960H'/p.stem/'original.npz'
            zz=np.load(q);methods['WAV2VEC2']=(zz['conv_anchor_start'][:,1],zz['conv_anchor_end'][:,1])
            methods['equal_fusion']=tuple((a+b)/2 for a,b in zip(methods['MMS_FA'],methods['WAV2VEC2']))
            for method,(s,e) in methods.items():
                row[method+'_start_s']=float(s[i]);row[method+'_end_s']=float(e[i]);row[method+'_start_diff_s']=float(s[i]-ps);row[method+'_end_diff_s']=float(e[i]-pe);row[method+'_max_abs_s']=max(abs(float(s[i]-ps)),abs(float(e[i]-pe)))
            rows.append(row)
    samples.append(sr)
def metric(rr,method):
    start=np.array([r[method+'_start_diff_s'] for r in rr]);end=np.array([r[method+'_end_diff_s'] for r in rr]);e=np.maximum(abs(start),abs(end));both=np.r_[abs(start),abs(end)]
    return dict(words=len(rr),samples=len({r['sample_id'] for r in rr}),source_videos=len({r['source_video'] for r in rr}),boundary_mae_s=float(both.mean()),boundary_median_s=float(np.median(both)),boundary_p95_s=float(np.quantile(both,.95)),max_boundary_median_s=float(np.median(e)),max_boundary_p95_s=float(np.quantile(e,.95)),within50ms=float(np.mean(e<=.05)),within100ms=float(np.mean(e<=.1)),within200ms=float(np.mean(e<=.2)),duration_mae_s=float(np.mean(abs(end-start))))
result={}
for name,rr in [('all_eligible',[r for r in rows if r['eligible']]),('current_supported',[r for r in rows if r['eligible'] and r['supported']]),('current_unsupported',[r for r in rows if r['eligible'] and not r['supported']])]:
    result[name]={method:metric(rr,method) for method in ['MMS_FA','WAV2VEC2','uniform','equal_fusion']}
    vids=sorted({r['source_video'] for r in rr});groups=[[r for r in rr if r['source_video']==v] for v in vids];rng=np.random.default_rng(20260924);boot=[]
    for _ in range(2000):
        sample=[r for ix in rng.integers(0,len(groups),len(groups)) for r in groups[ix]]
        boot.append(np.mean([r['MMS_FA_max_abs_s']<=.1 for r in sample]))
    result[name]['MMS_FA']['within100ms_cluster_bootstrap95']=np.quantile(boot,[.025,.975]).tolist()
result['coverage']=dict(total_samples=len(samples),total_words=sum(r['local_words'] for r in samples),unique_exact_transcript_samples=sum(r['exact_occurrences']==1 for r in samples),anchor_accepted_samples=sum(r['anchor_accepted'] for r in samples),cohort_samples=sum(r['exact_occurrences']==1 and r['anchor_accepted'] for r in samples),out_of_audio_words=sum(r['outside_words'] for r in samples))
(W/'word_summary.json').write_text(json.dumps(result,indent=2));(W/'word_records.json').write_text(json.dumps(rows,indent=2));(W/'sample_eligibility.json').write_text(json.dumps(samples,indent=2))
for name,data in [('逐词公开参考差异',rows),('全100条外部验证覆盖',samples)]:
    with (W/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
print(json.dumps(result,indent=2))
