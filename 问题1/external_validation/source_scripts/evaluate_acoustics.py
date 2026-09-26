import os
import json,csv
from pathlib import Path
import numpy as np
W=Path(os.environ.get('Q1_EXTERNAL_DIR',str(Path(__file__).resolve().parent.parent)));R=Path(os.environ.get('Q1_WORKSPACE',str(W.parent)));F=Path(os.environ.get('Q1_FEATURES',str(W.parent/'features/samples')))
anchors={r['sample_id']:r for r in json.loads((W/'waveform_anchors.json').read_text())};rows=[];allframes=[];out=W/'reference_frames';out.mkdir(exist_ok=True)
for p in F.glob('*.json'):
    m=json.loads(p.read_text(encoding='utf-8'));sid=m['sample_id'];vid=sid.split('$_$')[0];a=anchors[sid];cp=W/'csd_subset/CMU_MOSEI_COVAREP'/(vid+'.npz')
    if not cp.exists() or not a['accepted']:continue
    z=np.load(p.with_suffix('.npz'));c=np.load(cp);ct=c['intervals'].mean(1);t=z['audio_time']+a['source_offset_s'];ix=np.searchsorted(ct,t);ix=np.clip(ix,0,len(ct)-1);prev=np.maximum(0,ix-1);ix=np.where(abs(ct[prev]-t)<abs(ct[ix]-t),prev,ix)
    valid=np.isfinite(c['features'][ix,:2]).all(1)&(abs(ct[ix]-t)<=.015)
    f0=z['audio_native'][:,45];voice=z['audio_voiced'].astype(bool);ref=c['features'][ix,0];rv=c['features'][ix,1]>.5;both=valid&voice&rv&(f0>0)&(ref>0)
    semi=np.full(len(f0),np.nan);semi[both]=12*np.log2(f0[both]/ref[both])
    mcep_col=2 if c['features'].shape[1]==3 else 11
    corr=float(np.corrcoef(z['audio_native'][valid,39],c['features'][ix[valid],mcep_col])[0,1]) if valid.sum()>3 else None
    r=dict(sample_id=sid,source_video=vid,frames=int(valid.sum()),joint_voiced_frames=int(both.sum()),voicing_agreement=float(np.mean(voice[valid]==rv[valid])),pitch_abs_semitone_median=float(np.median(abs(semi[both]))) if both.any() else None,pitch_abs_semitone_p95=float(np.quantile(abs(semi[both]),.95)) if both.any() else None,pitch_within_one_semitone=float(np.mean(abs(semi[both])<=1)) if both.any() else None,logrms_mcep0_correlation=corr)
    rows.append(r);allframes.append((valid,voice,rv,semi))
    np.savez_compressed(out/(p.stem+'.npz'),local_time=z['audio_time'],source_time=t,public_time=ct[ix],time_eligible=valid,local_f0=f0,local_voiced=voice,public_f0=ref,public_voiced=rv,semitone_difference=semi,public_mcep0=c['features'][ix,mcep_col])
v=np.concatenate([r[0] for r in allframes]);l=np.concatenate([r[1] for r in allframes]);r=np.concatenate([r[2] for r in allframes]);e=np.concatenate([r[3] for r in allframes]);e=e[np.isfinite(e)]
summary=dict(available_source_videos=len(list((W/'csd_subset/CMU_MOSEI_COVAREP').glob('*.npz'))),samples=len(rows),frames=int(v.sum()),joint_voiced_frames=len(e),voicing_agreement=float(np.mean(l[v]==r[v])),voicing_confusion=dict(both_voiced=int(np.sum(v&l&r)),both_unvoiced=int(np.sum(v&~l&~r)),local_only=int(np.sum(v&l&~r)),public_only=int(np.sum(v&~l&r))),pitch_abs_semitone_median=float(np.median(abs(e))),pitch_abs_semitone_p95=float(np.quantile(abs(e),.95)),pitch_within_one_semitone=float(np.mean(abs(e)<=1)),pitch_within_two_semitones=float(np.mean(abs(e)<=2)),pitch_over_octave=float(np.mean(abs(e)>12)),video_grouping='See per-sample rows; pooled frame figures are descriptive, not independent Bernoulli trials')
(W/'acoustic_summary.json').write_text(json.dumps(summary,indent=2));(W/'acoustic_samples.json').write_text(json.dumps(rows,indent=2))
with (W/'逐样本声学公开对照.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(summary,indent=2));print('lowest energy correspondence',sorted(rows,key=lambda r:r['logrms_mcep0_correlation'])[:5])
