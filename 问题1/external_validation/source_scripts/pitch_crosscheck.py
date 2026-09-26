import os
import json
from pathlib import Path
import numpy as np
W=Path(os.environ.get('Q1_EXTERNAL_DIR',str(Path(__file__).resolve().parent.parent)));R=Path(os.environ.get('Q1_WORKSPACE',str(W.parent)));O=W/'pitch_crosscheck';O.mkdir(exist_ok=True)
rows=[];records=[]
for p in (Path(os.environ.get('Q1_FEATURES',str(W.parent/'features/samples')))).glob('*.npz'):
    z=np.load(p);e=np.load(W.parent/'egemaps/samples'/p.name);t=z['audio_time'];et=e['intervals'].mean(1);ix=np.searchsorted(et,t);ix=np.clip(ix,0,len(et)-1);prev=np.maximum(0,ix-1);ix=np.where(abs(et[prev]-t)<abs(et[ix]-t),prev,ix)
    semi=e['lld'][ix,10];ef0=np.where(semi>0,27.5*2.0**(semi/12),0);f0=z['audio_native'][:,45];both=z['audio_voiced'].astype(bool)&(semi>0)&(abs(et[ix]-t)<=.015);gap=np.full(len(t),np.nan);gap[both]=12*np.log2(f0[both]/ef0[both]);agree=both&(abs(gap)<=1)
    np.savez_compressed(O/p.name,time=t,pyin_f0=f0,egemaps_f0=ef0,joint_voiced=both,absolute_semitone_gap=np.where(both,abs(gap),0),within_one_semitone=agree)
    rr=dict(sample_id=p.stem.replace('__','$_$'),frames=len(t),joint_voiced=int(both.sum()),within_one_semitone=int(agree.sum()));rows.append(rr)
    ref=W/'reference_frames'/p.name
    if ref.exists():
        c=np.load(ref);valid=c['time_eligible']&c['public_voiced']&(c['public_f0']>0)&both;delta=c['semitone_difference'];records.extend([dict(sample_id=rr['sample_id'],agree=bool(agree[i]),gap=float(delta[i])) for i in np.flatnonzero(valid)])
summary=dict(samples=len(rows),frames=sum(r['frames'] for r in rows),joint_voiced=sum(r['joint_voiced'] for r in rows),within_one_semitone=sum(r['within_one_semitone'] for r in rows),rule='Optional diagnostic only: nearest eGeMAPS window midpoint <=15ms; both F0 outputs voiced and <=1 semitone apart. Public COVAREP is not used in this flag or any primary feature.')
for key,rr in [('all_joint',[r for r in records]),('agreement',[r for r in records if r['agree']]),('disagreement',[r for r in records if not r['agree']])]:
    a=abs(np.array([r['gap'] for r in rr]));summary[key]=dict(frames=len(a),median_semitones=float(np.median(a)),p95_semitones=float(np.quantile(a,.95)),within_one_semitone=float(np.mean(a<=1)))
(W/'pitch_crosscheck_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
