import os
import json,csv
from pathlib import Path
import numpy as np
W=Path(os.environ.get('Q1_EXTERNAL_DIR',str(Path(__file__).resolve().parent.parent)));R=Path(os.environ.get('Q1_WORKSPACE',str(W.parent)));F=Path(os.environ.get('Q1_FEATURES',str(W.parent/'features/samples')));O=W/'visual_reference_frames';O.mkdir(exist_ok=True)
anchors={r['sample_id']:r for r in json.loads((W/'waveform_anchors.json').read_text())};rows=[];aggregates={}
for shift in [0,-.04,.04]:
    all_local=[];all_public=[];all_conf=[]
    for p in F.glob('*.json'):
        m=json.loads(p.read_text(encoding='utf-8'));sid=m['sample_id'];vid=sid.split('$_$')[0];a=anchors[sid];cp=W/'csd_subset/CMU_MOSEI_OpenFace2'/(vid+'.npz')
        if not cp.exists() or not a['accepted']:continue
        z=np.load(p.with_suffix('.npz'));c=np.load(cp);ct=c['features'][:,1];t=z['vision_time']+a['source_offset_s']+shift
        assert np.all(np.diff(ct)>=0)
        ix=np.searchsorted(ct,t);ix=np.clip(ix,0,len(ct)-1);prev=np.maximum(0,ix-1);ix=np.where(abs(ct[prev]-t)<abs(ct[ix]-t),prev,ix)
        valid=np.isfinite(c['features'][ix,2:4]).all(1)&(abs(ct[ix]-t)<=.04);local=z['expression_valid'].astype(bool);public=c['features'][ix,3]>.5;conf=c['features'][ix,2]
        all_local.extend(local[valid].tolist());all_public.extend(public[valid].tolist());all_conf.extend(conf[valid].tolist())
        if shift==0:
            r=dict(sample_id=sid,frames=int(valid.sum()),local_detected=int(np.sum(local&valid)),public_success=int(np.sum(public&valid)),both=int(np.sum(local&public&valid)),neither=int(np.sum(~local&~public&valid)),local_only=int(np.sum(local&~public&valid)),public_only=int(np.sum(~local&public&valid)));rows.append(r)
            np.savez_compressed(O/(p.stem+'.npz'),local_time=z['vision_time'],source_time=t,public_time=ct[ix],time_eligible=valid,local_detected=local,public_success=public,public_confidence=conf)
    l=np.asarray(all_local,dtype=bool);p=np.asarray(all_public,dtype=bool);conf=np.asarray(all_conf)
    aggregates[str(shift)]=dict(frames=len(l),agreement=float(np.mean(l==p)),both=int(np.sum(l&p)),neither=int(np.sum(~l&~p)),local_only=int(np.sum(l&~p)),public_only=int(np.sum(~l&p)),agreement_confidence05=float(np.mean(l==(p&(conf>=.5)))))
summary=dict(available_source_videos=len(list((W/'csd_subset/CMU_MOSEI_OpenFace2').glob('*.npz'))),samples=len(rows),nearest_frame_tolerance_s=.04,clock_shift_sensitivity=aggregates,interpretation='Comparison of detection outputs only; OpenFace success is not ground-truth visibility, and AU vectors are not compared with MediaPipe blendshapes. Audio-derived source origin reused under shared media clock; +/-40ms sensitivity reported.')
(W/'visual_summary.json').write_text(json.dumps(summary,indent=2));(W/'visual_samples.json').write_text(json.dumps(rows,indent=2))
with (W/'逐样本人脸检测公开对照.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(summary,indent=2))
