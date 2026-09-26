import os
import json
from pathlib import Path
import numpy as np
W=Path(os.environ.get('Q1_EXTERNAL_DIR',str(Path(__file__).resolve().parent.parent)));P=Path(os.environ.get('Q1_PROJECT','C:/WorkArea/Huawei-Model'));rows=[];arrays={k:[] for k in ['initial','relaxed_fullframe','roi_hybrid','union']};refs=[]
for p in (W/'visual_reference_frames').glob('*.npz'):
    v=np.load(p);g=v['time_eligible'];f=np.load(P/'results/q1_refinement_v1/faces'/p.name);base=np.load(P/'results/q1/samples'/p.name);assert len(base['vision_valid'])==len(g)
    masks=dict(initial=base['vision_valid'],relaxed_fullframe=f['relaxed_fullframe_valid'],roi_hybrid=f['vision_expression_valid']);masks['union']=masks['roi_hybrid']|masks['relaxed_fullframe'];ref=v['public_success'];refs.extend(ref[g].tolist())
    for key,x in masks.items():arrays[key].extend(x[g].tolist())
    rows.append(dict(sample_id=p.stem.replace('__','$_$'),frames=int(g.sum()),**{k:int((x&g).sum()) for k,x in masks.items()}))
r=np.asarray(refs);result={}
for k,x in arrays.items():
    x=np.asarray(x);result[k]=dict(frames=len(x),detected=int(x.sum()),agreement=float(np.mean(x==r)),local_only=int(np.sum(x&~r)),public_only=int(np.sum(~x&r)))
result['interpretation']='Previously executed detector variants compared on the same reference frames. Union is a coverage-only mask ablation; no new blendshape features or verified speaker identification are implied.'
(W/'face_variant_summary.json').write_text(json.dumps(result,indent=2));(W/'face_variant_samples.json').write_text(json.dumps(rows,indent=2));print(json.dumps(result,indent=2))
