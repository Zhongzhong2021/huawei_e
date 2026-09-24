"""Evaluate only after immutable freeze; no selection follows test access."""
import gc
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from q2.data import load_data, save_json, digest
from q2.engine import evaluate_model, save_scenarios, infer
from q2.missing import MAIN, SUPPLEMENT
from q2v2.deploy import load_deployment, predict

ROOT=Path(__file__).resolve().parents[1]
def main():
    freeze=json.loads((ROOT/'study/frozen.json').read_text())
    assert digest(ROOT/'final/model.pt')==freeze['checkpoint_sha256']
    reference=ROOT/'final/reference_fp32.pt'
    if not reference.exists(): shutil.copy2(ROOT.parent/'round2/runs/confirmed_bert2_distill_s42/best.pt',reference)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    for role,checkpoint in [('final',ROOT/'final/model.pt'),('reference',reference)]:
        model,record=load_deployment(checkpoint,device)
        for split in ['valid','test']:
            data=load_data(ROOT.parent,split)
            out=ROOT/'evaluation'/role/split
            if not (out/'metrics.json').exists():
                masks=save_scenarios(ROOT,data,split)
                result=evaluate_model(model,data,device,['clean']+MAIN+SUPPLEMENT,out=out/'predictions',batch_size=32,masks=masks)
                save_json(out/'metrics.json',result)
                print(role,split,result['clean'],flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    checks=predict(ROOT/'final/model.pt',ROOT/'final/scaler.npz',ROOT.parent/'data/raw/attachment3',ROOT/'final/attachment3_predictions.csv',device)
    save_json(ROOT/'audit/prediction_checks.json',checks)
    # CPU/GPU tolerance and malformed-observation safety of the actual final model.
    from q2v2.deploy import raw_inputs
    data=raw_inputs(ROOT.parent/'data/raw/attachment3',ROOT/'final/scaler.npz')
    model,_=load_deployment(ROOT/'final/model.pt','cpu')
    p_cpu,y_cpu=infer(model,data,'cpu',batch_size=32)
    empty={k:np.array(v[:2],copy=True) for k,v in data.items()}
    empty['valid'][:]=False; empty['observed'][:]=False
    ep,ey=infer(model,empty,'cpu',batch_size=2)
    assert np.isfinite(ep).all() and np.isfinite(ey).all()
    del model
    model,_=load_deployment(ROOT/'final/model.pt',device)
    p_gpu,y_gpu=infer(model,data,device,batch_size=32)
    assert np.allclose(p_cpu,p_gpu,atol=1e-5,rtol=0) and np.allclose(y_cpu,y_gpu,atol=1e-5,rtol=0)
    assert np.array_equal(p_cpu.argmax(1),p_gpu.argmax(1))
    save_json(ROOT/'audit/cpu_gpu.json',{'max_probability_difference':float(abs(p_cpu-p_gpu).max()),
       'max_intensity_difference':float(abs(y_cpu-y_gpu).max()),'class_equal':True,'empty_final_model_finite':True})
    # Archive old evidence without mutating old runs or duplicating all old weights.
    old=ROOT.parent/'round2'
    for folder in ['runs','study','audit','pretrained']:
        for source in (old/folder).rglob('*'):
            if source.is_file() and source.suffix in ['.json','.csv','.log','.txt','.md']:
                target=ROOT/'evidence/round2'/source.relative_to(old)
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    source=ROOT.parent/'data/processed/valid.npz'
    valid=load_data(ROOT.parent,'valid')
    np.savez_compressed(ROOT/'evidence/valid_metadata.npz',ids=valid['ids'],valid=valid['valid'],observed=valid['observed'])
    save_json(ROOT/'audit/evaluation_complete.json',{'checkpoint_sha256':digest(ROOT/'final/model.pt'),
        'scenarios_per_split':46,'models':2,'splits':['valid','test_descriptive'],'attachment3':checks})
if __name__=='__main__':main()
