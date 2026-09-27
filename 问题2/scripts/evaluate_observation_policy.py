"""Paired frozen-model validation; writes status instead of inventing missing scores."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from q2.data import load_pickle,convert,normalize,save_json,digest
from q2.observation_study import SCENARIOS,apply_condition

def main():
 p=argparse.ArgumentParser();p.add_argument('--aligned',type=Path,required=True);p.add_argument('--checkpoint',type=Path,default=ROOT/'final/model.pt');p.add_argument('--scaler',type=Path,default=ROOT/'final/scaler.npz');p.add_argument('--output',type=Path,required=True);p.add_argument('--special',type=Path,help='Optional input-state checks only; no special-set scoring');p.add_argument('--device',default='cpu');a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 raw=load_pickle(a.aligned)
 report={'protocol':'q2-observation-validation-v1','split':'valid','seed':20260926,'policies':['retain','mask_unk'],'scenarios':SCENARIOS,'selection':'Diagnostic comparison only; no automatic policy promotion','training_or_special_labels_used_for_policy_comparison':False,'source_sha256':digest(a.aligned),'input_counts':{},'validation_conditions':[]}
 for split in ['train','valid']:
  _,audit=convert(raw[split],raw[split]['id']);report['input_counts'][split]=audit
 if a.special:
  report['special_input_counts']=[]
  for path in sorted(a.special.glob('*.pkl')):
   item=load_pickle(path)['test'];ids=[f'{path.name}::row{i:03d}' for i in range(len(item['text_bert']))]
   for policy in report['policies']:
    data,audit=convert(item,ids,unknown_policy=policy)
    report['special_input_counts'].append({'file':path.name,'sha256':digest(path),'policy':policy,'unknown_positions':audit['text_unknown_positions'],'unknown_observed_as_text':int((data['text_unknown']&data['observed'][:,:,0]).sum())})
 valid,_=convert(raw['valid'],raw['valid']['id']);del raw
 for scenario in SCENARIOS:
  for policy in report['policies']:
   data,selected=apply_condition(valid,scenario,policy)
   report['validation_conditions'].append({'scenario':scenario,'policy':policy,'selected_positions':int(selected.sum()),'unknown_positions':int(data['text_unknown'].sum()),'observed_counts':data['observed'].sum(axis=(0,1)).tolist()})
 missing=[str(p) for p in [a.checkpoint,a.scaler] if not p.is_file()]
 if missing:
  report.update(status='model_validation_pending',missing_files=missing,metrics=None);save_json(a.output/'results.json',report);print(json.dumps({'status':report['status'],'missing_files':missing}));return
 from q2v2.deploy import load_deployment
 from q2.engine import infer,write_predictions
 from q2.metrics import metrics
 model,_=load_deployment(a.checkpoint,a.device)
 with np.load(a.scaler,allow_pickle=False) as stats:valid=normalize(valid,stats)
 scores=[];artifacts=[];paired={};masks={}
 for scenario in SCENARIOS:
  for policy in report['policies']:
   data,selected=apply_condition(valid,scenario,policy);prob,value=infer(model,data,a.device,batch_size=32)
   if scenario in masks and not np.array_equal(masks[scenario],selected):raise AssertionError('Unpaired perturbation masks')
   masks[scenario]=selected
   path=a.output/'predictions'/f'{scenario}__{policy}.csv'
   write_predictions(path,data,prob,value,scenario)
   artifacts.append({'path':str(path.relative_to(a.output)),'sha256':digest(path),'rows':len(value)})
   paired[(scenario,policy)]={'labels':data['labels'],'targets':data['targets'],'classes':prob.argmax(1),'predictions':value}
   scores.append({'scenario':scenario,'policy':policy,**metrics(data['labels'],data['targets'],prob,value)})
 mask_path=a.output/'selected_positions.npz';np.savez_compressed(mask_path,ids=valid['ids'],**masks)
 from q2v3.analysis import cluster_weights,boot_metrics
 weights,groups=cluster_weights(valid['ids'],replicates=2000,seed=20260926)
 intervals=[];draws={}
 for scenario in SCENARIOS:
  boots={policy:boot_metrics(paired[(scenario,policy)],weights) for policy in report['policies']}
  for metric in ['macro_f1','mae']:
   delta=boots['mask_unk'][metric]-boots['retain'][metric]
   draws[(scenario,metric)]=delta
   by_policy={s['policy']:s[metric] for s in scores if s['scenario']==scenario}
   intervals.append({'scenario':scenario,'metric':metric,'difference':by_policy['mask_unk']-by_policy['retain'],'lower':float(np.quantile(delta,.025)),'upper':float(np.quantile(delta,.975))})
 for metric in ['macro_f1','mae']:
  delta=np.mean([draws[(s,metric)] for s in SCENARIOS if s!='clean'],axis=0)
  points=[r['difference'] for r in intervals if r['scenario']!='clean' and r['metric']==metric]
  intervals.append({'scenario':'perturbed_mean','metric':metric,'difference':float(np.mean(points)),'lower':float(np.quantile(delta,.025)),'upper':float(np.quantile(delta,.975))})
 report.update(status='completed',metrics=scores,checkpoint_sha256=digest(a.checkpoint),scaler_sha256=digest(a.scaler),predictions=artifacts,mask_sha256=digest(mask_path),paired_intervals=intervals,bootstrap={'replicates':2000,'seed':20260926,'clusters':len(groups),'unit':'video ID before $_$','difference':'mask_unk minus retain','interval':'pointwise percentile 95%; no multiplicity correction','scope':'Conditional on this fitted model and fixed perturbations; not training-seed uncertainty'})
 save_json(a.output/'results.json',report);print('Completed 26 paired validation conditions')
if __name__=='__main__':main()
