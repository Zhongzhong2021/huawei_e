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
 from q2.engine import infer
 from q2.metrics import metrics
 model,_=load_deployment(a.checkpoint,a.device)
 with np.load(a.scaler,allow_pickle=False) as stats:valid=normalize(valid,stats)
 scores=[]
 for scenario in SCENARIOS:
  for policy in report['policies']:
   data,_=apply_condition(valid,scenario,policy);prob,value=infer(model,data,a.device,batch_size=32)
   scores.append({'scenario':scenario,'policy':policy,**metrics(data['labels'],data['targets'],prob,value)})
 report.update(status='completed',metrics=scores,checkpoint_sha256=digest(a.checkpoint),scaler_sha256=digest(a.scaler));save_json(a.output/'results.json',report);print('Completed 26 paired validation conditions')
if __name__=='__main__':main()
