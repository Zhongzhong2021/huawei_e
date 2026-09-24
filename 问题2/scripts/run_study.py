"""Bounded, predeclared development and one three-seed confirmation."""
import gc
import json
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from q2.data import load_data, save_json, digest
from q2v2.data import load_fold
from q2v2.engine import train_run

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'round2'
NEW_STOP = datetime.fromisoformat('2026-09-24T12:26:25+00:00')
STOP = datetime.fromisoformat('2026-09-24T13:26:25+00:00')
def now(): return datetime.now(timezone.utc)
def read(p): return json.loads(Path(p).read_text())
def summarize(ms):
    return {k:float(np.mean(v)) for k,v in {
        'clean_f1':[m['clean']['macro_f1'] for m in ms],
        'clean_mae':[m['clean']['mae'] for m in ms],
        'missing_f1':[m['selection']['mean_missing_macro_f1'] for m in ms],
        'missing_mae':[m['selection']['mean_missing_mae'] for m in ms],
        'score':[m['selection']['score'] for m in ms]}.items()}
def main():
    if (ROOT/'study/frozen.json').exists():
        raise RuntimeError('Already frozen; use a new experiment root')
    oldprotocol = read(OLD/'study/protocol.json')
    cfg = dict(kind='bert', layers=list(range(12)), pretrained=True,
               multimodal=True, augmentation='span', distillation=False, seed=42, max_epochs=6)
    protocol = {'version':'round3-predeclared-v1', 'started_utc':now().isoformat(),
        'new_configuration_deadline':NEW_STOP.isoformat(), 'optimization_deadline':STOP.isoformat(),
        'delivery_deadline':'2026-09-24T17:26:25+00:00',
        'reference':'round2 bert2_distill FP32, same three video-group folds',
        'candidates':['teacher12_compact_reuse','bert12_full_span'],
        'ablation_only':['bert12_full_none'], 'seeds':[42,2026,3407],
        'selection_score':oldprotocol['selection_score'], 'guardrails':oldprotocol['guardrails'],
        'training':{**oldprotocol['training'],'max_epochs':6},
        'vocabulary':'Full public pretrained 30522 IDs; never derived from valid/test/attachment3',
        'rules':'Three complete folds required; rounded median best epoch; one winner confirmed; >=2 paired seed passes and mean score/clean guards; no fallback search on valid',
        'test_policy':'Frozen-model descriptive evaluation only; test previously observed',
        'bootstrap':{'replicates':2000,'seed':240924,'unit':'video group','pairing':'same sampled groups for both models and all scenarios','interval':'95% percentile conditional on fixed seed42 predictions'},
        'case_rule':'Per true class: largest-MAE misclassification and median-MAE correct case, sorted by sample ID to break ties; absent groups omitted',
        'no_size_filter':True}
    if (ROOT/'study/protocol.json').exists():
        protocol = read(ROOT/'study/protocol.json')
    else: save_json(ROOT/'study/protocol.json',protocol)
    source = torch.load(OLD/'pretrained/bert_base_uncased.pt',map_location='cpu',weights_only=True)
    references = [read(OLD/f'runs/bert2_distill_fold{f}/metrics.json') for f in range(3)]
    base = summarize(references)
    compact = [read(OLD/f'runs/teacher12_fold{f}/metrics.json') for f in range(3)]
    dev = {'reference':references,'reference_summary':base,'entries':[], 'events':[]}
    def entry(name,metrics,config,eligible_model,origin):
        mean=summarize(metrics)
        improvements=sum(m['selection']['score']>r['selection']['score'] for m,r in zip(metrics,references))
        eligible=eligible_model and improvements>=2 and mean['clean_f1']>=base['clean_f1']-.01 and mean['clean_mae']<=base['clean_mae']+.03
        return dict(name=name,metrics=metrics,summary=mean,config=config,origin=origin,
                    fold_improvements=improvements,eligible=eligible,
                    fixed_epochs=int(np.floor(np.median([m['run_metadata']['best_epoch'] for m in metrics])+.5)))
    # Reuse only immutable old source, preprocessing and 6-epoch training records.
    for f in range(3):
        oldcfg=read(OLD/f'runs/teacher12_fold{f}/config.json')
        assert oldcfg==cfg
        assert not (OLD/f'runs/teacher12_fold{f}/events.log').exists(), 'Old OOM protocol mismatch'
    dev['entries'].append(entry('teacher12_compact_reuse',compact,cfg,True,'round2'))
    save_json(ROOT/'study/development.json',dev)
    for name,augmentation in [('bert12_full_span','span'),('bert12_full_none','none')]:
        if now()>=NEW_STOP:
            dev['events'].append({'name':name,'status':'not_started_deadline','at':now().isoformat()})
            continue
        config={**cfg,'augmentation':augmentation,'vocabulary_mode':'full'}
        records=[]
        try:
            for f in range(3):
                if now()>=STOP: raise TimeoutError('Optimization deadline')
                train,valid=load_fold(OLD,f,'train'),load_fold(OLD,f,'valid')
                records.append(train_run(ROOT,f'{name}_fold{f}',config,train,valid,np.arange(30522),source=source))
                del train,valid
                gc.collect(); torch.cuda.empty_cache()
            dev['entries'].append(entry(name,records,config,augmentation=='span','round3'))
        except Exception as exc:
            dev['events'].append({'name':name,'status':'failed_or_incomplete','completed_folds':len(records),'error':repr(exc),'traceback':traceback.format_exc()})
        save_json(ROOT/'study/development.json',dev)
    candidates=[e for e in dev['entries'] if e['eligible']]
    selected=max(candidates,key=lambda e:e['summary']['score']) if candidates else None
    save_json(ROOT/'study/internal_selection.json',{'selected':selected,'selected_utc':now().isoformat(),'no_test_access':True})
    records=[]; pairs=[]; stable=False
    if selected and now()<STOP:
        training,valid=load_data(ROOT.parent,'train'),load_data(ROOT.parent,'valid')
        vocabulary=np.arange(30522) if selected['name']=='bert12_full_span' else np.load(OLD/'study/full_train_vocabulary.npy')
        for seed in [42,2026,3407]:
            if now()>=STOP: break
            name=f"confirmed_{selected['name']}_s{seed}"
            config={**selected['config'],'seed':seed}
            try:
                if selected['name']=='teacher12_compact_reuse' and seed==42:
                    old=OLD/'runs/teacher12_full'
                    assert read(old/'provenance.json')['fixed_epochs']==selected['fixed_epochs']
                    shutil.copytree(old,ROOT/'runs'/name,dirs_exist_ok=False)
                    result=read(old/'metrics.json')
                else:
                    result=train_run(ROOT,name,config,training,valid,vocabulary,source=source,fixed_epochs=selected['fixed_epochs'])
                reference=read(OLD/f'runs/confirmed_bert2_distill_s{seed}/metrics.json')
                delta={'score':result['selection']['score']-reference['selection']['score'],
                       'clean_f1':result['clean']['macro_f1']-reference['clean']['macro_f1'],
                       'clean_mae':result['clean']['mae']-reference['clean']['mae']}
                pairs.append({'seed':seed,'deltas':delta,'passed':delta['score']>0 and delta['clean_f1']>=-.01 and delta['clean_mae']<=.03})
                records.append({'seed':seed,'run':name,'metrics':result,'reference':reference})
                save_json(ROOT/'study/confirmation_progress.json',{'records':records,'pairs':pairs})
                gc.collect(); torch.cuda.empty_cache()
            except Exception as exc:
                save_json(ROOT/'audit/confirmation_failure.json',{'seed':seed,'error':repr(exc),'traceback':traceback.format_exc()}); break
    if len(records)==3 and now()<=STOP:
        a=summarize([r['metrics'] for r in records]); b=summarize([r['reference'] for r in records])
        stable=sum(p['passed'] for p in pairs)>=2 and a['score']>b['score'] and a['clean_f1']>=b['clean_f1']-.01 and a['clean_mae']<=b['clean_mae']+.03
    confirmed={'stable_improvement':stable,'selected':selected,'records':records,'pairs':pairs,'completed_utc':now().isoformat()}
    save_json(ROOT/'study/confirmed.json',confirmed)
    chosen=ROOT/'runs'/records[0]['run']/'best.pt' if stable else OLD/'runs/confirmed_bert2_distill_s42/best.pt'
    final=ROOT/'final'; final.mkdir(exist_ok=True)
    shutil.copy2(chosen,final/'model.pt'); shutil.copy2(ROOT.parent/'data/processed/scaler.npz',final/'scaler.npz')
    save_json(ROOT/'study/frozen.json',{'frozen_utc':now().isoformat(),'selected_checkpoint':str(chosen),
        'model_name':selected['name'] if stable else 'round2_bert2_distill_fp32',
        'stable_improvement':stable,'checkpoint_sha256':digest(final/'model.pt'),'scaler_sha256':digest(final/'scaler.npz'),
        'protocol_sha256':digest(ROOT/'study/protocol.json'),'test_used_for_selection':False,
        'test_role':'previously observed; descriptive only','quantization':None})
    print(json.dumps({'stable':stable,'pairs':pairs,'selected':str(chosen)},indent=2),flush=True)

if __name__=='__main__': main()
