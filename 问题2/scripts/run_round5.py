"""Predeclared deferred-weighting study against the immutable round-four model."""
import argparse
from datetime import datetime, timedelta, timezone
import gc
import json
from pathlib import Path
import shutil
import traceback
import numpy as np
import torch
from q2.data import digest, load_data, save_json
from q2v2.data import load_fold
from q2v2.engine import train_run
from run_round4 import compare_tensors


def now(): return datetime.now(timezone.utc)
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def summarize(records):
    fields = {'score': lambda r:r['selection']['score'], 'clean_f1':lambda r:r['clean']['macro_f1'],
        'clean_mae':lambda r:r['clean']['mae'], 'accuracy':lambda r:r['clean']['accuracy'],
        'neutral_f1':lambda r:r['clean']['class_f1'][1], 'positive_f1':lambda r:r['clean']['class_f1'][2],
        'missing_f1':lambda r:r['selection']['mean_missing_macro_f1'], 'missing_mae':lambda r:r['selection']['mean_missing_mae']}
    return {k:float(np.mean([fn(r) for r in records])) for k,fn in fields.items()}


def guards(a, b):
    checks = {'score_improved': a['score'] > b['score'], 'clean_f1':a['clean_f1'] >= b['clean_f1']-.01,
        'clean_mae':a['clean_mae'] <= b['clean_mae']+.03, 'accuracy':a['accuracy'] >= b['accuracy']-.005,
        'neutral_f1':a['neutral_f1'] >= b['neutral_f1']-.01, 'positive_f1_improved':a['positive_f1'] > b['positive_f1']}
    return {'passed':all(checks.values()), 'checks':checks, 'deltas':{k:a[k]-b[k] for k in a}}


def main(root, data_root, old2, old4):
    if any(root == p or root in p.parents or p in root.parents for p in [old2, old4]) or root == data_root:
        raise ValueError('Independent round-five directory required')
    if (root / 'study/started.json').exists() or (root / 'study/frozen.json').exists():
        raise FileExistsError('Study already started: inspect its live process or terminal record; never restart blindly')
    for folder in ['study','audit','runs','final']: (root / folder).mkdir(parents=True, exist_ok=True)
    old_protocol = read(old4 / 'study/protocol.json')
    base_config = read(old4 / 'runs/balanced_inverse_fold0/config.json')
    reference = [read(old4 / f'runs/balanced_inverse_fold{i}/metrics.json') for i in range(3)]
    baseline = summarize(reference)
    started = now()
    protocol = {'version':'round5-deferred-weighting-predeclared-v1', 'started_utc':started.isoformat(),
        'new_configuration_deadline':(started+timedelta(hours=2)).isoformat(),
        'optimization_deadline':(started+timedelta(hours=3)).isoformat(), 'delivery_deadline':(started+timedelta(hours=4)).isoformat(),
        'reference':'round4 balanced_inverse full-vocabulary 12-layer multimodal model',
        'candidates':[{'name':'defer1','class_weight_delay_epochs':1},{'name':'defer2','class_weight_delay_epochs':2}],
        'training':old_protocol['training'], 'class_weight_power':1.,
        'schedule':'unweighted CE for epochs 1..d; frozen fitting-only inverse-frequency CE afterward; Huber and LR schedule unchanged',
        'checkpoint_selection':'Only epochs > d eligible; patience counted only after the switch. Median internal best epoch fixes confirmation length.',
        'selection_score':old_protocol['selection_score'],
        'guardrails':{'min_fold_score_improvements':2, 'clean_f1_max_drop':.01, 'clean_mae_max_increase':.03,
                     'accuracy_max_drop':.005, 'neutral_f1_max_drop':.01, 'positive_f1':'strictly improve'},
        'selection_rule':'At least 2/3 folds improve S, all mean guards pass; select highest mean eligible S; confirm only one candidate on 42/2026/3407, >=2/3 paired guards and all mean guards',
        'folds':3, 'development_seed':42, 'confirmation_seeds':[42,2026,3407],
        'controls':'Replay delay=0 fold0 with every tensor identical before reusing round4 evidence',
        'data_policy':'Official train only for fitting; internal validation for development; official validation once for selected recipe confirmation; test and attachment3 never used for selection',
        'scope':'DRW-inspired schedule adaptation, not LDAM loss, not a replication of the original paper, no long-tail performance guarantee',
        'reference_paper':{'authors':'Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga, Tengyu Ma',
            'title':'Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss', 'venue':'NeurIPS 2019',
            'url':'https://papers.neurips.cc/paper_files/paper/2019/hash/621461af90cadfdaf0e8d4cc25129f91-Abstract.html'},
        'statistical_limit':'Official validation reused across rounds; repeated adaptive development precludes unbiased generalization claims.'}
    save_json(root / 'study/protocol.json', protocol)
    save_json(root / 'study/started.json', {'started_utc':started.isoformat(), 'protocol_sha256':digest(root / 'study/protocol.json')})
    save_json(root / 'audit/source_hashes.json', {str(p.relative_to(root)):digest(p) for folder in ['src','scripts','tests'] for p in sorted((root / folder).rglob('*.py'))})
    source_path = old2 / 'pretrained/bert_base_uncased.pt'
    save_json(root / 'audit/input_hashes.json', {'pretrained_sha256':digest(source_path), 'reference_model_sha256':digest(old4 / 'final/model.pt'),
        'reference_freeze_sha256':digest(old4 / 'study/frozen.json'),
        'folds':{f'{f}/{s}':digest(old2 / f'folds/fold{f}/{s}.npz') for f in range(3) for s in ['train','valid','scaler']}})
    stop = datetime.fromisoformat(protocol['optimization_deadline']); new_stop = datetime.fromisoformat(protocol['new_configuration_deadline'])
    source = torch.load(source_path, map_location='cpu', weights_only=True)
    fit, val = load_fold(old2,0,'train'), load_fold(old2,0,'valid')
    control = train_run(root,'control_delay0_fold0',base_config,fit,val,np.arange(30522),source=source)
    comparison = compare_tensors(root / 'runs/control_delay0_fold0/best.pt',old4 / 'runs/balanced_inverse_fold0/best.pt')
    comparison['score_delta'] = control['selection']['score']-reference[0]['selection']['score']
    save_json(root / 'audit/control_reproduction.json', comparison)
    if not comparison['all_tensors_exact'] or comparison['score_delta'] != 0:
        save_json(root / 'study/terminal.json',{'status':'control_failed',**comparison})
        raise RuntimeError('Control failed; no new candidates allowed')
    del fit,val; gc.collect(); torch.cuda.empty_cache()
    development = {'reference':reference,'reference_summary':baseline,'entries':[],'events':[]}
    save_json(root / 'study/development.json',development)
    for candidate in protocol['candidates']:
        name = candidate['name']; records = []
        if now() >= new_stop:
            development['events'].append({'name':name,'status':'not_started_deadline'}); continue
        config = {**base_config, 'class_weight_delay_epochs':candidate['class_weight_delay_epochs']}
        try:
            for fold in range(3):
                if now() >= stop: raise TimeoutError('Optimization deadline reached')
                fit,val = load_fold(old2,fold,'train'),load_fold(old2,fold,'valid')
                records.append(train_run(root,f'{name}_fold{fold}',config,fit,val,np.arange(30522),source=source))
                del fit,val; gc.collect(); torch.cuda.empty_cache()
            summary = summarize(records); check = guards(summary,baseline)
            improvements = sum(a['selection']['score']>b['selection']['score'] for a,b in zip(records,reference))
            development['entries'].append({'name':name,'config':config,'metrics':records,'summary':summary,
                'guard':check,'fold_improvements':improvements,'eligible':improvements>=2 and check['passed'],
                'fixed_epochs':int(np.median([m['run_metadata']['best_epoch'] for m in records]))})
        except Exception as exc:
            development['events'].append({'name':name,'status':'failed_or_incomplete','completed_folds':len(records),'error':repr(exc),'traceback':traceback.format_exc()})
        save_json(root / 'study/development.json',development)
        print(json.dumps({'completed_candidate':name,'entries':[{k:e[k] for k in ['name','summary','guard','fold_improvements','eligible']} for e in development['entries']]}),flush=True)
    save_json(root / 'study/development.json',development)
    eligible = [e for e in development['entries'] if e['eligible']]
    selected = max(eligible,key=lambda e:e['summary']['score']) if eligible else None
    save_json(root / 'study/internal_selection.json',{'selected':selected,'selected_utc':now().isoformat(),'test_accessed':False})
    records=[]; pairs=[]
    if selected:
        fit,val=load_data(data_root,'train'),load_data(data_root,'valid')
        for seed in protocol['confirmation_seeds']:
            if now() >= stop: break
            name=f"confirmed_{selected['name']}_s{seed}"
            try:
                m=train_run(root,name,{**selected['config'],'seed':seed},fit,val,np.arange(30522),source=source,fixed_epochs=selected['fixed_epochs'])
                ref=read(old4 / f'runs/confirmed_balanced_inverse_s{seed}/metrics.json')
                records.append({'seed':seed,'run':name,'metrics':m,'reference':ref})
                pairs.append({'seed':seed,**guards(summarize([m]),summarize([ref]))})
                save_json(root / 'study/confirmation_progress.json',{'records':records,'pairs':pairs})
                gc.collect(); torch.cuda.empty_cache()
            except Exception as exc:
                save_json(root / 'audit/confirmation_failure.json',{'seed':seed,'error':repr(exc),'traceback':traceback.format_exc()}); break
    mean_guard = guards(summarize([r['metrics'] for r in records]),summarize([r['reference'] for r in records])) if len(records)==3 else None
    stable = len(records)==3 and now()<=stop and sum(p['passed'] for p in pairs)>=2 and mean_guard['passed']
    save_json(root / 'study/confirmed.json',{'stable_improvement':stable,'selected':selected,'records':records,'pairs':pairs,'mean_guard':mean_guard,'completed_utc':now().isoformat()})
    chosen=root / 'runs' / records[0]['run'] / 'best.pt' if stable else old4 / 'final/model.pt'
    shutil.copy2(chosen,root / 'final/model.pt'); shutil.copy2(old4 / 'final/scaler.npz',root / 'final/scaler.npz')
    save_json(root / 'study/frozen.json',{'frozen_utc':now().isoformat(),'selected_checkpoint':str(chosen),
        'model_name':selected['name'] if stable else 'round4_balanced_inverse','stable_improvement':stable,
        'checkpoint_sha256':digest(root / 'final/model.pt'),'scaler_sha256':digest(root / 'final/scaler.npz'),
        'protocol_sha256':digest(root / 'study/protocol.json'),'test_used_for_selection':False,'quantization':None})
    save_json(root / 'study/terminal.json',{'status':'completed','stable_improvement':stable,'completed_utc':now().isoformat()})
    print(json.dumps({'frozen':read(root / 'study/frozen.json'),'pairs':pairs}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','data-root','round2-root','round4-root']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.data_root.resolve(),a.round2_root.resolve(),a.round4_root.resolve())
