"""Two fixed frozen-model blends; training/test/special inputs never opened here."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import numpy as np

from q2.data import digest, save_json, CLASSES
from q2.metrics import metrics
from q2.missing import MAIN
from q2v3.analysis import arrays, write_rows
from q2v3.blending import blend_predictions
from run_round5 import summarize, guards
from finish_round4 import validate_prediction_file


def now():return datetime.now(timezone.utc)
def read(path):return json.loads(path.read_text(encoding='utf-8'))


def paired_run(current, previous, weight, destination, sources):
    result={}
    for scene in ['clean']+MAIN:
        paths=[p/f'predictions/{scene}.csv' for p in [current,previous]]
        a,b=[arrays(p) for p in paths]
        for run,p,values in zip([current,previous],paths,[a,b]):
            validate_prediction_file(p,read(run/'metrics.json')[scene])
            sources[str(p)] = digest(p)
        combined=blend_predictions(a,b,weight)
        result[scene]=metrics(combined['labels'],combined['targets'],combined['probabilities'],combined['predictions'])
        rows=[{'sample_id':str(combined['ids'][i]),'scenario':scene,'true_class':int(combined['labels'][i]),
            'true_intensity':float(combined['targets'][i]),'class_id':int(combined['classes'][i]),
            'sentiment':CLASSES[combined['classes'][i]],'intensity':float(combined['predictions'][i]),
            **{'prob_'+c:float(combined['probabilities'][i,j]) for j,c in enumerate(['negative','neutral','positive'])}}
            for i in range(len(combined['ids']))]
        write_rows(destination/f'predictions/{scene}.csv',rows)
        validate_prediction_file(destination/f'predictions/{scene}.csv',result[scene])
    f1=float(np.mean([result[s]['macro_f1'] for s in MAIN]));mae=float(np.mean([result[s]['mae'] for s in MAIN]))
    result['selection']={'mean_missing_macro_f1':f1,'mean_missing_mae':mae,'score':.5*f1+.5*(1-mae/6)}
    save_json(destination/'metrics.json',result)
    return result


def main(root, old3, old4):
    if root.exists():raise FileExistsError('A round-six study root must be new')
    if any(root==p or root in p.parents or p in root.parents for p in [old3,old4]):raise ValueError('Independent study directory required')
    root.mkdir(parents=True)
    start=now()
    protocol={'version':'round6-predeclared-probability-blend-v1','started_utc':start.isoformat(),
        'optimization_deadline':(start+timedelta(hours=1)).isoformat(),'delivery_deadline':(start+timedelta(hours=2)).isoformat(),
        'candidates':[{'name':'r4_90_r3_10','round4_weight':.9},{'name':'r4_75_r3_25','round4_weight':.75}],
        'method':'Same convex weight on class probabilities and regression intensities; no fitted mixing model, threshold, calibration or logit averaging',
        'reference':'round4 balanced_inverse','components':'already frozen round3 full-span and round4 inverse-weight models of matching fold/seed',
        'internal_protocol':'Three original video-group folds, seed42, saved predictions only; both components fitted on same internal fitting subset',
        'selection_score':'.5*mean_missing_macro_f1 + .5*(1-mean_missing_mae/6)',
        'guardrails':{'min_fold_score_improvements':2,'clean_f1_max_drop':.01,'clean_mae_max_increase':.03,
                     'accuracy_max_drop':.005,'neutral_f1_max_drop':.01,'positive_f1':'strictly improve'},
        'confirmation':'Only best eligible mean-S candidate; reuse matching seed42/2026/3407 frozen component predictions; >=2/3 paired guard passes AND mean guards pass',
        'promotion':'Requires paired confirmation, independently loaded two-model inference matching saved mixtures, and full attachment3 checks; statistical confirmation alone cannot change deployment',
        'limits':'Exactly two fixed blends; no new training or parameter search; no test or attachment3 access before selection is frozen',
        'selection_bias':'Both components and official validation have been used in prior rounds; this is adaptive development, not independent generalization evidence'}
    save_json(root/'study/protocol.json',protocol)
    save_json(root/'study/started.json',{'started_utc':start.isoformat(),'protocol_sha256':digest(root/'study/protocol.json')})
    save_json(root/'audit/implementation.json',{'runner_sha256':digest(Path(__file__)),
        'blending_sha256':digest(Path(__file__).parents[1]/'src/q2v3/blending.py')})
    # Check preprocessing identity before computing any new mixture metrics.
    assert digest(old3/'final/scaler.npz')==digest(old4/'final/scaler.npz')
    refs=[read(old4/f'runs/balanced_inverse_fold{f}/metrics.json') for f in range(3)]
    sources={};entries=[];baseline=summarize(refs)
    for candidate in protocol['candidates']:
        if now()>datetime.fromisoformat(protocol['optimization_deadline']):raise TimeoutError('Budget expired')
        records=[]
        for f in range(3):
            current=old4/f'runs/balanced_inverse_fold{f}';previous=old3/f'runs/bert12_full_span_fold{f}'
            cp,pp=read(current/'provenance.json'),read(previous/'provenance.json')
            assert cp['fit_ids']==pp['fit_ids'] and cp['validation_ids']==pp['validation_ids']
            records.append(paired_run(current,previous,candidate['round4_weight'],root/f"runs/{candidate['name']}_fold{f}",sources))
        summary=summarize(records);guard=guards(summary,baseline)
        improvements=sum(m['selection']['score']>r['selection']['score'] for m,r in zip(records,refs))
        entries.append({**candidate,'metrics':records,'summary':summary,'guard':guard,
                        'fold_improvements':improvements,'eligible':improvements>=2 and guard['passed']})
        save_json(root/'study/development.json',{'reference':refs,'reference_summary':baseline,'entries':entries})
        print(json.dumps({k:entries[-1][k] for k in ['name','summary','guard','fold_improvements','eligible']}),flush=True)
    eligible=[e for e in entries if e['eligible']]
    selected=max(eligible,key=lambda e:e['summary']['score']) if eligible else None
    save_json(root/'study/internal_selection.json',{'selected':selected,'selected_utc':now().isoformat(),'test_used':False})
    records=[];pairs=[]
    if selected:
        for seed in [42,2026,3407]:
            if now()>datetime.fromisoformat(protocol['optimization_deadline']):raise TimeoutError('Budget expired')
            current=old4/f'runs/confirmed_balanced_inverse_s{seed}';previous=old3/f'runs/confirmed_bert12_full_span_s{seed}'
            m=paired_run(current,previous,selected['round4_weight'],root/f"runs/confirmed_{selected['name']}_s{seed}",sources)
            ref=read(current/'metrics.json')
            records.append({'seed':seed,'metrics':m,'reference':ref})
            pairs.append({'seed':seed,**guards(summarize([m]),summarize([ref]))})
            save_json(root/'study/confirmation_progress.json',{'records':records,'pairs':pairs})
    mean=guards(summarize([r['metrics'] for r in records]),summarize([r['reference'] for r in records])) if records else None
    success=len(records)==3 and sum(p['passed'] for p in pairs)>=2 and mean['passed']
    save_json(root/'study/confirmed.json',{'selected':selected,'records':records,'pairs':pairs,'mean_guard':mean,
        'statistical_conditions_passed':success,'deployment_changed':False,'next_step':'Validate actual two-model inference before promotion' if success else 'Retain round4'})
    save_json(root/'audit/prediction_sources.json',sources)
    save_json(root/'study/terminal.json',{'status':'awaiting_deployment_verification' if success else 'completed_no_promotion',
        'completed_utc':now().isoformat(),'test_used':False,'attachment3_used':False,'new_training_runs':0,'deployment_changed':False})
    print(json.dumps({'statistical_conditions_passed':success,'pairs':pairs,'mean_guard':mean}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','round3-root','round4-root']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.round3_root.resolve(),a.round4_root.resolve())
