"""Prediction-derived metrics, video-cluster paired bootstrap, and cases."""
import csv
import json
from pathlib import Path
import numpy as np
from q2.data import save_json, digest
from q2.metrics import metrics
from q2.missing import MAIN, SUPPLEMENT, scenario_mask

SCENARIOS=['clean']+MAIN+SUPPLEMENT
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_rows(path, records):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def arrays(path):
    r=rows(path)
    return dict(ids=np.array([x['sample_id'] for x in r]),labels=np.array([int(x['true_class']) for x in r]),
        targets=np.array([float(x['true_intensity']) for x in r]),classes=np.array([int(x['class_id']) for x in r]),
        predictions=np.array([float(x['intensity']) for x in r]),
        probabilities=np.array([[float(x['prob_'+c]) for c in ['negative','neutral','positive']] for x in r]))
def cluster_weights(ids,replicates=2000,seed=240924):
    groups=np.array([str(i).split('$_$')[0] for i in ids])
    names,inverse=np.unique(groups,return_inverse=True)
    rng=np.random.default_rng(seed)
    counts=rng.multinomial(len(names),np.full(len(names),1/len(names)),size=replicates)
    return counts[:,inverse],names
def weighted_f1(labels,classes,weights):
    cells=np.stack([(labels==a)&(classes==b) for a in range(3) for b in range(3)],axis=1).astype(float)
    cm=(weights@cells).reshape(-1,3,3)
    tp=np.diagonal(cm,axis1=1,axis2=2)
    den=cm.sum(1)+cm.sum(2)
    return np.divide(2*tp,den,out=np.zeros_like(tp),where=den!=0).mean(1)
def boot_metrics(a,weights):
    return {'macro_f1':weighted_f1(a['labels'],a['classes'],weights),
            'mae':(weights@abs(a['targets']-a['predictions']))/weights.sum(1)}
def main(root):
    root=Path(root);out=root/'analysis';out.mkdir(exist_ok=True)
    registry=[];verified=[];loaded={}
    for role in ['reference','final']:
        for split in ['valid','test']:
            saved=read(root/f'evaluation/{role}/{split}/metrics.json')
            for s in SCENARIOS:
                path=root/f'evaluation/{role}/{split}/predictions/{s}.csv'
                a=arrays(path);loaded[(role,split,s)]=a
                assert len(a['ids'])==len(set(a['ids']))
                assert np.array_equal(a['classes'],a['probabilities'].argmax(1))
                m=metrics(a['labels'],a['targets'],a['probabilities'],a['predictions'])
                for key in ['accuracy','macro_f1','mae','pearson']:
                    if m[key] is None:assert saved[s][key] is None
                    else:assert abs(m[key]-saved[s][key])<1e-10,(role,split,s,key)
                for key in ['accuracy','macro_f1','mae','pearson']:
                    registry.append(dict(model=role,split='test_descriptive' if split=='test' else split,seed=42,scenario=s,
                                         metric=key,value=m[key],n=m['n'],source=str(path.relative_to(root)),sha256=digest(path)))
                verified.append(dict(model=role,split=split,scenario=s,passed=True))
    write_rows(out/'metrics_long.csv',registry)
    save_json(root/'audit/metric_recomputation.json',{'verified_count':len(verified),'checks':verified,'tolerance':1e-10})
    # A single shared set of cluster draws preserves pairing across models/scenarios.
    anchor=loaded[('final','valid','clean')]
    weights,groups=cluster_weights(anchor['ids'])
    draws={};intervals=[]
    for role in ['reference','final']:
        for s in ['clean']+MAIN:
            a=loaded[(role,'valid',s)]
            assert np.array_equal(a['ids'],anchor['ids'])
            assert np.array_equal(a['labels'],anchor['labels']) and np.array_equal(a['targets'],anchor['targets'])
            draws[(role,s)]=boot_metrics(a,weights)
        draws[(role,'missing_average')]={metric:np.mean([draws[(role,s)][metric] for s in MAIN],axis=0) for metric in ['macro_f1','mae']}
    for s in ['clean','missing_average']+MAIN:
        for metric in ['macro_f1','mae']:
            d=draws[('final',s)][metric]-draws[('reference',s)][metric]
            if s=='missing_average':
                point=np.mean([next(r['value'] for r in registry if r['model']=='final' and r['split']=='valid' and r['scenario']==q and r['metric']==metric)-next(r['value'] for r in registry if r['model']=='reference' and r['split']=='valid' and r['scenario']==q and r['metric']==metric) for q in MAIN])
            else:
                point=next(r['value'] for r in registry if r['model']=='final' and r['split']=='valid' and r['scenario']==s and r['metric']==metric)-next(r['value'] for r in registry if r['model']=='reference' and r['split']=='valid' and r['scenario']==s and r['metric']==metric)
            intervals.append(dict(scenario=s,metric=metric,difference=float(point),lower=float(np.quantile(d,.025)),upper=float(np.quantile(d,.975))))
    write_rows(out/'paired_cluster_intervals.csv',intervals)
    np.savez_compressed(out/'bootstrap_draws.npz',**{f'{r}__{s}__{m}':v for (r,s),ms in draws.items() for m,v in ms.items()})
    save_json(out/'bootstrap_protocol.json',{'replicates':2000,'seed':240924,'clusters':len(groups),'samples':len(anchor['ids']),
              'unit':'video ID prefix before $_$','paired_across_models_and_scenarios':True,'interval':'percentile .025/.975',
              'interpretation':'Conditional on fixed seed42 fitted predictions. Not seed variability or selection-corrected inference.'})
    cases=[];diagnostics={}
    for role in ['reference','final']:
        a=loaded[(role,'valid','clean')];error=abs(a['targets']-a['predictions'])
        signclass=np.where(a['predictions']<0,0,np.where(a['predictions']>0,2,1))
        diagnostics[role]={'inconsistent_count':int((signclass!=a['classes']).sum()),'n':len(error),
            'inconsistent_rate':float((signclass!=a['classes']).mean()),'regression_zero_definition':'exact zero; no posthoc threshold',
            'mean_error_by_true_class':[float(error[a['labels']==c].mean()) for c in range(3)]}
        if role!='final':continue
        for c in range(3):
            for kind in ['wrong_largest_mae','correct_median_mae']:
                ids=np.flatnonzero((a['labels']==c)&((a['classes']!=c) if kind.startswith('wrong') else (a['classes']==c)))
                ids=sorted(ids,key=lambda i:(error[i],a['ids'][i]))
                if not ids:continue
                i=ids[-1] if kind.startswith('wrong') else ids[(len(ids)-1)//2]
                cases.append(dict(sample_id=a['ids'][i],selection_rule=kind,true_class=c,predicted_class=int(a['classes'][i]),
                     true_intensity=float(a['targets'][i]),predicted_intensity=float(a['predictions'][i]),absolute_error=float(error[i])))
    write_rows(out/'case_selection.csv',cases);save_json(out/'diagnostics.json',diagnostics)
    metadata=np.load(root/'evidence/valid_metadata.npz',allow_pickle=False)
    data={k:metadata[k] for k in metadata.files};lengths=data['valid'].sum(1)
    rates=[]
    for s in MAIN+SUPPLEMENT:
        d=scenario_mask(data,s);selected=d.sum((1,2));j=['text','audio','vision'].index(s.split('_')[0])
        actual=np.divide(selected,lengths,out=np.zeros(len(lengths),float),where=lengths>0)
        rates.append(dict(scenario=s,requested_percent=int(s.split('_')[1]),selected_positions=int(selected.sum()),
                          newly_missing_positions=int((d&data['observed']).sum()),actual_ratio_mean=float(actual.mean()),
                          actual_ratio_min=float(actual.min()),actual_ratio_max=float(actual.max()),empty_sequences=int((lengths==0).sum())))
    write_rows(out/'actual_missing_rates.csv',rates)
    confirmation=read(root/'study/confirmed.json');seeds=[]
    for r in confirmation['records']:
        for role in ['reference','metrics']:
            m=r[role]
            for s in ['clean']+MAIN:
                for metric in ['accuracy','macro_f1','mae','pearson']:
                    seeds.append(dict(model='candidate' if role=='metrics' else 'reference',seed=r['seed'],scenario=s,metric=metric,value=m[s][metric]))
    if seeds:write_rows(out/'seed_metrics.csv',seeds)
    summary={'frozen':read(root/'study/frozen.json'),'development':read(root/'study/development.json'),
             'confirmation':confirmation,'intervals':intervals,'diagnostics':diagnostics,'cases':cases,
             'valid':{r:read(root/f'evaluation/{r}/valid/metrics.json') for r in ['reference','final']},
             'test_descriptive':{r:read(root/f'evaluation/{r}/test/metrics.json') for r in ['reference','final']}}
    save_json(out/'summary.json',summary)
    sources={str(p.relative_to(root)):digest(p) for p in sorted(root.glob('evaluation/*/*/predictions/*.csv'))}
    save_json(out/'source_manifest.json',sources)
    print(json.dumps({'verified':len(verified),'bootstrap_clusters':len(groups),'intervals':intervals[:4]},ensure_ascii=False,indent=2))
