"""Postfreeze audit and prediction-derived tables for the deferred-weighting study."""
import argparse
import json
from pathlib import Path
import shutil
import statistics
import numpy as np
from q2.data import digest, save_json
from q2v2.data import load_fold
from q2v3.analysis import arrays, write_rows
from finish_round4 import validate_prediction_file
from q2.missing import MAIN
from run_round5 import guards


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def scalar(m):
    c=m['clean']; s=m['selection']
    return {'score':s['score'],'clean_f1':c['macro_f1'],'clean_mae':c['mae'],'accuracy':c['accuracy'],
        'neutral_f1':c['class_f1'][1],'positive_f1':c['class_f1'][2],
        'missing_f1':s['mean_missing_macro_f1'],'missing_mae':s['mean_missing_mae']}


def average(records):
    values=[scalar(m) for m in records]
    return {k:statistics.mean(r[k] for r in values) for k in values[0]}


def recompute_saved(root):
    """Portable CSV verification; does not pretend to refit unavailable raw-data counts."""
    root=Path(root);dev=read(root/'study/development.json');protocol=read(root/'study/protocol.json')
    verified=[];fold_rows=[];decisions=[]
    for directory in [root/'runs',root/'evidence/reference/runs']:
        for run in sorted(directory.iterdir()):
            if not (run/'metrics.json').exists():continue
            m=read(run/'metrics.json')
            for scene in ['clean']+MAIN:
                path=run/f'predictions/{scene}.csv';validate_prediction_file(path,m[scene]);verified.append(str(path.relative_to(root)))
    references=[read(root/f'evidence/reference/runs/balanced_inverse_fold{f}/metrics.json') for f in range(3)]
    assert references==dev['reference']
    base=average(references)
    for fold,m in enumerate(references):fold_rows.append({'model':'reference','delay':0,'fold':fold,'n':m['clean']['n'],**scalar(m)})
    for entry in dev['entries']:
        actual=[read(root/f"runs/{entry['name']}_fold{f}/metrics.json") for f in range(3)]
        assert actual==entry['metrics']
        check=guards(average(actual),base)
        for k,v in check['deltas'].items():assert abs(v-entry['guard']['deltas'][k])<1e-12
        assert check['checks']==entry['guard']['checks']
        improvements=sum(a['selection']['score']>b['selection']['score'] for a,b in zip(actual,references))
        assert (improvements>=2 and check['passed'])==entry['eligible']
        decisions.append({'model':entry['name'],'fold_improvements':improvements,'eligible':entry['eligible'],
            'failed_conditions':';'.join([k for k,v in check['checks'].items() if not v]+([] if improvements>=2 else ['fold_improvements'])),
            **{'delta_'+k:v for k,v in check['deltas'].items()}})
        for fold,m in enumerate(actual):fold_rows.append({'model':entry['name'],'delay':entry['config']['class_weight_delay_epochs'],'fold':fold,'n':m['clean']['n'],**scalar(m)})
    write_rows(root/'analysis/internal_folds.csv',fold_rows)
    if decisions:write_rows(root/'analysis/decision_table.csv',decisions)
    save_json(root/'audit/portable_prediction_recomputation.json',{'verified_prediction_scenarios':len(verified),
        'selection_checks_recomputed':True,'raw_fitting_count_audit':'See original poststudy_checks.json; not repeated by this portable command', 'verified_paths':verified})
    print(json.dumps({'verified_prediction_scenarios':len(verified),'round':'5'}))


def main(root, old4, old2):
    frozen=read(root/'study/frozen.json'); protocol=read(root/'study/protocol.json')
    assert digest(root/'final/model.pt')==frozen['checkpoint_sha256']
    assert digest(root/'final/scaler.npz')==frozen['scaler_sha256']
    assert digest(root/'study/protocol.json')==frozen['protocol_sha256']
    for relative,value in read(root/'audit/source_hashes.json').items(): assert digest(root/relative)==value,relative
    original=read(root/'audit/input_hashes.json')
    assert digest(old4/'final/model.pt')==original['reference_model_sha256']
    assert digest(old4/'study/frozen.json')==original['reference_freeze_sha256']
    for key,value in original['folds'].items():
        fold,split=key.split('/'); assert digest(old2/f'folds/fold{fold}/{split}.npz')==value
    dev=read(root/'study/development.json'); confirmed=read(root/'study/confirmed.json')
    closure=read(root/'audit/interruption_closure.json') if (root/'audit/interruption_closure.json').exists() else None
    control=read(root/'audit/control_reproduction.json'); assert control['all_tensors_exact'] and control['score_delta']==0
    for fold in range(3):
        source=old4/f'runs/balanced_inverse_fold{fold}'; target=root/f'evidence/reference/runs/balanced_inverse_fold{fold}'
        target.mkdir(parents=True,exist_ok=True)
        for p in source.glob('*.json'): shutil.copy2(p,target/p.name)
        shutil.copytree(source/'predictions',target/'predictions',dirs_exist_ok=True)
        assert read(source/'metrics.json')==dev['reference'][fold]
    verified=[]; weight_audit=[]; fold_rows=[]; decision_rows=[]
    for directory in [root/'runs',root/'evidence/reference/runs']:
        for run in sorted(directory.iterdir()):
            if not (run/'metrics.json').exists():continue
            m=read(run/'metrics.json')
            for scene in ['clean']+MAIN:
                path=run/f'predictions/{scene}.csv'; validate_prediction_file(path,m[scene])
                verified.append({'path':str(path.relative_to(root)),'sha256':digest(path)})
    base=average(dev['reference']); eligible=[]
    for fold,m in enumerate(dev['reference']):fold_rows.append({'model':'reference','delay':0,'fold':fold,'n':m['clean']['n'],**scalar(m)})
    limits=protocol['guardrails']
    for entry in dev['entries']:
        assert entry['metrics']==[read(root/f"runs/{entry['name']}_fold{f}/metrics.json") for f in range(3)]
        candidate=average(entry['metrics']); diff={k:candidate[k]-base[k] for k in base}
        checks={'score_improved':diff['score']>0,'clean_f1':diff['clean_f1']>=-limits['clean_f1_max_drop'],
            'clean_mae':diff['clean_mae']<=limits['clean_mae_max_increase'], 'accuracy':diff['accuracy']>=-limits['accuracy_max_drop'],
            'neutral_f1':diff['neutral_f1']>=-limits['neutral_f1_max_drop'],'positive_f1_improved':diff['positive_f1']>0}
        improvements=sum(a['selection']['score']>b['selection']['score'] for a,b in zip(entry['metrics'],dev['reference']))
        passed=all(checks.values()) and improvements>=limits['min_fold_score_improvements']
        assert checks==entry['guard']['checks'] and improvements==entry['fold_improvements'] and passed==entry['eligible']
        assert entry['fixed_epochs']==int(statistics.median(m['run_metadata']['best_epoch'] for m in entry['metrics']))
        if passed:eligible.append((candidate['score'],entry['name']))
        decision_rows.append({'model':entry['name'],'fold_improvements':improvements,'eligible':passed,
            'failed_conditions':';'.join([k for k,v in checks.items() if not v]+([] if improvements>=2 else ['fold_improvements'])),**{'delta_'+k:v for k,v in diff.items()}})
        for fold,m in enumerate(entry['metrics']):
            run=root/f"runs/{entry['name']}_fold{fold}"; delay=entry['config']['class_weight_delay_epochs']
            fitting,validation=load_fold(old2,fold,'train'),load_fold(old2,fold,'valid')
            assert not set(str(x).split('$_$')[0] for x in fitting['ids']) & set(str(x).split('$_$')[0] for x in validation['ids'])
            counts=np.bincount(fitting['labels'],minlength=3); weights=1/counts; weights/=weights.mean()
            prov=read(run/'provenance.json'); assert prov['fit_ids']==fitting['ids'].tolist()
            np.testing.assert_allclose(prov['classification_weights'],weights,atol=1e-7,rtol=1e-6)
            for row in read(run/'epochs.json'):
                if row['epoch']<=delay:assert row['classification_weights'] is None
                else:np.testing.assert_allclose(row['classification_weights'],weights,atol=1e-7,rtol=1e-6)
            assert m['run_metadata']['best_epoch']>delay
            a=arrays(run/'predictions/clean.csv'); b=arrays(root/f'evidence/reference/runs/balanced_inverse_fold{fold}/predictions/clean.csv')
            assert np.array_equal(a['ids'],b['ids']) and np.array_equal(a['labels'],b['labels']) and np.array_equal(a['targets'],b['targets'])
            weight_audit.append({'run':str(run.relative_to(root)),'delay':delay,'fit_counts':counts.tolist(),'best_epoch':m['run_metadata']['best_epoch'],'schedule_verified':True})
            fold_rows.append({'model':entry['name'],'delay':delay,'fold':fold,'n':m['clean']['n'],**scalar(m)})
    selected=max(eligible)[1] if eligible else None
    assert selected==(confirmed['selected']['name'] if confirmed['selected'] else None)
    if selected is None:
        assert not confirmed['records'] and not confirmed['stable_improvement']
        assert digest(root/'final/model.pt')==digest(old4/'final/model.pt')
    write_rows(root/'analysis/internal_folds.csv',fold_rows)
    if decision_rows:write_rows(root/'analysis/decision_table.csv',decision_rows)
    save_json(root/'analysis/summary.json',{'baseline':base,'candidates':[{k:e[k] for k in ['name','config','summary','fold_improvements','eligible','fixed_epochs','guard']} for e in dev['entries']],
        'frozen':frozen,'confirmed':confirmed,'events':dev['events'],'decisions':decision_rows,
        'interruption':closure,'incomplete_candidates':closure['incomplete_candidates'] if closure else []})
    save_json(root/'audit/poststudy_checks.json',{'verified_prediction_scenarios':len(verified),'prediction_sources':verified,
        'fit_weight_and_epoch_audit':weight_audit,'independent_selection_verified':True,'control':control,
        'reference_unchanged':True,'official_confirmation_runs':len(confirmed['records']),
        'new_official_validation_used':bool(confirmed['records']),'test_used_for_selection':False,'sources_unchanged':True,
        'analysis_script_sha256':digest(Path(__file__)),
        'study_status':'interrupted_incomplete' if closure else 'completed',
        'incomplete_candidates_excluded_from_inference':closure['incomplete_candidates'] if closure else []})
    print(json.dumps({'verified_prediction_scenarios':len(verified),'decisions':decision_rows,'stable_improvement':frozen['stable_improvement']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','round4-root','round2-root']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();main(a.root,a.round4_root,a.round2_root)
