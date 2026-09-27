"""Export audited paired simplification results and the three-arm descriptive comparison."""
import argparse
import csv
from pathlib import Path
import shutil
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from q2.data import digest,save_json
from q2.joint_study import mean_summary
from run_joint_missing_study import read
from report_joint_missing_study import audit_data
from run_no_augmentation_study import simplicity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('run','aligned','output'):
        p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args()
    terminal=read(a.run/'terminal.json')
    assert terminal['status']=='completed'
    assert not a.output.exists()
    started=read(a.run/'started.json')
    protocol=read(a.run/'protocol.json')
    assert digest(a.run/'protocol.json')==started['protocol_sha256']
    assert digest(a.aligned)==started['aligned_sha256']
    for relative,sha in started['source_sha256'].items():
        assert digest(ROOT/relative)==sha,relative
    result={'model_root':str(a.run),'terminal':terminal,'data_audit':audit_data(a.run,a.aligned,protocol)}
    stages=['development']+(['confirmation'] if terminal['validation_evaluated'] else [])
    for stage in stages:
        analysis=read(a.run/f'{stage}_analysis.json')
        assert simplicity(analysis,protocol)==read(a.run/f'{stage}_simplicity.json')
        for row in analysis['verified_tables']:
            assert digest(a.run/row['path'])==row['sha256'],row['path']
        result[stage]=analysis
    conditions,runs={},{}
    for folder in sorted((a.run/'evaluation').iterdir()):
        conditions[folder.name]=read(folder/'metrics.json')
        run=a.run/'runs'/folder.name
        config=read(run/'config.json')
        records={p.name:read(p) for p in run.glob('*.json')}
        records['checkpoint_sha256']=digest(run/'best.pt')
        runs[folder.name]=records
        reference=read(a.run/'runs'/folder.name.replace('candidate','reference')/'config.json')
        assert {k:v for k,v in config.items() if k!='augmentation'}=={k:v for k,v in reference.items() if k!='augmentation'}
        assert config['augmentation']==('none' if 'candidate' in folder.name else 'span')
    for record in read(a.run/'reused_references.json'):
        assert runs[record['run']]['checkpoint_sha256']==record['checkpoint_sha256']==digest(Path(record['source'])/'best.pt')
    previous=Path(started['reference_root'])
    old=read(previous/'development.json')
    current=read(a.run/'development.json')
    assert [r['reference'] for r in old]==[r['reference'] for r in current]
    result['three_arm_development']={
        'span':mean_summary([r['reference'] for r in current]),
        'none':mean_summary([r['candidate'] for r in current]),
        'mixed_joint':mean_summary([r['candidate'] for r in old])}
    result['clean_class_f1']={}
    for mode,root,arm in [('span',a.run,'reference'),('none',a.run,'candidate'),('mixed_joint',previous,'candidate')]:
        scores=[read(root/'evaluation'/f'fold{fold}_{arm}_s42'/'metrics.json')['clean'] for fold in range(3)]
        result['clean_class_f1'][mode]={'negative_neutral_positive_fold_mean':np.mean([s['class_f1'] for s in scores],axis=0).tolist(),'pooled_confusion_matrix':np.sum([s['confusion_matrix'] for s in scores],axis=0).tolist()}
    result['three_arm_scope']='Descriptive same-fold comparison; selection concerns none versus span only; no independent new benchmark.'
    a.output.mkdir(parents=True)
    for path in sorted(a.run.glob('*.json')):
        if path.name.endswith('_analysis.json'):
            continue
        shutil.copyfile(path,a.output/path.name)
    rows=[]
    for scenario in conditions['fold0_reference_s42']:
        if scenario=='selection':
            continue
        row={'scenario':scenario}
        for mode,root,arm in [('span',a.run,'reference'),('none',a.run,'candidate'),('mixed_joint',previous,'candidate')]:
            scores=[read(root/'evaluation'/f'fold{fold}_{arm}_s42'/'metrics.json')[scenario] for fold in range(3)]
            for metric in ('accuracy','macro_f1','mae','pearson'):
                row[mode+'_'+metric]=float(np.mean([score[metric] for score in scores]))
        for metric in ('accuracy','macro_f1','mae','pearson'):
            row['none_minus_span_'+metric]=row['none_'+metric]-row['span_'+metric]
        rows.append(row)
    with (a.output/'comparison_by_condition.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result['condition_counts']={'total':len(rows),'none_f1_higher':sum(r['none_minus_span_macro_f1']>0 for r in rows),'none_mae_lower':sum(r['none_minus_span_mae']<0 for r in rows)}
    save_json(a.output/'analysis.json',result)
    save_json(a.output/'condition_metrics.json',conditions)
    save_json(a.output/'runs.json',runs)
    save_json(a.output/'verification.json',{'tables':sum(len(result[s]['verified_tables']) for s in stages),
        'csv_metrics_recomputed_during_analysis':True,'csv_hashes_rechecked':True,
        'data_masks_scalers_class_weights_reconstructed':True,'paired_configs_differ_only_in_augmentation':True,
        'reused_baseline_checkpoints_exact':True,'report_script_sha256':digest(Path(__file__))})
    save_json(a.output/'files.json',{p.name:digest(p) for p in sorted(a.output.iterdir()) if p.name!='files.json'})
    print({'output':str(a.output),'candidate_promoted':terminal['candidate_promoted']})

if __name__=='__main__':
    main()
