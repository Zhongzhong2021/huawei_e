"""Fixed-recipe augmentation removal; audited baseline reuse and conditional confirmation."""
import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from q2.data import convert,digest,load_pickle,save_json
from q2.engine import evaluate_model
from q2.joint_study import decision,summarize
from q2v2.engine import load_model,train_run
from run_joint_missing_study import prepare_pair,read
from report_joint_missing_study import analyze_stage,audit_data


def simplicity(analysis,protocol):
    selection=analysis['selection']
    checks={k:v for k,v in selection['checks'].items() if k not in ('mean_score','paired_successes')}
    spec=protocol['simplification']
    checks['mean_score_noninferior']=bool(np.mean(selection['paired_score_differences'])>=-spec['low_medium_score_margin'])
    checks['paired_noninferior']=sum(d>=-spec['low_medium_score_margin'] for d in selection['paired_score_differences'])>=spec['minimum_noninferior_folds']
    for row in analysis['paired_intervals']:
        prefix=row['group'] if row['group'] in ('clean','single') else 'joint'
        metric=row['metric']
        margin=spec[prefix+('_f1_margin' if metric=='macro_f1' else '_mae_margin')]
        checks[row['group']+'_'+metric+'_interval']=bool(row['lower']>-margin if metric=='macro_f1' else row['upper']<margin)
    return {'passed':all(checks.values()),'checks':checks,'scope':spec['interval_rule']}


def fit_arm(root,name,seed,arm,data,source,config,training,protocol):
    fitting,validation,masks=data
    run_name=f'{name}_{arm}_s{seed}'
    cfg={**config,'seed':seed,'augmentation':'span' if arm=='reference' else 'none','separate_augmentation_rng':True}
    train_run(root,run_name,cfg,fitting,validation,np.arange(30522),source=source,fixed_epochs=4,reproduction=True,training_protocol=training)
    gc.collect()
    torch.cuda.empty_cache()
    model,_=load_model(root/'runs'/run_name/'best.pt','cuda')
    out=root/'evaluation'/run_name
    scores=evaluate_model(model,validation,'cuda',list(masks),out=out/'predictions',batch_size=32,masks=masks)
    save_json(out/'metrics.json',scores)
    summary=summarize(scores,protocol)
    save_json(out/'summary.json',summary)
    print({'run':run_name,'summary':summary},flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def finish_stage(root,protocol,stage,records):
    save_json(root/f'{stage}.json',records)
    result=decision([r['reference'] for r in records],[r['candidate'] for r in records],protocol)
    save_json(root/f'{stage}_decision.json',result)
    analysis=analyze_stage(root,protocol,stage)
    save_json(root/f'{stage}_analysis.json',analysis)
    gate=simplicity(analysis,protocol)
    save_json(root/f'{stage}_simplicity.json',gate)
    print({'stage':stage,'simplicity':gate,'comparison':result},flush=True)
    return gate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('reference','aligned','pretrained','output'):
        p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    old=read(a.reference/'started.json')
    for relative,sha in old['source_sha256'].items():
        assert digest(ROOT/relative)==sha,relative
    assert digest(a.aligned)==old['aligned_sha256']
    assert digest(a.pretrained)==old['initialization']['converted_sha256']
    assert read(a.reference/'terminal.json')['status']=='completed'
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    protocol_path=ROOT/'configs/no_augmentation_study.json'
    protocol=read(protocol_path)
    config,training=old['base_config'],old['training_protocol']
    audits=audit_data(a.reference,a.aligned,protocol)
    a.output.mkdir(parents=True)
    shutil.copyfile(protocol_path,a.output/'protocol.json')
    started=time.time()
    save_json(a.output/'started.json',{
        'started_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':digest(protocol_path),
        'aligned_sha256':digest(a.aligned),'initialization':old['initialization'],
        'base_config':config,'training_protocol':training,'reference_root':str(a.reference),
        'reference_data_audit':audits,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
        'source_sha256':{str(path.relative_to(ROOT)):digest(path) for folder in ('src','scripts','tests') for path in sorted((ROOT/folder).rglob('*.py'))}})
    source=torch.load(a.pretrained,map_location='cpu',weights_only=True)
    records,reused=[],[]
    for fold in range(3):
        name=f'fold{fold}'
        folder=a.output/'data'/name
        shutil.copytree(a.reference/'data'/name,folder)
        run=f'{name}_reference_s42'
        shutil.copytree(a.reference/'evaluation'/run,a.output/'evaluation'/run)
        target=a.output/'runs'/run
        target.mkdir(parents=True)
        for path in (a.reference/'runs'/run).iterdir():
            if path.name=='best.pt':
                (target/path.name).hardlink_to(path)
            elif path.is_file():
                shutil.copyfile(path,target/path.name)
        reused.append({'run':run,'checkpoint_sha256':digest(target/'best.pt'),'source':str(a.reference/'runs'/run)})
        data=[]
        for split in ('train','valid'):
            with np.load(folder/f'{split}.npz',allow_pickle=False) as stored:
                data.append({k:stored[k] for k in stored.files})
        with np.load(folder/'masks.npz',allow_pickle=False) as stored:
            data.append({k:stored[k] for k in stored.files if k!='ids'})
        candidate=fit_arm(a.output,name,42,'candidate',data,source,config,training,protocol)
        records.append({'fold':fold,'video_overlap':0,'reference':read(a.output/'evaluation'/run/'summary.json'),'candidate':candidate})
        save_json(a.output/'development.json',records)
    save_json(a.output/'reused_references.json',reused)
    dev=finish_stage(a.output,protocol,'development',records)
    terminal={'development_simplicity':dev,'validation_evaluated':False,'candidate_promoted':False,
              'test_evaluated':False,'special_labels_used':False,'new_fits':3,'reused_fits':3}
    if dev['passed']:
        raw=load_pickle(a.aligned)
        train,_=convert(raw['train'],raw['train']['id'])
        valid,_=convert(raw['valid'],raw['valid']['id'])
        data=prepare_pair(a.output,'full',train,valid,protocol)
        confirmation=[]
        for seed in protocol['confirmation_seeds']:
            scores={arm:fit_arm(a.output,'full',seed,arm,data,source,config,training,protocol) for arm in ('reference','candidate')}
            confirmation.append({'seed':seed,**scores})
            save_json(a.output/'confirmation.json',confirmation)
        confirmed=finish_stage(a.output,protocol,'confirmation',confirmation)
        terminal.update(confirmation_simplicity=confirmed,validation_evaluated=True,candidate_promoted=confirmed['passed'],new_fits=9)
    terminal.update(status='completed',seconds=time.time()-started,completed_utc=datetime.now(timezone.utc).isoformat())
    save_json(a.output/'terminal.json',terminal)
    print(terminal,flush=True)

if __name__=='__main__':
    main()
