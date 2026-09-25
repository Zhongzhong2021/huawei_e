"""Close an expired, terminal interrupted study without rewriting its evidence.

Never extends the original time budget or declares incomplete candidates failed
scientifically. Only adds closure/fallback records after checking no live runner.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import zipfile

import torch
from q2.data import digest, save_json


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def main(root, old4):
    if not Path('/proc').exists():
        raise RuntimeError('Run inside the original WSL environment for live-process checks')
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        if int(path.parent.name) == os.getpid():continue
        try: args = path.read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError):continue
        if any(x.endswith(b'/run_round5.py') for x in args) and str(root).encode() in args:
            raise RuntimeError(f'Original study runner is still live: PID {path.parent.name}')
    for name in ['frozen.json','terminal.json','confirmed.json','internal_selection.json']:
        if (root/'study'/name).exists():raise FileExistsError(f'Already has {name}; inspect before closure')
    if any((root/'final').iterdir()):raise FileExistsError('Final directory is not empty')
    protocol=read(root/'study/protocol.json')
    timestamp=datetime.now(timezone.utc)
    if timestamp <= datetime.fromisoformat(protocol['optimization_deadline']):
        raise ValueError('Original optimization budget has not expired')
    before = {str(p.relative_to(root)):digest(p) for p in (root/'study').glob('*.json')}
    for rel,h in read(root/'audit/source_hashes.json').items():
        if digest(root/rel)!=h:raise ValueError(f'Original source changed: {rel}')
    original=read(root/'audit/input_hashes.json')
    if digest(old4/'final/model.pt')!=original['reference_model_sha256']:raise ValueError('Reference changed')
    development=read(root/'study/development.json')
    if any(e['eligible'] for e in development['entries']):
        raise ValueError('Eligible candidate requires explicit confirmation-status review')
    missing=[]
    for candidate in protocol['candidates']:
        if any(e['name']==candidate['name'] for e in development['entries']):continue
        checkpoints=[]
        completed=0
        for f in range(protocol['folds']):
            run=root/f"runs/{candidate['name']}_fold{f}"
            completed += (run/'metrics.json').exists()
            if (run/'best.pt').exists():
                path=run/'best.pt';error=None
                try:torch.load(path,map_location='cpu',weights_only=True)
                except Exception as exc:error=repr(exc)
                checkpoints.append({'path':str(path.relative_to(root)), 'bytes':path.stat().st_size,
                    'sha256':digest(path),'mtime_utc':datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),
                    'zip_complete':zipfile.is_zipfile(path),'load_error':error})
        missing.append({'name':candidate['name'],'status':'interrupted_incomplete','completed_folds':completed,'checkpoints':checkpoints})
    closure={'closed_utc':timestamp.isoformat(),'status':'interrupted_closed_without_budget_extension',
        'process_check':'No live run_round5.py process with this root in /proc',
        'current_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'cause':'Checkpoint archive incomplete; original stall and process exit causes unproven. WSL has rebooted since the run.',
        'completed_candidates':[e['name'] for e in development['entries']], 'incomplete_candidates':missing,
        'original_study_hashes':before,'original_protocol_unchanged':True,
        'new_official_validation_used':False,'new_test_used':False,'optimization_success':False}
    save_json(root/'audit/interruption_closure.json',closure)
    save_json(root/'study/internal_selection.json',{'selected':None,'selected_utc':timestamp.isoformat(),'test_accessed':False,
        'reason':'Completed candidate ineligible; other candidate incomplete at original deadline'})
    save_json(root/'study/confirmed.json',{'stable_improvement':False,'selected':None,'records':[],'pairs':[],
        'mean_guard':None,'completed_utc':None,'closed_utc':timestamp.isoformat(),'status':'not_run_after_interruption'})
    shutil.copy2(old4/'final/model.pt',root/'final/model.pt')
    shutil.copy2(old4/'final/scaler.npz',root/'final/scaler.npz')
    save_json(root/'study/frozen.json',{'frozen_utc':timestamp.isoformat(),
        'selected_checkpoint':str(old4/'final/model.pt'),'model_name':'round4_balanced_inverse','stable_improvement':False,
        'checkpoint_sha256':digest(root/'final/model.pt'),'scaler_sha256':digest(root/'final/scaler.npz'),
        'protocol_sha256':digest(root/'study/protocol.json'),'test_used_for_selection':False,'quantization':None,
        'status':'retained_prior_model_after_incomplete_study','not_a_newly_confirmed_round5_model':True})
    for rel,h in before.items():
        if digest(root/rel)!=h:raise RuntimeError('Original evidence was unexpectedly changed')
    save_json(root/'study/terminal.json',closure)
    print(json.dumps(closure,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','round4-root']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.round4_root.resolve())
