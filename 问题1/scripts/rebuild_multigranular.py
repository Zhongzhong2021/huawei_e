"""Rebuild the enhanced policy over a verified core in an isolated new directory."""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from common import write,digest

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--core',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--whisper-python',type=Path,required=True)
    p.add_argument('--whisper-model',type=Path,required=True)
    p.add_argument('--mfa-prefix',type=Path,required=True)
    p.add_argument('--mfa-models',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(root/'scripts',output/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(root/'config',output/'config')
    # Copy the actual verified core outputs; keep the selected core's provenance.
    for name in ['features','automatic_alignment','local_evidence_corroborated','temporal_index']:
        shutil.copytree(args.core/'data'/name,output/'data'/name)
    for name in ['inputs.json']:
        shutil.copy2(args.core/'data'/name,output/'data'/name)
    (output/'data/acoustic_comparison').mkdir()
    shutil.copy2(root/'data/acoustic_comparison/models.json',output/'data/acoustic_comparison/models.json')
    commands=[
        [str(args.whisper_python),'scripts/run_whisper_recognition.py','--models',str(args.whisper_model),'--source',str(args.source),'--output',str(output/'data/whisper_recognition'),'--device',args.device],
        [sys.executable,'scripts/prepare_whisper_local.py','--recognition',str(output/'data/whisper_recognition'),'--source',str(args.source),'--output',str(output/'data/whisper_local'),'--work',str(output/'work')],
        [sys.executable,'scripts/run_whisper_local_mfa.py','--output',str(output/'data/whisper_local'),'--work',str(output/'work'),'--mfa-prefix',str(args.mfa_prefix),'--models',str(args.mfa_models)],
        [sys.executable,'scripts/build_multigranular_alignment.py','--local',str(output/'data/whisper_local'),'--output',str(output/'data/multigranular_alignment')],
        [sys.executable,'scripts/evaluate_multigranular_alignment.py','--recognition',str(output/'data/whisper_recognition'),'--output',str(output/'data/multigranular_alignment/evaluation.json')],
    ]
    env=dict(os.environ,PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1')
    # venv Python may be a symlink to system Python; derive package paths from
    # the supplied environment path rather than resolving the executable link.
    libdirs=sorted(args.whisper_python.absolute().parent.parent.glob('lib/python*/site-packages/nvidia/*/lib'))
    if libdirs:env['LD_LIBRARY_PATH']=':'.join(map(str,libdirs))+':'+env.get('LD_LIBRARY_PATH','')
    record={'status':'running','core':str(args.core),'source':str(args.source),'human_data_used':False,
            'core_input_manifest_sha256':digest(args.core/'data/inputs.json'),
            'scripts':{p.name:digest(p) for p in (output/'scripts').glob('*.py')},'steps':[]}
    for index,command in enumerate(commands):
        log=output/f'step-{index+1}.log';start=time.monotonic()
        with log.open('w') as stream:result=subprocess.run(command,cwd=output,env=env,stdout=stream,stderr=subprocess.STDOUT)
        record['steps'].append({'command':command,'exit_code':result.returncode,'elapsed_s':time.monotonic()-start,'log':log.name,'log_sha256':digest(log)})
        if result.returncode:record['status']='failed'
        write(output/'data/multigranular_rebuild.json',record)
        if result.returncode:raise RuntimeError('Enhanced rebuild failed; see '+str(log))
        print(index+1,'completed',flush=True)
    record['status']='complete';write(output/'data/multigranular_rebuild.json',record)

if __name__=='__main__':main()
