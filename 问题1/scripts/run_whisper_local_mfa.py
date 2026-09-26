"""Run the fixed MFA phonetic backend on ASR-proposed reference crops."""
import argparse
import os
import subprocess
import time
from pathlib import Path
from common import read, write, digest

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--mfa-prefix', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True)
    args = p.parse_args()
    preparation = read(args.output/'preparation.json')
    for j in preparation['jobs']:
        for kind,suffix in [('wav','.wav'),('text','.lab')]:
            assert digest(args.work/'corpus'/(j['file_id']+suffix)) == j[kind+'_sha256']
    models = read(ROOT/'data/acoustic_comparison/models.json')
    paths = {}
    for row in models:
        path = args.models/Path(row['path']).name
        assert digest(path) == row['sha256']
        paths[row['kind']] = str(path)
    out = args.output/'raw'; assert not out.exists()
    cfg = args.output/'mfa.yaml'
    cfg.write_text('uses_speaker_adaptation: false\nbeam: 10\nretry_beam: 40\ndither: 0.0\n')
    env = dict(os.environ, MFA_ROOT_DIR=str(args.work/'cache'), OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    env['PATH'] = str(args.mfa_prefix/'bin')+':'+env['PATH']
    mfa = str(args.mfa_prefix/'bin/mfa')
    version = subprocess.check_output([mfa,'version'],env=env,text=True).strip()
    assert version == '3.3.9'
    command = [mfa,'align',str(args.work/'corpus'),paths['dictionary'],paths['acoustic'],str(out),
               '-c',str(cfg),'--g2p_model_path',paths['g2p'],'--output_format','json','-s','6',
               '--no_use_postgres','-j','2','-t',str(args.work/'temp')]
    start = time.monotonic()
    with (args.output/'mfa.log').open('w') as stream:
        result = subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
    write(args.output/'execution.json', {'command':command,'version':version,'models':models,
          'returncode':result.returncode,'elapsed_s':time.monotonic()-start,
          'preparation_sha256':digest(args.output/'preparation.json'),'config_sha256':digest(cfg),
          'script_sha256':digest(__file__),'log_sha256':digest(args.output/'mfa.log'),'human_data_used':False})
    assert result.returncode == 0, 'See mfa.log'
    print({'returncode':result.returncode,'elapsed_s':time.monotonic()-start})

if __name__ == '__main__':
    main()
