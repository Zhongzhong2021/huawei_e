"""Recompute input signal checks using only contest attachments and their inventory."""
import argparse,hashlib,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from common import read,write,digest
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path);args=parser.parse_args()
    inventory=read(ROOT/'data/inputs.json');source=args.source or Path(inventory['source_root'])
    def check(s):
        p=source/s['path'];assert digest(p)==s['sha256']
        command=['ffmpeg','-nostdin','-v','warning','-i',str(p),'-map','0:a:0','-f','f32le','-']
        run=subprocess.run(command,capture_output=True,check=True);a=np.frombuffer(run.stdout,'<f4')
        channels=s['audio']['channels'];assert len(a)//channels==s['pcm_samples']
        row={'sample_id':s['sample_id'],'source_sha256':s['sha256'],'command':command,'pcm_frames':len(a)//channels,'channels':channels,'sample_rate':int(s['audio']['sample_rate']),'rms':float(np.sqrt(np.mean(a.astype(float)**2))),'absolute_peak':float(np.max(np.abs(a))),'all_zero':bool(np.all(a==0)),'decode_warnings':run.stderr.decode(errors='replace'),'pcm_sha256':hashlib.sha256(run.stdout).hexdigest(),'container_duration_s':s['container_duration_s'],'pcm_duration_s':len(a)/channels/int(s['audio']['sample_rate']),'video_support_start_s':s['video']['start_s'],'video_support_end_s':s['video']['end_s']}
        assert row['all_zero']==s['zero_audio']
        if row['all_zero']:
            command=['ffmpeg','-nostdin','-v','warning','-ignore_editlist','1','-i',str(p),'-map','0:a:0','-f','f32le','-'];run=subprocess.run(command,capture_output=True,check=True);alt=np.frombuffer(run.stdout,'<f4')
            row['ignore_editlist_check']={'command':command,'all_zero':bool(np.all(alt==0)),'absolute_peak':float(np.max(np.abs(alt))),'samples_all_channels':len(alt),'decode_warnings':run.stderr.decode(errors='replace')}
        return row
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(check,inventory['samples']))
    write(ROOT/'data/attachment_audit/checks.json',{'schema':'q1-attachment-signal-check-v1','input_inventory_sha256':digest(ROOT/'data/inputs.json'),'generator_sha256':digest(Path(__file__)),'ffmpeg_version':subprocess.check_output(['ffmpeg','-version']).decode().splitlines()[0],'samples':len(rows),'zero_audio_samples':[r['sample_id'] for r in rows if r['all_zero']],'warning_samples':sum(bool(r['decode_warnings']) for r in rows),'external_reference_data_used':False,'human_review_used':False,'rows':rows})
    print('Recomputed signal checks from all 100 original attachments.')
if __name__=='__main__':main()
