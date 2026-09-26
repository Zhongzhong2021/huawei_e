"""Generate all proposals from original word identities and unprompted recognition."""
import os,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
Q=Path(os.environ.get('Q1_PROJECT',R.parents[1]))
sys.path.insert(0,str(Q/'scripts'))
from common import read,write,digest
from proposals import propose

def main():
    cfg=read(Q/'config/automatic_alignment.json');protocol=read(R/'protocol.json');rows=[];inputs=read(Q/'data/inputs.json')
    manifest=read(Q/'data/whisper_recognition/manifest.json');hashes={s['sample_id']:s['sha256'] for s in manifest['samples']}
    for s in inputs['samples']:
        sid=s['sample_id'];path=Q/'data/whisper_recognition'/(sid+'.json');assert digest(path)==hashes[sid]
        meta=read(Q/'data/features'/(sid+'.json'));asr=read(path);assert asr['source_sha256']==s['sha256']
        for p in propose(meta['words'],asr['words'],cfg,[s['audio']['start_s'],s['audio']['end_s']],protocol['maximum_interior_words']):rows.append({'sample_id':sid,**p})
    write(R/'data/proposals.json',rows)
    write(R/'data/proposal_manifest.json',{'source_inputs_sha256':digest(Q/'data/inputs.json'),'asr_manifest_sha256':digest(Q/'data/whisper_recognition/manifest.json'),'config_sha256':digest(Q/'config/automatic_alignment.json'),'proposals_sha256':digest(R/'data/proposals.json'),'samples':len(inputs['samples']),'proposals':len(rows),'human_data_used':False,'stable_acceptance_results_read':False})
    print('Generated',len(rows),'proposals across',len({r['sample_id'] for r in rows}),'samples')
if __name__=='__main__':main()
