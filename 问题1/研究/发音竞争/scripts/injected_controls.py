"""New fixed controls that bypass proposal filtering to test acoustic verification."""
from pathlib import Path
import os
import sys,json,hashlib,copy,argparse
import numpy as np
import torch
sys.path.insert(0,str(Path(os.environ.get('Q1_PROJECT',Path(__file__).resolve().parents[3]))/'scripts'))
from common import read,write,digest
from evaluate_competition import score
from phone_evidence import Pronunciations
R=Path(__file__).resolve().parents[1];Q=Path(os.environ.get('Q1_PROJECT',Path(__file__).resolve().parents[3]))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['canonical','contextual','lexicon'],default='contextual');args=parser.parse_args()
    torch.set_num_threads(2);protocol=read(R/'protocol.json');sources={s['sample_id']:s for s in read(Q/'data/inputs.json')['samples']}
    posteriors={}
    for sid in sources:
        with np.load(Path(os.environ.get('Q1_PHONE_POSTERIORS','/workspace/q1-phone-posteriors'))/(sid+'.npz')) as a:posteriors[sid]=(a['logp'].copy(),a['frame_cells'].copy())
    pron=Pronunciations(read((Path(os.environ.get('Q1_PHONE_MODEL','/workspace/q1-phone-model'))/'vocab.json')),mode=args.mode,dictionary=os.environ.get('Q1_PHONE_DICTIONARY','/workspace/q1-mfa-models/dictionary-english_mfa.dict') if args.mode=='lexicon' else None)
    rows=[]
    for p in read(R/'data/proposals.json'):
        source=sources[p['sample_id']];length=p['interval_s'][1]-p['interval_s'][0]
        candidates=[s for s in sources.values() if not s['zero_audio'] and s['video_id']!=source['video_id'] and s['audio']['end_s']-s['audio']['start_s']>=length]
        if not candidates:continue
        key=p['sample_id']+'|'+','.join(map(str,p['word_indices']))
        destination=min(candidates,key=lambda s:hashlib.sha256((key+'|'+s['sample_id']).encode()).digest())
        fraction=int.from_bytes(hashlib.sha256((key+'|window').encode()).digest()[:4],'big')/(2**32-1)
        start=destination['audio']['start_s']+fraction*(destination['audio']['end_s']-destination['audio']['start_s']-length)
        proposal=copy.deepcopy(p);proposal['interval_s']=[start,start+length]
        result=score(proposal,destination['sample_id'],posteriors,sources,pron,protocol)
        rows.append({'reference_sample_id':source['sample_id'],'audio_sample_id':destination['sample_id'],'word_indices':p['word_indices'],'interval_s':proposal['interval_s'],**result})
    report={'status':'complete','protocol_sha256':digest(R/'protocol.json'),'mode':args.mode,'proposals':len(rows),'accepted':sum(r['accepted'] for r in rows),'rows':rows,'scope':'Different-video candidate injection, bypassing lexical screen. Constructed acoustic stress control, not natural false acceptance probability.','script_sha256':digest(__file__)}
    write(R/'data'/args.mode/'injected.json',report);print(args.mode,len(rows),report['accepted'])
if __name__=='__main__':main()
