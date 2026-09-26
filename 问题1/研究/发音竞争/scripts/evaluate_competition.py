"""Fixed full-corpus proposals and controls, scored without human review inputs."""
from pathlib import Path
import os
import sys,json,hashlib,collections,copy,time,argparse
import numpy as np
import torch
sys.path.insert(0,str(Path(os.environ.get('Q1_PROJECT',Path(__file__).resolve().parents[3]))/'scripts'))
from common import read,write,digest
from math_features import words as tokenize
from proposals import propose
from phone_evidence import Pronunciations,compete
R=Path(__file__).resolve().parents[1];Q=Path(os.environ.get('Q1_PROJECT',Path(__file__).resolve().parents[3]))

def score(proposal,audio_id,posteriors,sources,pron,protocol):
    lp,clock=posteriors[audio_id];source=sources[audio_id]
    evidence=[]
    alternatives=[w['text'] for w in tokenize(proposal['recognized_text'])]
    for context in protocol['contexts_s']:
        lo=max(source['audio']['start_s'],proposal['interval_s'][0]-context)
        hi=min(source['audio']['end_s'],proposal['interval_s'][1]+context)
        center=clock.mean(1);frames=np.flatnonzero((center>=lo)&(center<hi))
        if not len(frames):evidence.append({'accepted':False,'status':'empty_audio_window'});continue
        result=compete(lp[frames],proposal['reference_words'],alternatives,pron,protocol['minimum_log_likelihood_margin'])
        evidence.append({'context_s':context,'frame_span':[int(frames[0]),int(frames[-1]+1)],'interval_s':[float(clock[frames[0],0]),float(clock[frames[-1],1])],**result})
    return {'accepted':all(r['accepted'] for r in evidence),'evidence':evidence}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['canonical','contextual','lexicon'],default='contextual');args=parser.parse_args()
    output=R/'data'/args.mode;output.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    protocol=read(R/'protocol.json');cfg=read(Q/'config/automatic_alignment.json');inputs=read(Q/'data/inputs.json')
    sources={s['sample_id']:s for s in inputs['samples']};posts=read((Path(os.environ.get('Q1_PHONE_POSTERIORS','/workspace/q1-phone-posteriors'))/'manifest.json'));assert posts['complete']
    posteriors={};meta={};asrs={}
    for s in posts['records']:
        sid=s['sample_id'];p=Path(os.environ.get('Q1_PHONE_POSTERIORS','/workspace/q1-phone-posteriors'))/(sid+'.npz');assert digest(p)==s['posterior_sha256']
        with np.load(p) as a:posteriors[sid]=(a['logp'].copy(),a['frame_cells'].copy())
        meta[sid]=read(Q/'data/features'/(sid+'.json'))['words'];asrs[sid]=read(Q/'data/whisper_recognition'/(sid+'.json'))['words']
    pron=Pronunciations(read((Path(os.environ.get('Q1_PHONE_MODEL','/workspace/q1-phone-model'))/'vocab.json')),mode=args.mode,dictionary=os.environ.get('Q1_PHONE_DICTIONARY','/workspace/q1-mfa-models/dictionary-english_mfa.dict') if args.mode=='lexicon' else None)
    rows=[];cases=[]
    for index,p in enumerate(read(R/'data/proposals.json')):
        result=score(p,p['sample_id'],posteriors,sources,pron,protocol)
        rows.append({**p,**result})
        if index%50==0:print('identity',index,sum(r['accepted'] for r in rows),flush=True)
    write(output/'identity.json',{'protocol_sha256':digest(R/'protocol.json'),'rows':rows})
    comparisons=0
    for sid,s in sources.items():
        if s['zero_audio']:continue
        for ref_id,other in sources.items():
            if s['video_id']==other['video_id']:continue
            comparisons+=1
            for p in propose(meta[ref_id],asrs[sid],cfg,[s['audio']['start_s'],s['audio']['end_s']],protocol['maximum_interior_words']):
                result=score(p,sid,posteriors,sources,pron,protocol)
                cases.append({'audio_sample_id':sid,'reference_sample_id':ref_id,**p,**result})
    write(output/'cross_video.json',{'comparisons':comparisons,'rows':cases})
    shifted=[]
    for p in rows:
        sid=p['sample_id'];s=sources[sid];duration=p['interval_s'][1]-p['interval_s'][0]
        a,b=s['audio']['start_s'],s['audio']['end_s']
        # Select the farther endpoint window, only when disjoint from original.
        choices=[[a,a+duration],[b-duration,b]]
        choices=[t for t in choices if (t[1]<=p['interval_s'][0] or t[0]>=p['interval_s'][1]) and t[0]>=a and t[1]<=b]
        if not choices:continue
        bounds=max(choices,key=lambda t:abs(t[0]-p['interval_s'][0]))
        shifted_proposal={**p,'interval_s':bounds};result=score(shifted_proposal,sid,posteriors,sources,pron,protocol)
        shifted.append({'sample_id':sid,'word_indices':p['word_indices'],'original_interval_s':p['interval_s'],'shifted_interval_s':bounds,**result})
    write(output/'shifted.json',{'rows':shifted})
    # Text replacement keeps true anchor words and position; corruption is deterministic.
    vocabulary=sorted({w['text'].lower() for ws in meta.values() for w in ws if w['text'].isalpha() and len(w['text'])>=3})
    corrupted=[]
    for p in rows:
        ws=p['reference_words'];index=1+(len(ws)-2)//2;word=ws[index].lower()
        candidates=[w for w in vocabulary if w not in [s.lower() for s in ws] and abs(len(w)-len(word))<=2]
        if not candidates:continue
        replacement=min(candidates,key=lambda w:hashlib.sha256((p['sample_id']+'|'+word+'|'+w).encode()).digest())
        new=copy.deepcopy(p);new['reference_words'][index]=replacement
        result=score(new,p['sample_id'],posteriors,sources,pron,protocol)
        corrupted.append({'sample_id':p['sample_id'],'word_indices':p['word_indices'],'original_word':word,'replacement':replacement,**result})
    write(output/'corrupted.json',{'rows':corrupted})
    summary={'pronunciation_mode':args.mode,'status':'complete','identity':{'proposals':len(rows),'accepted':sum(r['accepted'] for r in rows),'status_counts':dict(collections.Counter(e['status'] for r in rows for e in r['evidence']))},
      'cross_video':{'comparisons':comparisons,'proposals':len(cases),'accepted':sum(r['accepted'] for r in cases)},
      'shifted':{'proposals':len(shifted),'accepted':sum(r['accepted'] for r in shifted)},
      'text_corruption':{'proposals':len(corrupted),'accepted':sum(r['accepted'] for r in corrupted)},
      'protocol_sha256':digest(R/'protocol.json'),'posterior_manifest_sha256':digest((Path(os.environ.get('Q1_PHONE_POSTERIORS','/workspace/q1-phone-posteriors'))/'manifest.json')),
      'scripts_sha256':{p.name:digest(p) for p in (R/'scripts').glob('*.py')},'human_data_used':False,'ground_truth_accuracy_measured':False}
    write(output/'summary.json',summary);print(summary)
if __name__=='__main__':main()
