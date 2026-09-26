"""Positive tests of a deletion hypothesis; window-local evidence, never global absence."""
import os
import sys
from pathlib import Path
import numpy as np
import torch
from phone_evidence import Pronunciations, compete
from evaluate_competition import Q, R
from common import read, write, digest


def confirmed_windows(evidence):
    return (len(evidence) == 2 and [e.get('context_s') for e in evidence] == [0.0, 0.1]
            and all(e.get('accepted') for e in evidence))


def main():
    torch.set_num_threads(2)
    cache=Path(os.environ.get('Q1_PHONE_POSTERIORS','/workspace/q1-phone-posteriors'))
    model=Path(os.environ.get('Q1_PHONE_MODEL','/workspace/q1-phone-model'))
    manifest=read(cache/'manifest.json')
    assert digest(cache/'manifest.json')==digest(R/'data/posterior_manifest.json')
    pron=Pronunciations(read(model/'vocab.json'),mode='contextual')
    protocol=read(R/'protocol.json');sources={s['sample_id']:s for s in read(Q/'data/inputs.json')['samples']}
    records={s['sample_id']:s for s in manifest['records']};posteriors={};rows=[]
    for number,p in enumerate(read(R/'data/proposals.json')):
        sid=p['sample_id'];source=sources[sid]
        if sid not in posteriors:
            path=cache/(sid+'.npz');assert digest(path)==records[sid]['posterior_sha256']
            with np.load(path) as a:posteriors[sid]=(a['logp'].copy(),a['frame_cells'].copy())
        logp,clock=posteriors[sid]
        for word_index in p['interior_indices']:
            local=p['word_indices'].index(word_index)
            full=p['reference_words'];deleted=full[:local]+full[local+1:];evidence=[]
            for context in protocol['contexts_s']:
                lo=max(source['audio']['start_s'],p['interval_s'][0]-context)
                hi=min(source['audio']['end_s'],p['interval_s'][1]+context)
                selected=np.flatnonzero((clock.mean(1)>=lo)&(clock.mean(1)<hi))
                result=(compete(logp[selected],deleted,full,pron,protocol['minimum_log_likelihood_margin'])
                        if len(selected) else {'accepted':False,'status':'empty_audio_window'})
                evidence.append({'context_s':context,'interval_s':[float(lo),float(hi)],**result})
            rows.append({'sample_id':sid,'proposal_row':number,'word_index':word_index,
                         'word_indices':p['word_indices'],'interval_s':p['interval_s'],
                         'full_words':full,'deletion_words':deleted,'evidence':evidence,
                         'local_omission_supported':confirmed_windows(evidence),
                         'whole_audio_absence_established':False})
        if number%100==0:print('proposals',number,flush=True)
    result={'schema':'q1-local-omission-evidence-v1','scope':'Two anchor-bounded windows only; not a whole-audio absence decision.',
            'human_data_used':False,'threshold_calibrated':False,'code_sha256':digest(__file__),
            'phone_evidence_sha256':digest(R/'scripts/phone_evidence.py'),
            'protocol_sha256':digest(R/'protocol.json'),'proposals_sha256':digest(R/'data/proposals.json'),
            'posterior_manifest_sha256':digest(cache/'manifest.json'),
            'tests':len(rows),'supported_tests':sum(x['local_omission_supported'] for x in rows),'rows':rows}
    write(R/'data/local_omissions.json',result)
    print({k:v for k,v in result.items() if k!='rows'})

if __name__=='__main__':main()
