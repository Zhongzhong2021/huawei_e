"""Load native sequences; padding is separate from feature extraction."""
from pathlib import Path
import json
import numpy as np

def load(root,sid):
    root=Path(root)
    with np.load(root/(sid+'.npz'),allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    return arrays,json.loads((root/(sid+'.json')).read_text())

def pad(samples,key):
    if key not in ['text','audio','scene','faces']:raise ValueError('Unknown modality')
    lengths=np.array([len(a[key]) for a,m in samples],np.int64)
    if not len(samples):raise ValueError('Empty batch')
    dim=samples[0][0][key].shape[1];width=int(lengths.max(initial=0));values=np.zeros((len(samples),width,dim),np.float32);valid=np.zeros(values.shape,bool)
    available=np.array([bool(length) and not (key=='audio' and m.get('alignment',{}).get('status')=='no_signal')
                        for length,(a,m) in zip(lengths,samples)],bool)
    for i,(a,m) in enumerate(samples):
        assert a[key].shape==(lengths[i],dim);values[i,:lengths[i]]=a[key]
        valid[i,:lengths[i]]=(a['audio_dim_mask'] if key=='audio' else True) & available[i]
    values[~valid]=0
    return {'values':values,'lengths':lengths,'sequence_mask':np.arange(width)[None,:]<lengths[:,None],
            'dimension_mask':valid,'sample_available':available,
            'sample_ids':[m['sample_id'] for a,m in samples]}

def candidate_rows(arrays,word_index,video_native):
    """Time-overlap weights, not a semantic acceptance gate or speaker selection."""
    if not arrays['word_candidate_mask'][word_index]:return {'audio':[],'scene':[]}
    lo,hi=arrays['word_intervals'][word_index];out={}
    from fractions import Fraction
    tb=float(Fraction(video_native['time_base']));starts=np.array(video_native['pts_ticks'])*tb;ends=starts+np.array(video_native['lengths'])*tb
    for name,key in [('audio','audio_cells'),('scene','video_cells')]:
        hits=[]
        for i,(a,b) in enumerate(arrays[key]):
            left,right=max(a,lo),min(b,hi)
            weight=max(0,right-left) if name=='audio' else sum(max(0,min(right,e)-max(left,s)) for s,e in zip(starts,ends))
            if weight>0:hits.append([i,float(weight)])
        out[name]=hits
    return out
