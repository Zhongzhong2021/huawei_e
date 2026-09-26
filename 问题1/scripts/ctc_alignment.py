"""CTC outer wildcards may consume zero real frames; virtual boundary frames are not media."""
from dataclasses import dataclass
import torch
import torchaudio

@dataclass(frozen=True)
class Span:
    token:int
    start:int
    end:int
    score:float

def align_tokens(logp,ids,outer=False,blank=0):
    target=list(ids);scores=logp
    if outer:
        star=logp.shape[-1];real=torch.cat([logp,torch.zeros((*logp.shape[:2],1),dtype=logp.dtype,device=logp.device)],dim=2)
        virtual=torch.full((1,1,star+1),float('-inf'),dtype=logp.dtype,device=logp.device);virtual[0,0,star]=0
        scores=torch.cat([virtual,real,virtual],dim=1);target=[star]+target+[star]
    path,values=torchaudio.functional.forced_align(scores,torch.tensor([target],dtype=torch.int32,device=logp.device),blank=blank)
    spans=torchaudio.functional.merge_tokens(path[0],values[0].exp(),blank=blank)
    if [s.token for s in spans]!=target:raise ValueError('Token reconstruction failed')
    if outer:spans=spans[1:-1]
    shift=int(outer);out=[Span(s.token,s.start-shift,s.end-shift,float(s.score)) for s in spans]
    if any(not 0<=s.start<s.end<=logp.shape[1] for s in out):raise ValueError('Virtual frame escaped into source time')
    return out
