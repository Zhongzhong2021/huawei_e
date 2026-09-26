"""Phonetic variants and CTC transcript competition; scores are not correctness probabilities."""
import itertools, math, json
from functools import lru_cache
from pathlib import Path
import os
import sys
sys.path.insert(0,str(Path(os.environ.get('Q1_PROJECT',Path(__file__).resolve().parents[3]))/'scripts'))
from math_features import words as tokenize
import numpy as np
import torch
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

class Pronunciations:
    def __init__(self, vocab, mode="contextual", dictionary=None):
        self.vocab=vocab
        self.mode=mode
        self.lexicon={}
        if dictionary:
            for line in Path(dictionary).read_text().splitlines():
                fields=line.split('\t')
                if len(fields)<6:continue
                phone_tokens=fields[-1].split()
                if all(p in vocab and vocab[p]>=4 for p in phone_tokens):
                    self.lexicon.setdefault(fields[0],[]).append((float(fields[1]),tuple(vocab[p] for p in phone_tokens)))
            self.lexicon={w:[p for _,p in sorted(rows,key=lambda x:(-x[0],x[1]))[:3]] for w,rows in self.lexicon.items()}

        self.backend=EspeakBackend('en-us',preserve_punctuation=False,with_stress=False,tie=False,language_switch='remove-flags',words_mismatch='ignore')
        self.separator=Separator(phone=' ',word=' | ',syllable='')

    @lru_cache(maxsize=10000)
    def phones(self, text):
        normalized=' '.join(p for word in tokenize(text) for p in word['parts']).lower()
        return self.backend.phonemize([normalized],separator=self.separator,strip=True,njobs=1)[0]

    @lru_cache(maxsize=10000)
    def variants(self, words):
        isolated=[self.phones(w).replace('|',' ').split() for w in words]
        contextual=[p.split() for p in self.phones(' '.join(words)).split('|')]
        if self.mode=='canonical':
            flat=[p for word in isolated for p in word]
            return (tuple(self.vocab[p] for p in flat),) if flat and all(p in self.vocab and self.vocab[p]>=4 for p in flat) else ()
        choices=[]
        for i,original in enumerate(isolated):
            alternatives=[original]
            if len(contextual)==len(words) and contextual[i] and contextual[i]!=original:alternatives.append(contextual[i])
            choices.append(alternatives)
        if not choices or math.prod(map(len,choices))>64:return ()
        sequences=set()
        for combination in itertools.product(*choices):
            flat=[p for word in combination for p in word]
            if not flat or any(p not in self.vocab or self.vocab[p]<4 for p in flat):return ()
            sequences.add(tuple(self.vocab[p] for p in flat))
        # eSpeak may fuse contextual words. Keep that whole-group path without
        # inventing phone ownership for its internal words.
        group=self.phones(' '.join(words)).replace('|',' ').split()
        if group and all(p in self.vocab and self.vocab[p]>=4 for p in group):
            sequences.add(tuple(self.vocab[p] for p in group))
        if self.mode=='lexicon':
            # A single interior-word substitution per path controls expansion.
            # Contextual fused paths remain available as complete hypotheses.
            for index,word in enumerate(words[1:-1],1):
                left=[self.vocab[p] for group in isolated[:index] for p in group]
                right=[self.vocab[p] for group in isolated[index+1:] for p in group]
                for variant in self.lexicon.get(word.lower(),[]):
                    sequences.add(tuple(left)+variant+tuple(right))
        return tuple(sorted(sequences)) if len(sequences)<=64 else ()


def ctc_scores(logp, sequences, blank=0):
    if not sequences:return np.zeros(0)
    values=torch.as_tensor(np.asarray(logp),dtype=torch.float32)
    targets=torch.tensor([x for s in sequences for x in s],dtype=torch.long)
    lengths=torch.tensor([len(s) for s in sequences],dtype=torch.long)
    with torch.inference_mode():
        losses=torch.nn.functional.ctc_loss(values[:,None,:].expand(-1,len(sequences),-1).contiguous(),targets,
                   torch.full((len(sequences),),len(logp),dtype=torch.long),lengths,blank=blank,reduction='none',zero_infinity=False)
    return -losses.numpy()


def mixture_score(logp, variants):
    scores=ctc_scores(logp,variants)
    if not len(scores):return None
    finite=scores[np.isfinite(scores)]
    if not len(finite):return None
    maximum=float(finite.max())
    return maximum+math.log(float(np.exp(finite-maximum).sum()))-math.log(len(scores))


def compete(logp, words, alternative, pronunciations, margin):
    target=pronunciations.variants(tuple(words))
    if not target:return {'accepted':False,'status':'unrepresentable_target'}
    hypotheses=[('delete_'+str(i),pronunciations.variants(tuple(words[:i]+words[i+1:]))) for i in range(len(words))]
    alternate=pronunciations.variants(tuple(alternative)) if alternative else ()
    asr_equivalent=bool(alternate and set(alternate)&set(target))
    normalized=lambda words:[p for word in tokenize(' '.join(words)) for p in word['parts']]
    if asr_equivalent and normalized(words) != normalized(alternative):
        return {'accepted':False,'status':'phonemically_ambiguous_alternative'}
    if alternate and not asr_equivalent:hypotheses.append(('asr_alternative',alternate))
    if alternative and not alternate:return {'accepted':False,'status':'unrepresentable_asr_alternative'}
    target_score=mixture_score(logp,target)
    if target_score is None:return {'accepted':False,'status':'impossible_target'}
    margins={};scores={}
    for name,variants in hypotheses:
        if not variants:return {'accepted':False,'status':'unrepresentable_competitor'}
        if set(variants)&set(target):return {'accepted':False,'status':'indistinguishable_deletion'}
        value=mixture_score(logp,variants)
        if value is None:return {'accepted':False,'status':'impossible_competitor'}
        scores[name]=value;margins[name]=target_score-value
    closed_set=bool(margins) and min(margins.values())>=margin
    argmax=np.asarray(logp).argmax(-1)
    unconstrained=tuple(int(p) for i,p in enumerate(argmax) if p!=0 and (i==0 or p!=argmax[i-1]))
    unconstrained_score=float(ctc_scores(logp,[unconstrained])[0])
    open_set_margin=target_score-unconstrained_score
    normalized_open_set_margin=open_set_margin/min(map(len,target))
    accepted=closed_set and normalized_open_set_margin>=-margin
    return {'accepted':accepted,'status':'model_supported' if accepted else 'open_set_rejected' if closed_set else 'competition_rejected',
            'closed_set_accepted':closed_set,'unconstrained_score':unconstrained_score,'open_set_margin':open_set_margin,'normalized_open_set_margin':normalized_open_set_margin,'target_minimum_phones':min(map(len,target)),
            'target_score':target_score,'competitor_scores':scores,'margins':margins,
            'target_variants':len(target),'asr_phonemically_equivalent':asr_equivalent}
