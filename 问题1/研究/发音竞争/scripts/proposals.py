import sys
from pathlib import Path
import os
sys.path.insert(0,str(Path(os.environ.get('Q1_PROJECT',Path(__file__).resolve().parents[3]))/'scripts'))
from whisper_support import lexical_support


def propose(reference, recognized, cfg, bounds, max_interior=4):
    support=lexical_support(reference,recognized,cfg,*bounds)
    anchors=[]
    for i,w in enumerate(reference):
        row=support['words'][i]
        content={p for p in w['parts'] if len(p)>=cfg['informative_min_chars'] and p not in cfg['function_words']}
        if content and row['exact_unique'] and row['valid_time'] and row['atomic_word_time']:
            anchors.append((i,row,content))
    proposals=[]
    for (i,a,ca),(j,b,cb) in zip(anchors,anchors[1:]):
        if not 1<=j-i-1<=max_interior or not ca.isdisjoint(cb):continue
        if a['interval_s'][1]>b['interval_s'][0] or a['raw_rows'][-1]>=b['raw_rows'][0]:continue
        raw_start=a['raw_rows'][0];raw_stop=b['raw_rows'][-1]+1
        proposals.append({'word_indices':list(range(i,j+1)),'interior_indices':list(range(i+1,j)),
                          'interval_s':[a['interval_s'][0],b['interval_s'][1]],
                          'reference_words':[w['text'] for w in reference[i:j+1]],
                          'recognized_text':' '.join(w['text'] for w in recognized[raw_start:raw_stop]),
                          'recognized_raw_rows':[raw_start,raw_stop]})
    return proposals
