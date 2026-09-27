"""Map tool-normalized labels back to immutable source words; no guessed boundaries."""
import re

def letters(text):return re.sub('[^a-z]','',text.lower())
def map_word_tier(source_words,entries,offset_s,duration_s):
    observed=[(float(a),float(b),str(t)) for a,b,t in entries if str(t).strip()]
    expected=[letters(''.join(w['parts'])) for w in source_words]
    obs=[letters(t) for a,b,t in observed]
    base=[{'word_index':i,'text':w['text'],'char_span':w['char_span'],'interval_s':None,'status':'unmapped','human_verified':False} for i,w in enumerate(source_words)]
    if any(not t for t in obs) or any(t in ['<unk>','spn','<eps>'] for a,b,t in observed) or ''.join(obs)!=''.join(expected):return base,'label_sequence_mismatch'
    if any(not 0<=a<b<=duration_s+.001 for a,b,t in observed) or any(right[0]<left[1]-1e-7 for left,right in zip(observed,observed[1:])):return base,'invalid_tool_interval'
    starts={};ends={};pos=0
    for i,t in enumerate(obs):starts[pos]=i;pos+=len(t);ends[pos]=i
    pos=0
    for i,token in enumerate(expected):
        end=pos+len(token)
        if pos in starts and end in ends and token:
            first,last=starts[pos],ends[end];base[i].update(interval_s=[offset_s+observed[first][0],offset_s+min(observed[last][1],duration_s)],status='acoustic_candidate',tool_word_rows=list(range(first,last+1)))
        else:base[i]['status']='boundary_inside_tool_token'
        pos=end
    return base,'mapped'
