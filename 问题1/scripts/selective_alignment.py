"""Edit-tolerant, all-optimal-path reference/ASR correspondence; no human input."""
import numpy as np

def char_distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1):row.append(min(row[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=row
    return previous[-1]

def optimal_relations(reference,hypothesis,cfg):
    """Free external hypothesis tokens; charged internal insertion/deletion/substitution.
    Enumerate possible mates/unmatched states over ALL optimal paths using forward/backward DP.
    """
    n,m=len(reference),len(hypothesis);unit=cfg['edit_cost'];cost=np.full((n,m),unit,dtype=np.int64);compatible=np.zeros((n,m),bool)
    for i,a in enumerate(reference):
        for j,b in enumerate(hypothesis):
            if a==b:cost[i,j]=0;compatible[i,j]=True
            elif min(len(a),len(b))>=cfg['approximate_min_chars']:
                distance=char_distance(a,b)/max(len(a),len(b))
                if distance<=cfg['approximate_max_distance']:cost[i,j]=round(unit*distance);compatible[i,j]=True
    f=np.zeros((n+1,m+1),dtype=np.int64);f[:,0]=np.arange(n+1)*unit
    for i in range(1,n+1):
        for j in range(1,m+1):f[i,j]=min(f[i-1,j-1]+cost[i-1,j-1],f[i-1,j]+unit,f[i,j-1]+unit)
    b=np.zeros_like(f);b[:,m]=np.arange(n,-1,-1)*unit
    for i in range(n-1,-1,-1):
        for j in range(m-1,-1,-1):b[i,j]=min(cost[i,j]+b[i+1,j+1],unit+b[i+1,j],unit+b[i,j+1])
    optimum=int(f[n].min());out=[]
    for i in range(n):
        mates=[j for j in range(m) if compatible[i,j] and f[i,j]+cost[i,j]+b[i+1,j+1]==optimum]
        unmatched=any(f[i,j]+unit+b[i+1,j]==optimum for j in range(m+1)) or any(not compatible[i,j] and f[i,j]+cost[i,j]+b[i+1,j+1]==optimum for j in range(m))
        stable=len(mates)==1 and not unmatched
        out.append({'hypothesis_options':mates,'unmatched_possible':bool(unmatched),'stable':stable,'exact':bool(stable and reference[i]==hypothesis[mates[0]])})
    return {'edit_cost':optimum/unit,'reference_tokens':out,'optimal_endpoints':np.flatnonzero(f[n]==optimum).tolist()}

def align_sample(words,hypothesis,baseline,cfg,no_signal=False):
    ref=[p for w in words for p in w['parts']];hyp=[h['text'] for h in hypothesis];dp=optimal_relations(ref,hyp,cfg)
    informative=[i for i,t in enumerate(ref) if len(t)>=cfg['informative_min_chars'] and t not in cfg['function_words']]
    anchors={ref[i] for i in informative if any(ref[i]==hyp[j] for j in dp['reference_tokens'][i]['hypothesis_options'])}
    informative={ref[i] for i in informative}
    coverage=len(anchors)/len(informative) if informative else 0.
    status='supported' if len(anchors)>=cfg['minimum_informative_anchors'] and coverage>=cfg['minimum_informative_coverage'] else 'insufficient_evidence'
    if not anchors:status='no_informative_support'
    if no_signal:status='no_signal'
    outputs=[];cursor=0
    for i,w in enumerate(words):
        parts=dp['reference_tokens'][cursor:cursor+len(w['parts'])];cursor+=len(w['parts'])
        stable=bool(parts) and all(p['stable'] for p in parts)
        indices=[p['hypothesis_options'][0] for p in parts] if stable else []
        stable=stable and indices==list(range(indices[0],indices[0]+len(indices)))
        exact=stable and all(p['exact'] for p in parts)
        interval=[hypothesis[indices[0]]['interval_s'][0],hypothesis[indices[-1]]['interval_s'][1]] if stable else None
        difference=max(abs(interval[j]-float(baseline[i,j])) for j in range(2)) if interval else None
        eligible=status=='supported' and exact and difference<=cfg['boundary_agreement_s']
        reason='automatic_candidate' if eligible else 'sample_support_insufficient' if status!='supported' else 'unmatched_or_ambiguous' if not stable else 'approximate_lexical_match' if not exact else 'boundary_disagreement'
        outputs.append({'word_index':i,'text':w['text'],'status':reason,'automatic_eligible':bool(eligible),'interval_s':interval if eligible else None,'diagnostic_interval_s':interval,'boundary_disagreement_s':difference,'recognition_options':parts,'human_verified':False})
    # Context envelopes are navigation aids, never word intervals or acceptance masks.
    # They bracket uncertain words by neighboring stable lexical anchors. At the edges,
    # retain the observed audio extent rather than inventing a missing word boundary.
    context=[]
    anchor_words=[i for i,w in enumerate(outputs) if w['diagnostic_interval_s'] is not None and all(p['exact'] for p in w['recognition_options'])]
    if status=='supported' and hypothesis:
        pending=[i for i,w in enumerate(outputs) if not w['automatic_eligible']]
        runs=[]
        for i in pending:
            if runs and i==runs[-1][-1]+1:runs[-1].append(i)
            else:runs.append([i])
        for run in runs:
            left=next((i for i in reversed(anchor_words) if i<run[0]),None)
            right=next((i for i in anchor_words if i>run[-1]),None)
            lo=outputs[left]['diagnostic_interval_s'][0] if left is not None else hypothesis[0]['interval_s'][0]
            hi=outputs[right]['diagnostic_interval_s'][1] if right is not None else hypothesis[-1]['interval_s'][1]
            if hi>lo:context.append({'word_indices':run,'interval_s':[lo,hi],'left_anchor_word':left,'right_anchor_word':right,'status':'context_only_not_alignment','automatic_eligible':False})
    return {'context_envelopes':context,'sample_status':status,'informative_anchors':len(anchors),'informative_reference_tokens':len(informative),'informative_coverage':coverage,'edit_cost':dp['edit_cost'],'optimal_endpoints':dp['optimal_endpoints'],'words':outputs,'automatic_eligible_words':sum(w['automatic_eligible'] for w in outputs)}

def automatic_rows(sample,result,word_index,video_native):
    """Automatic relation reader. Never consumes an annotation or reviewed relation."""
    arrays,meta=sample
    if result['source_sha256']!=meta['source_sha256'] or result['sample_id']!=meta['sample_id']:raise ValueError('Automatic result source mismatch')
    word=result['words'][word_index]
    if word['word_index']!=word_index or word['text']!=meta['words'][word_index]['text']:raise ValueError('Automatic result text index mismatch')
    out={'status':word['status'],'automatic_eligible':word['automatic_eligible'],'human_verified':False,'audio':[],'scene':[]}
    if not word['automatic_eligible']:return out
    interval=word['interval_s']
    if interval is None or not np.isfinite(interval).all() or not meta['audio_start_s']<=interval[0]<interval[1]<=meta['audio_start_s']+meta['resampled_samples']/16000:raise ValueError('Invalid automatic time interval')
    from feature_io import candidate_rows
    view=dict(arrays);view['word_candidate_mask']=np.array([True]);view['word_intervals']=np.array([interval]);out.update(candidate_rows(view,0,video_native));return out
