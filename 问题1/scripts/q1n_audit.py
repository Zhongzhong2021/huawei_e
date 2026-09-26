# 本程序及代码是在人工智能工具辅助下完成的。
# 工具：OpenAI Codex；模型：GPT-6 Astra (gpt-6-astra)；开发机构：OpenAI；模型发布日期：2026-09-03。
# AI 来源说明于 2026-09-24 补录，未改变算法；参见 AI辅助使用记录与队内审阅要求.md。
"""Prespecified fusion of independent lexical evidence and CTC support; no labels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from q1_extract import normalize,dump,sha
from q1n_posterior import soft_pool

ROOT=Path('results/q1_next');BASE=Path('results/q1/samples')
def match_subsequence(ref,hyp):
    """Levenshtein: all reference required, arbitrary surrounding audio ignored."""
    m,n=len(ref),len(hyp);d=np.zeros((m+1,n+1),int);d[:,0]=np.arange(m+1)
    for i in range(1,m+1):
        for j in range(1,n+1):d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(ref[i-1]!=hyp[j-1]))
    i,j=m,int(np.argmin(d[-1]));cost=int(d[i,j]);pairs=[]
    while i:
        if j and d[i,j]==d[i-1,j-1]+(ref[i-1]!=hyp[j-1]):
            if ref[i-1]==hyp[j-1]:pairs.append((i-1,j-1))
            i-=1;j-=1
        elif d[i,j]==d[i-1,j]+1:i-=1
        else:j-=1
    return cost/max(m,1),list(reversed(pairs))

def main():
    assert match_subsequence(['a','b'],['z','a','b','c'])==(0.,[(0,1),(1,2)])
    assert match_subsequence(['a','b'],[])==(1.,[])
    config=dict(max_subsequence_word_error=.25,min_ctc_word_score=.3,max_conditional_boundary_width_s=.2,
                max_asr_ctc_interval_gap_s=.2,min_asr_word_probability=.5,min_local_anchor_words=3,
                note='Prespecified diagnostic rules, not calibrated probability or accuracy; no sentiment labels.')
    dump(ROOT/'fusion_config.json',config);dest=ROOT/'enhanced';dest.mkdir(exist_ok=True)
    rows=[];word_rows=[];maxpool=0.;checks=0;allwidth=[];oldwidth=[];deltas={k:[] for k in ('audio','vision','scene')}
    for p in sorted(BASE.glob('*.json'),key=lambda p:json.loads(p.read_text())['table_row_0based']):
        m=json.loads(p.read_text());a=json.loads((ROOT/'asr'/p.name).read_text());post=json.loads((ROOT/'posterior'/p.name).read_text())
        z=np.load(p.with_suffix('.npz'));f=np.load(ROOT/'posterior'/p.with_suffix('.npz').name)
        assert sha(p.with_suffix('.npz'))==post['baseline_feature_sha256'];checks+=1
        ref=[w['word'] for w in m['words']];hyp=[];hwords=[]
        for s in a['segments']:
            for w in s['words'] or []:
                normalized,_=normalize(w['word'])
                for token in normalized:
                    hyp.append(token['word']);hwords.append(w)
        wer,pairs=match_subsequence(ref,hyp)
        english=a['language']=='en';full_support=english and wer<=config['max_subsequence_word_error']
        anchors=set();run=[]
        for pair in pairs+[(-99,-99)]:
            if run and pair!=(run[-1][0]+1,run[-1][1]+1):
                if len(run)>=3:anchors.update(i for i,j in run)
                run=[]
            run.append(pair)
        stored_membership=f['word_time_membership'].astype(np.float64);ctc_cells=f['ctc_cells']
        starts=f['start_quantiles'];ends=f['end_quantiles'];width=np.maximum(starts[:,2]-starts[:,0],ends[:,2]-ends[:,0]);allwidth.extend(width)
        oldwidth.extend(width[z['alignment_valid']]);good=np.zeros(len(ref),bool);exact=np.zeros(len(ref),bool);gaps=np.full(len(ref),-1.,np.float32);aprob=np.zeros(len(ref),np.float32)
        for i,j in pairs:
            exact[i]=True;w=hwords[j];aprob[i]=w['probability'];offset=m['audio_start_s']
            gaps[i]=max(0,starts[i,1]-(w['end']+offset),(w['start']+offset)-ends[i,1])
            good[i]=english and (full_support or i in anchors) and gaps[i]<=config['max_asr_ctc_interval_gap_s'] and aprob[i]>=config['min_asr_word_probability']
        supported=good&(z['alignment_score']>=config['min_ctc_word_score'])&(width<=config['max_conditional_boundary_width_s'])
        result=dict(alignment_supported=supported,asr_exact_match=exact,asr_time_gap_s=gaps,asr_word_probability=aprob,
                    conditional_boundary_width_s=width.astype(np.float32))
        for k in ('audio','vision','scene'):
            value=f[k+'_soft_candidate'].copy();mask=supported&f[k+'_soft_presence'];value[~mask]=0
            # Store masks only; the loader applies them to shared posterior candidates.
            # Avoid duplicating ~5 MB of continuous features in the contest attachment.
            result[k+'_soft_mask']=mask
            old=z[k+'_candidate'].astype(np.float64);new=f[k+'_soft_candidate'];eligible=z['alignment_valid']&f[k+'_soft_presence']
            if eligible.any():
                # Direction change is descriptive, not a downstream accuracy gain.
                cosine=np.sum(old*new,1)/np.maximum(np.linalg.norm(old,axis=1)*np.linalg.norm(new,axis=1),1e-12)
                deltas[k].extend((1-cosine[eligible]).tolist())
            # Independent cell integration check at first/middle/last word.
            cells=z['audio_cells' if k=='audio' else 'vision_cells'];native=z[k+'_native'].astype(np.float64)
            valid=z['vision_valid'] if k=='vision' else np.ones(len(native),bool)
            for i in sorted(set([0,len(ref)//2,len(ref)-1])):
                weights=np.zeros(len(cells),np.float64)
                for t,ct in enumerate(ctc_cells):
                    weights+=stored_membership[i,t]*np.maximum(0,np.minimum(cells[:,1],ct[1])-np.maximum(cells[:,0],ct[0]))
                weights*=valid
                norm=weights/max(weights.sum(),1e-12);mean=norm@native
                expected=np.r_[mean,np.sqrt(np.maximum(norm@(native*native)-mean*mean,0))] if k=='audio' else mean
                if not f[k+'_soft_presence'][i]:expected[:]=0
                # The pooling path uses exactly the saved float32 occupancy values.
                err=float(np.max(abs(expected-new[i])/np.maximum(1,abs(new[i]))));maxpool=max(maxpool,err)
                assert err<.001, f'{p.name} {k} word {i}: scaled error {err}'
                checks+=1
        np.savez_compressed(dest/p.with_suffix('.npz').name,**result)
        status='no_vad_speech' if not a['segments'] else ('english_sentence_supported' if full_support else ('non_english_or_language_uncertain' if not english else 'english_low_text_agreement'))
        row=dict(sample_id=m['sample_id'],row=m['table_row_0based'],words=len(ref),baseline_accepted=bool(z['alignment_valid'].any()),
                 language=a['language'],language_probability=a['language_probability'],vad_seconds=a['duration_after_vad'],
                 subsequence_word_error_proxy=wer,status=status,baseline_words=int(z['alignment_valid'].sum()),
                 enhanced_words=int(supported.sum()),recovered_words=int((supported&~z['alignment_valid']).sum()),
                 withheld_words=int((~supported&z['alignment_valid']).sum()),wide_words=int((width>.2).sum()),
                 original_text=m['text_original'],asr_text=a['transcript'],translation=a.get('translation',''))
        rows.append(row)
        for i,w in enumerate(ref):word_rows.append(dict(sample_id=m['sample_id'],word_index=i,word=w,baseline=bool(z['alignment_valid'][i]),supported=bool(supported[i]),exact_match=bool(exact[i]),asr_probability=float(aprob[i]),ctc_score=float(z['alignment_score'][i]),boundary_width_s=float(width[i]),interval_gap_s=float(gaps[i])))
        dump(dest/p.name,dict(sample_id=m['sample_id'],baseline_feature_sha256=sha(p.with_suffix('.npz')),posterior_feature_sha256=sha(ROOT/'posterior'/p.with_suffix('.npz').name),feature_sha256=sha(dest/p.with_suffix('.npz').name),status=status,official_text_preserved=True))
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/'audit_100.csv',index=False,encoding='utf-8-sig');pd.DataFrame(word_rows).to_csv(ROOT/'word_audit.csv',index=False,encoding='utf-8-sig')
    metrics=dict(samples=len(rows),words=sum(r['words'] for r in rows),languages=frame.language.value_counts().to_dict(),status_counts=frame.status.value_counts().to_dict(),
                 baseline_supported_words=int(frame.baseline_words.sum()),enhanced_supported_words=int(frame.enhanced_words.sum()),
                 recovered_words=int(frame.recovered_words.sum()),withheld_words=int(frame.withheld_words.sum()),
                 samples_with_enhanced_words=int((frame.enhanced_words>0).sum()),boundary_width_quantiles_s=np.quantile(allwidth,[.5,.9,.95,.99]).tolist(),
                 baseline_accepted_boundary_width_quantiles_s=np.quantile(oldwidth,[.5,.9,.95,.99]).tolist(),
                 checks=checks,max_stored_membership_pool_relative_error=maxpool,
                 hard_soft_cosine_distance_median={k:float(np.median(v)) for k,v in deltas.items()})
    dump(ROOT/'metrics.json',metrics);print(json.dumps(metrics,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
