# 本程序及代码是在人工智能工具辅助下完成的。
# 工具：OpenAI Codex；模型：GPT-6 Astra (gpt-6-astra)；开发机构：OpenAI；模型发布日期：2026-09-03。
# AI 来源说明于 2026-09-24 补录，未改变算法；参见 AI辅助使用记录与队内审阅要求.md。
"""Reject unsupported word alignment, retaining native features and diagnostic candidates."""
import json,csv
from pathlib import Path
import numpy as np
from q1_extract import pool,sha

root=Path('results/q1')
config={'version':'q1-quality-v1','word_min_score':.3,'sample_min_mean_score':.55,
        'sample_max_low_score_fraction':.35,'empty_greedy_rejected':True,
        'interpretation':'heuristic acoustic support filter, not calibrated accuracy',
        'script_sha256':sha(__file__)}
(root/'quality_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
rows=[]
for f in sorted((root/'samples').glob('*.npz')):
    with np.load(f,allow_pickle=False) as z:s={k:z[k] for k in z.files}
    meta=f.with_suffix('.json');m=json.loads(meta.read_text(encoding='utf-8'))
    eligible=bool(m['greedy_acoustic_transcript_unsegmented']) and m['alignment_score_mean']>=.55 and m['low_score_words']/m['word_count']<=.35
    reasons=[]
    if not m['greedy_acoustic_transcript_unsegmented']:reasons.append('no lexical acoustic output; music/silence or model failure requires review')
    if m['alignment_score_mean']<.55:reasons.append('mean acoustic alignment score below 0.55')
    if m['low_score_words']/m['word_count']>.35:reasons.append('over 35% of words have score below 0.3')
    s['alignment_valid']=(s['alignment_score']>=.3)&eligible
    for modality in ['audio','vision']:
        candidate=modality+'_candidate'
        if candidate not in s:
            s[candidate]=s[modality].copy();s[modality+'_presence_mask']=s[modality+'_mask'].copy()
        s[modality+'_mask']=s[modality+'_presence_mask']&s['alignment_valid']
        s[modality]=s[candidate].copy();s[modality][~s[modality+'_mask']]=0
    s['alignment_review']=~s['alignment_valid']
    # Every sample still has complete native features and explicit whole-clip summaries.
    audio_interval=np.array([[m['audio_start_s'],m['audio_start_s']+m['decoded_audio_duration_s']]])
    vision_interval=np.array([[s['vision_cells'][0,0],s['vision_cells'][-1,1]]])
    s['clip_text']=s['text'].mean(0)
    s['clip_audio']=pool(audio_interval,s['audio_cells'],s['audio_native'],np.ones(len(s['audio_native']),bool),std=True)[0][0]
    s['clip_vision']=pool(vision_interval,s['vision_cells'],s['vision_native'],s['vision_valid'])[0][0]
    s['clip_presence_mask']=np.array([True,len(s['audio_native'])>0,bool(s['vision_valid'].any())])
    s['clip_intervals']=np.vstack([audio_interval,vision_interval])
    for i,w in enumerate(m['words']):
        w['review_required']=not bool(s['alignment_valid'][i]);w['alignment_valid']=bool(s['alignment_valid'][i])
    m.update(word_alignment_usable=eligible,reliable_word_count=int(s['alignment_valid'].sum()),
             reliable_audio_words=int(s['audio_mask'].sum()),reliable_vision_words=int(s['vision_mask'].sum()),
             alignment_rejection_reasons=reasons,
             fallback='native timelines and clip-level summaries; no reliable word cross-modal mapping' if not eligible else 'word-level with per-word confidence mask')
    np.savez_compressed(f,**s);m['feature_sha256']=sha(f);m['shape']={k:list(v.shape) for k,v in s.items()}
    meta.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    rows.append({k:m[k] for k in ['sample_id','word_alignment_usable','word_count','reliable_word_count','alignment_score_mean','low_score_words','face_coverage','fallback']})
with (root/'quality_status_100.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({'samples':len(rows),'word_alignment_usable_samples':sum(r['word_alignment_usable'] for r in rows),
                  'reliable_words':sum(r['reliable_word_count'] for r in rows),'review_samples':[r['sample_id'] for r in rows if not r['word_alignment_usable']]},indent=2))
