import os
import json,hashlib,csv
from pathlib import Path
import numpy as np
W=Path(os.environ.get('Q1_EXTERNAL_DIR',str(Path(__file__).resolve().parent.parent)));R=Path(os.environ.get('Q1_WORKSPACE',str(W.parent)))
a=json.loads((W/'audio_manifest.json').read_text());checks=[]
for r in a:
    p=W/'source_audio'/(r['video_id']+'.wav');digest=hashlib.md5(p.read_bytes()).hexdigest();assert digest==r['reference']['checksum'];checks.append(dict(video_id=r['video_id'],md5=digest,sha256=r['sha256'],bytes=r['bytes']))
(W/'audio_integrity.json').write_text(json.dumps(dict(files=len(checks),all_match_published_md5=True,bytes=sum(r['bytes'] for r in checks),checks=checks),indent=2))
anchors=json.loads((W/'waveform_anchors.json').read_text());ok=[r for r in anchors if r['accepted']]
rows=json.loads((W/'word_records.json').read_text());rr=[r for r in rows if r['eligible'] and r['supported']];vids=sorted({r['source_video'] for r in rr});groups=[[r for r in rr if r['source_video']==v] for v in vids];rng=np.random.default_rng(20260924);diff=[]
for _ in range(2000):
    b=[r for i in rng.integers(0,len(groups),len(groups)) for r in groups[i]];diff.append(np.mean([(r['MMS_FA_max_abs_s']<=.1)-(1.0 if r['WAV2VEC2_max_abs_s']<=.1 else 0.0) for r in b]))
large=[r for r in rr if r['MMS_FA_max_abs_s']>.2];large.sort(key=lambda r:r['MMS_FA_max_abs_s'],reverse=True)
for r in large:
    p=Path(os.environ.get('Q1_PROJECT','C:/WorkArea/Huawei-Model'))/'results/q1_next/asr'/(r['sample_id'].replace('$_$','__')+'.json');d=json.loads(p.read_text(encoding='utf-8'));r['asr_transcript']=d['transcript'];r['asr_same_spelling_candidates']=[dict(word=x['word'],start=x['start'],end=x['end'],probability=x['probability']) for seg in d['segments'] for x in seg['words'] if x['word'].strip().lower().strip('.,!?\"')==r['word']]
(W/'large_difference_cases.json').write_text(json.dumps(large,ensure_ascii=False,indent=2),encoding='utf-8')
summary=dict(anchored_samples=len(ok),min_waveform_correlation=min(abs(r['correlation']) for r in ok),median_waveform_correlation=float(np.median([abs(r['correlation']) for r in ok])),max_three_window_spread_s=max(r['window_offset_spread_s'] for r in ok),rejected_anchors=[r for r in anchors if not r['accepted']],accepted_word_reference_coverage=len(rr)/1464,all_word_reference_coverage=sum(r['eligible'] for r in rows)/1948,over200ms_supported_words=len(large),paired_MMS_minus_wav2vec_within100ms_ci95=np.quantile(diff,[.025,.975]).tolist(),fixed_fusion_note='Equal averaging is an additional descriptive ablation proposed after the initial baseline comparison; no fitted combination or claimed held-out generalization.')
(W/'evidence_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='rejected_anchors'},indent=2))
