import argparse,json,csv,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
rows=[]
for file in sorted((a.features/'samples').glob('*.npz')):
 z=np.load(file);m=json.loads(file.with_suffix('.json').read_text(encoding='utf-8'));w=len(m['words']);assert z['text'].shape==(w,768);assert z['bert_word_membership'].shape==(w,len(z['bert_token_ids']));assert hashlib.sha256(file.read_bytes()).hexdigest()==m['feature_sha256']
 unsupported=~z['alignment_supported'];unknown=z['bert_token_ids']==100
 assert z['text_mask'][unsupported].all();assert np.isfinite(z['text']).all()
 row=dict(sample_id=m['sample_id'],text_words=w,bert_subtokens=len(z['bert_token_ids']),unknown_subtokens=int(unknown.sum()),words_with_unknown_subtokens=int((z['bert_word_membership'][:,unknown].sum(1)>0).sum()),text_valid=int(z['text_mask'].sum()),supported_words=int(z['alignment_supported'].sum()),unsupported_words_with_text=int((unsupported&z['text_mask']).sum()),native_audio_rows=len(z['audio_native']),native_scene_rows=len(z['scene_native']),native_expression_valid=int(z['expression_valid'].sum()))
 assert row['native_audio_rows']>0 and row['native_scene_rows']>0;rows.append(row)
with (a.output/'全100条文本与模态接口核验.csv').open('w',encoding='utf-8-sig',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
summary=dict(samples=len(rows),text_words=sum(r['text_words'] for r in rows),bert_subtokens=sum(r['bert_subtokens'] for r in rows),unknown_subtokens=sum(r['unknown_subtokens'] for r in rows),words_with_unknown_subtokens=sum(r['words_with_unknown_subtokens'] for r in rows),text_valid=sum(r['text_valid'] for r in rows),unsupported_words_with_text=sum(r['unsupported_words_with_text'] for r in rows),no_word_alignment_samples=sum(r['supported_words']==0 for r in rows),all_samples_retain_native_audio_scene=True,scope='Q1 original-video feature interface only; no Annex3 masks applied; no feature values changed.')
assert summary['samples']==100
(a.output/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False))
