import os
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(os.environ.get('Q1_PROJECT','C:/WorkArea/Huawei-Model'))/'scripts'))
from common import load_numeric
W=Path(os.environ.get('Q1_EXTERNAL_DIR',str(Path(__file__).resolve().parent.parent)));R=Path(os.environ.get('Q1_WORKSPACE',str(W.parent)));F=Path(os.environ.get('Q1_FEATURES',str(W.parent/'features/samples')))
d=load_numeric(Path(os.environ.get('Q1_PROJECT','C:/WorkArea/Huawei-Model'))/'data/raw/附件2-数据集特征文件/aligned_50.pkl')
lookup={str(sid):(split,i) for split,x in d.items() for i,sid in enumerate(x['id'])};rows=[];errs=[]
for p in F.glob('*.json'):
    m=json.loads(p.read_text(encoding='utf-8'));sid=m['sample_id'];r=dict(sample_id=sid,id_found=sid in lookup,exact_encoder_context=False)
    if sid in lookup:
        split,i=lookup[sid];ref=d[split];z=np.load(p.with_suffix('.npz'));tb=ref['text_bert'][i];ids=tb[0][tb[1]!=0];own=z['bert_token_ids'];r['split']=split;r['exact_encoder_context']=bool(np.array_equal(ids,own));r['reference_tokens']=len(ids);r['local_tokens']=len(own)
        if r['exact_encoder_context']:
            membership=z['bert_word_membership'];h=ref['text'][i][:len(ids)];pooled=membership@h/np.maximum(membership.sum(1,keepdims=True),1e-12);delta=abs(pooled-z['text']);errs.append(delta.ravel());r.update(words=len(pooled),mae=float(delta.mean()),max_abs=float(delta.max()))
    rows.append(r)
allerr=np.concatenate(errs) if errs else np.array([])
summary=dict(samples=len(rows),same_id=sum(r['id_found'] for r in rows),same_encoder_input=sum(r['exact_encoder_context'] for r in rows),words=sum(r.get('words',0) for r in rows),mae=float(allerr.mean()) if len(allerr) else None,max_abs=float(allerr.max()) if len(allerr) else None,meaning='Annex2 supplied precomputed BERT representation compared after the exact same word-membership pooling; excludes different input contexts, truncation and unmatched IDs; no labels read')
(W/'text_reference_summary.json').write_text(json.dumps(summary,indent=2));(W/'text_reference_samples.json').write_text(json.dumps(rows,indent=2));print(json.dumps(summary,indent=2))
