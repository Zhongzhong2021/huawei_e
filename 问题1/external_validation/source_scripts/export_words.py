import os
import sys,json,re
from pathlib import Path
W=Path(os.environ.get('Q1_EXTERNAL_DIR',str(Path(__file__).resolve().parent.parent)));R=Path(os.environ.get('Q1_WORKSPACE',str(W.parent)));sys.path.insert(0,str(R/'work/q1_v8/python_libs'));import h5py
vids=sorted({json.loads(p.read_text(encoding='utf-8'))['sample_id'].split('$_$')[0] for p in (Path(os.environ.get('Q1_FEATURES',str(W.parent/'features/samples')))).glob('*.json')})
out={}
with h5py.File(W/'public/CMU_MOSEI_TimestampedWords.csd') as h:
    for vid in vids:
        d=h['words/data'][vid];ints=d['intervals'][:];raw=[v[0].decode() for v in d['features'][:]];words=[]
        for i,(word,t) in enumerate(zip(raw,ints)):
            if word.lower() in ['sp','sil','<sil>']:continue
            for token in re.findall(r"[a-z0-9]+(?:'[a-z]+)*",word.lower().replace('’',"'").replace('‘',"'")):
                words.append(dict(word=token,raw=word,index=i,start=float(t[0]),end=float(t[1])))
        out[vid]=words
(W/'public_words.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(len(out))
