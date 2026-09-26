import json,urllib.request,hashlib,zipfile,io,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
W=Path('/work');F=Path('/features/samples');base='https://audb-public.s3.dualstack.eu-north-1.amazonaws.com/'
def fetch(key):
    for i in range(3):
        try:return urllib.request.urlopen(base+key,timeout=90).read()
        except Exception:
            if i==2:raise
            time.sleep(2)
p=W/'db.parquet'
if not p.exists():p.write_bytes(fetch('cmu-mosei/1.2.4/db.parquet'))
df=pd.read_parquet(p).set_index('file');print(df.head().to_string(),flush=True);print(df.columns.tolist(),flush=True)
vids={json.loads(p.read_text())['sample_id'].split('$_$')[0] for p in F.glob('*.json')}
rows=[(name,row.to_dict()) for name,row in df.iterrows() if Path(name).stem in vids]
(W/'audio_index.json').write_text(json.dumps(rows,indent=2,default=str));print('matched',len(rows),flush=True)
out=W/'source_audio';out.mkdir(exist_ok=True)
def one(item):
    name,row=item;target=out/Path(name).name;key=f"cmu-mosei/media/{row['version']}/{row['archive']}.zip"
    if not target.exists():
        data=fetch(key)
        with zipfile.ZipFile(io.BytesIO(data)) as z:target.write_bytes(z.read(name))
    result=dict(video_id=target.stem,path=name,url=base+key,bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),reference=row)
    print('audio',target.stem,target.stat().st_size,flush=True);return result
results=list(ThreadPoolExecutor(6).map(one,rows));(W/'audio_manifest.json').write_text(json.dumps(results,indent=2,default=str))
