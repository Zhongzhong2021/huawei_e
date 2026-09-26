import json,urllib.request,hashlib
from pathlib import Path
import os
root=Path(__file__).resolve().parents[1];out=Path(os.environ.get('Q1_PHONE_MODEL','/workspace/q1-phone-model'))
info=json.loads((root/'model_info.json').read_text());rows=[]
for name in ['config.json','preprocessor_config.json','pytorch_model.bin','special_tokens_map.json','tokenizer_config.json','vocab.json','README.md']:
 url=f"https://huggingface.co/{info['id']}/resolve/{info['sha']}/{name}"
 path=out/name
 if not path.exists():
  request=urllib.request.Request(url,headers={'User-Agent':'q1-reproducible-research'})
  with urllib.request.urlopen(request,timeout=120) as response, path.with_suffix(path.suffix+'.partial').open('wb') as stream:
   while chunk:=response.read(1024*1024):stream.write(chunk)
  path.with_suffix(path.suffix+'.partial').replace(path)
 rows.append({'path':name,'url':url,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()});print(name,path.stat().st_size,flush=True)
(out/'manifest.json').write_text(json.dumps({'repository':info['id'],'revision':info['sha'],'files':rows},indent=2)+'\n')
