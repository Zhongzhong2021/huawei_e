# 本程序及代码是在人工智能工具辅助下完成的。
# 工具：OpenAI Codex；模型：GPT-6 Astra (gpt-6-astra)；开发机构：OpenAI；模型发布日期：2026-09-03。
# AI 来源说明于 2026-09-24 补录，未改变算法；参见 AI辅助使用记录与队内审阅要求.md。
import json,hashlib
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
out=Path('results/q1_next');out.mkdir(parents=True,exist_ok=True)
repo='Systran/faster-whisper-small';revision='536b0662742c02347bc0e980a01041f333bce120'
print('Pinned revision',revision,flush=True)
path=snapshot_download(repo,revision=revision,allow_patterns=['model.bin','config.json','tokenizer.json','vocabulary.*','preprocessor_config.json'],max_workers=3)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
lock={'repo':repo,'revision':revision,'path':path,'files':{p.name:sha(p) for p in Path(path).iterdir() if p.is_file()}}
if (out/'asr_model.lock.json').exists():
    previous=json.loads((out/'asr_model.lock.json').read_text())
    assert previous['revision']==revision
    for name,digest in previous['files'].items():assert lock['files'][name]==digest, f'Changed cached model file: {name}'
(out/'asr_model.lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8');print(json.dumps(lock,indent=2))
