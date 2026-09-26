# 本程序及代码是在人工智能工具辅助下完成的。
# 工具：OpenAI Codex；模型：GPT-6 Astra (gpt-6-astra)；开发机构：OpenAI；模型发布日期：2026-09-03。
# AI 来源说明于 2026-09-24 补录，未改变算法；参见 AI辅助使用记录与队内审阅要求.md。
"""Download immutable generic model assets and verify known hashes when available."""
import hashlib,json,os
from pathlib import Path
import requests,torchaudio
from huggingface_hub import snapshot_download

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

os.environ.setdefault('TORCH_HOME','/cache/torch')
model_id='google-bert/bert-base-uncased';revision='86b5e0934494bd15c9632b12f734a8a67f723594'
files=['config.json','tokenizer.json','tokenizer_config.json','vocab.txt','model.safetensors']
snapshot=snapshot_download(model_id,revision=revision,allow_patterns=files,max_workers=3)
lock={'model_id':model_id,'revision':revision,'local_snapshot':snapshot,
      'files_sha256':{name:sha(Path(snapshot)/name) for name in files}}
out=Path('results/text_encoder');out.mkdir(parents=True,exist_ok=True)
existing=out/'encoder.lock.json'
if existing.exists():
    old=json.loads(existing.read_text())
    for name,h in old['files_sha256'].items():assert sha(Path(snapshot)/name)==h
else:existing.write_text(json.dumps(lock,indent=2),encoding='utf-8')
face=Path('/cache/q1/face_landmarker.task');face.parent.mkdir(parents=True,exist_ok=True)
if not face.exists():
    r=requests.get('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',timeout=120)
    r.raise_for_status();tmp=face.with_suffix('.download');tmp.write_bytes(r.content);tmp.replace(face)
torchaudio.pipelines.MMS_FA.get_model(with_star=False)
from torchvision.models import resnet18,ResNet18_Weights
resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
config=Path('results/q1/run_config.json')
if config.exists():
    c=json.loads(config.read_text())
    assert sha(face)==c['models']['face_landmarker']['sha256']
    assert sha('/cache/torch/hub/checkpoints/model.pt')==c['models']['mms_fa']['sha256']
scene_config=Path('results/q1/scene_config.json')
if scene_config.exists():
    c=json.loads(scene_config.read_text())
    assert sha('/cache/torch/hub/checkpoints/resnet18-f37072fd.pth')==c['weight_sha256']
print('Generic model cache prepared and verified.')
