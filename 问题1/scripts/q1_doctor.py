# 本程序及代码是在人工智能工具辅助下完成的。
# 工具：OpenAI Codex；模型：GPT-6 Astra (gpt-6-astra)；开发机构：OpenAI；模型发布日期：2026-09-03。
# AI 来源说明于 2026-09-24 补录，未改变算法；参见 AI辅助使用记录与队内审阅要求.md。
import json,importlib.metadata,subprocess
from pathlib import Path
import torch,mediapipe,cv2
packages=['torch','torchvision','torchaudio','numpy','librosa','mediapipe','opencv-contrib-python','transformers']
result={'versions':{p:importlib.metadata.version(p) for p in packages},'cuda_available':torch.cuda.is_available(),
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
subprocess.run(['python','-m','pip','check'],check=True)
out=Path('results/q1');out.mkdir(parents=True,exist_ok=True)
(out/'environment.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
