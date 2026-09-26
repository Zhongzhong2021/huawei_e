# 本程序及代码是在人工智能工具辅助下完成的。
# 工具：OpenAI Codex；模型：GPT-6 Astra (gpt-6-astra)；开发机构：OpenAI；模型发布日期：2026-09-03。
# AI 来源说明于 2026-09-24 补录，未改变算法；参见 AI辅助使用记录与队内审阅要求.md。
"""Install the bundled, hash-locked generic YuNet model into the shared cache."""
import hashlib,json,shutil
from pathlib import Path

def main():
    source=Path('/opt/q1_refinement_models')
    if not source.exists():source=Path('models')
    lock=json.loads((source/'yunet.lock.json').read_text())
    model=source/'face_detection_yunet_2023mar.onnx'
    assert hashlib.sha256(model.read_bytes()).hexdigest()==lock['sha256']
    target=Path('/cache/q1');target.mkdir(parents=True,exist_ok=True)
    for name in ('face_detection_yunet_2023mar.onnx','yunet.lock.json','YuNet_LICENSE.txt'):
        shutil.copyfile(source/name,target/name)
    print(json.dumps(dict(model_bytes=model.stat().st_size,sha256=lock['sha256'],verified=True)))

if __name__=='__main__':main()
