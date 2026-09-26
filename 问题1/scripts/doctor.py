# 本程序及代码是在人工智能工具辅助下完成的。
# 工具：OpenAI Codex；模型：GPT-6 Astra (gpt-6-astra)；开发机构：OpenAI；模型发布日期：2026-09-03。
# AI 来源说明于 2026-09-24 补录，未改变算法；参见 AI辅助使用记录与队内审阅要求.md。
import argparse, json, platform, subprocess, importlib.metadata
from pathlib import Path
import torch

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
    packages=['torch','numpy','pandas','scipy','scikit-learn','transformers','librosa','opencv-python-headless','openpyxl']
    result={'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in packages},
            'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available()}
    if torch.cuda.is_available():
        result['gpu']=torch.cuda.get_device_name(0)
        torch.manual_seed(2026)
        model=torch.nn.Sequential(torch.nn.Linear(877,128),torch.nn.GELU(),torch.nn.Linear(128,4)).cuda()
        x=torch.randn(32,877,device='cuda'); y=model(x)
        y.square().mean().backward();torch.cuda.synchronize()
        result['gpu_forward_backward']=bool(torch.isfinite(y).all())
        assert result['gpu_forward_backward']
    result['ffmpeg']=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]
    subprocess.run(['python','-m','pip','check'],check=True)
    if args.output:
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
