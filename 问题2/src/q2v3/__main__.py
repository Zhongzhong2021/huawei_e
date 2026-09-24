import argparse
import importlib.util
from pathlib import Path
import json
import subprocess
import sys

def script(root,name):
    spec=importlib.util.spec_from_file_location(name,root/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def main():
    parser=argparse.ArgumentParser(description='Q2 round3 frozen-model research and paper evidence')
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ['audit','evaluate','analyze','figures','paper','report']:sub.add_parser(command)
    train=sub.add_parser('train');train.add_argument('--name',default='reproduction');train.add_argument('--seed',type=int,default=42)
    train.add_argument('--data-root',type=Path);train.add_argument('--pretrained',type=Path)
    predict=sub.add_parser('predict');predict.add_argument('--input-dir',type=Path,required=True)
    predict.add_argument('--output',type=Path,required=True);predict.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args();root=args.root.resolve()
    if args.command=='audit':
        return subprocess.run([sys.executable,'-m','unittest','discover','-s',str(root/'tests'),'-v'],check=True)
    if args.command=='predict':
        from q2v2.deploy import predict
        print(predict(root/'final/model.pt',root/'final/scaler.npz',args.input_dir,args.output,args.device))
    elif args.command=='train':
        if Path(args.name).name!=args.name or args.name in ['.','..']:
            raise ValueError('Run name must be a single directory name')
        if (root/'runs'/args.name).exists():
            raise FileExistsError('Run directories are immutable; select a fresh --name')
        import torch,numpy as np
        from q2.data import load_data
        from q2v2.engine import train_run
        record=torch.load(root/'final/model.pt',map_location='cpu',weights_only=True)
        data_root=args.data_root or root.parent
        path=args.pretrained or root.parent/'round2/pretrained/bert_base_uncased.pt'
        config={**record['config'],'seed':args.seed}
        if config.get('distillation'):
            raise ValueError('Fallback student reproduction uses preserved round2 recipe; teacher checkpoint required')
        # Reproduction is a new fixed-epoch run, never model selection. It is allowed
        # after the optimization cutoff, with the frozen config/epochs unchanged.
        print(train_run(root,args.name,config,load_data(data_root,'train'),load_data(data_root,'valid'),
            record['vocabulary'].numpy(),source=torch.load(path,map_location='cpu',weights_only=True),
            fixed_epochs=record['epoch'],reproduction=True))
    elif args.command=='evaluate':script(root,'evaluate_frozen').main()
    else:
        if args.command in ['analyze','report']:
            from .analysis import main as analyze
            analyze(root)
        if args.command in ['figures','report']:
            from .figures import main as figures
            figures(root)
        if args.command in ['paper','report']:script(root,'build_paper').main(root)

if __name__=='__main__':main()
