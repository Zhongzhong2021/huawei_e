import argparse
import importlib.util
from pathlib import Path
import json
import subprocess
import sys

def script(root,name):
    # Study helpers may import sibling helper modules. Preserve the process path
    # after execution while making the selected artifact's scripts authoritative.
    spec=importlib.util.spec_from_file_location(name,root/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(root/'scripts'))
    try:spec.loader.exec_module(module)
    finally:sys.path.pop(0)
    return module

def study_round(root):
    for relative in ['study/protocol.json', 'configs/frozen_protocol.json']:
        path=root/relative
        if path.is_file():
            version=json.loads(path.read_text(encoding='utf-8')).get('version','')
            if version.startswith('round5-'):return 5
            if version.startswith('round4-'):return 4
            return 3
    return 3

def is_round4(root):return study_round(root)==4

def main():
    parser=argparse.ArgumentParser(description='Q2 frozen-model research and paper evidence')
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ['audit','analyze','figures','paper','report']:sub.add_parser(command)
    evaluate=sub.add_parser('evaluate')
    evaluate.add_argument('--data-root',type=Path)
    evaluate.add_argument('--output-dir',type=Path)
    evaluate.add_argument('--split',choices=['valid','test'],default='valid')
    evaluate.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    train=sub.add_parser('train');train.add_argument('--name',default='reproduction');train.add_argument('--seed',type=int,default=42)
    train.add_argument('--data-root',type=Path);train.add_argument('--pretrained',type=Path)
    train.add_argument('--checkpoint',type=Path,help='Frozen checkpoint; default: <root>/final/model.pt')
    train.add_argument('--protocol',type=Path,help='Training protocol; defaults to study/protocol.json or configs/frozen_protocol.json')
    train.add_argument('--dry-run',action='store_true',help='Check inputs and print the fixed recipe without fitting or creating a run')
    predict=sub.add_parser('predict');predict.add_argument('--input-dir',type=Path,required=True)
    predict.add_argument('--output',type=Path,required=True);predict.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args();root=args.root.resolve()
    if args.command=='audit':
        return subprocess.run([sys.executable,'-m','unittest','discover','-s',str(root/'tests'),'-v'],check=True)
    if args.command=='predict':
        from q2v2.deploy import predict
        print(predict(root/'final/model.pt',root/'final/scaler.npz',args.input_dir,args.output,args.device))
    elif args.command=='train':
        from .reproduction import preflight
        plan=preflight(root,args.name,args.seed,args.data_root,args.pretrained,args.checkpoint,args.protocol)
        import torch
        from q2.data import load_data
        from q2v2.engine import train_run
        record=torch.load(plan['inputs']['checkpoint'],map_location='cpu',weights_only=True)
        config={**record['config'],'seed':args.seed}
        if config.get('distillation'):
            raise ValueError('Fallback student reproduction uses preserved round2 recipe; teacher checkpoint required')
        epoch=record.get('epoch')
        if not isinstance(epoch,int) or isinstance(epoch,bool) or epoch<1:
            raise ValueError('Frozen checkpoint must contain a positive integer epoch')
        if args.dry_run:
            print(json.dumps({**plan,'fixed_epochs':epoch,'model_config':config,'training_started':False},ensure_ascii=False,indent=2))
            return
        # Reproduction is a new fixed-epoch run, never model selection. It is allowed
        # after the optimization cutoff, with the frozen config/epochs unchanged.
        print(train_run(root,args.name,config,load_data(plan['data_root'],'train'),load_data(plan['data_root'],'valid'),
            record['vocabulary'].numpy(),source=torch.load(plan['inputs']['pretrained'],map_location='cpu',weights_only=True),
            fixed_epochs=epoch,reproduction=True,training_protocol=plan['training_protocol']))
    elif args.command=='evaluate':
        if study_round(root)>=4:
            from .evaluation import evaluate_frozen
            print(evaluate_frozen(root,args.data_root,args.output_dir,args.split,args.device))
        elif args.data_root is not None or args.output_dir is not None or args.split!='valid' or args.device!='cpu':
            raise ValueError('Explicit evaluation options are supported by round4; legacy round3 uses its archived evaluation recipe')
        else:script(root,'evaluate_frozen').main()
    else:
        if args.command in ['analyze','report']:
            if study_round(root)==5:script(root,'analyze_round5').recompute_saved(root)
            else:
                from .analysis import main as analyze
                analyze(root)
                if is_round4(root):script(root,'analyze_round4_tradeoffs').main(root)
        if args.command in ['figures','report']:
            if study_round(root)==5:script(root,'plot_round5').main(root)
            elif is_round4(root):script(root,'plot_round4').main(root)
            else:
                from .figures import main as figures
                figures(root)
        if args.command in ['paper','report']:script(root,{3:'build_paper',4:'paper_round4',5:'paper_round5'}[study_round(root)]).main(root)

if __name__=='__main__':main()
