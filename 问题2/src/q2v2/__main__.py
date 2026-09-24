"""One interface for audit, fixed-recipe training, evaluation and prediction."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from q2.data import prepare, load_data, save_json
from q2.engine import evaluate_model
from q2.missing import MAIN, SUPPLEMENT
from .deploy import load_deployment, predict
from .engine import train_run


parser = argparse.ArgumentParser()
parser.add_argument("--root",type=Path,default=Path("round2"))
sub = parser.add_subparsers(dest="command",required=True)
audit = sub.add_parser("audit")
audit.add_argument("--source",required=True)
audit.add_argument("--special",required=True)
train = sub.add_parser("train")
train.add_argument("--name",default="user_retrain")
train.add_argument("--seed",type=int,default=42)
evaluate = sub.add_parser("evaluate")
evaluate.add_argument("--checkpoint",required=True)
evaluate.add_argument("--split",choices=["valid","test"],default="valid")
evaluate.add_argument("--final",action="store_true")
evaluate.add_argument("--supplementary",action="store_true")
prediction = sub.add_parser("predict")
prediction.add_argument("--checkpoint")
prediction.add_argument("--scaler")
prediction.add_argument("--input-dir",required=True)
prediction.add_argument("--output",required=True)
prediction.add_argument("--device",choices=["cpu","cuda"],default="cpu")
sub.add_parser("report")
args = parser.parse_args()
root = args.root.resolve()
if args.command == "audit":
    result = prepare(root.parent,args.source,args.special)
    print(json.dumps({"classes":result["classes"],"scaler_fit_split":result["scaler_fit_split"]}))
elif args.command == "predict":
    print(predict(args.checkpoint or root/"final/model.pt",args.scaler or root/"final/scaler.npz",
                  args.input_dir,args.output,args.device))
elif args.command == "train":
    recipe = json.loads((root/"study/training_recipe.json").read_text())
    training,valid = load_data(root.parent,"train"),load_data(root.parent,"valid")
    vocabulary = np.unique(np.concatenate([training["tokens"].ravel(),[0,100,101,102,103]]))
    teacher_path = None
    if recipe["student"]["config"]["distillation"]:
        teacher = recipe["teacher"]
        teacher_name = args.name+"_teacher"
        train_run(root,teacher_name,teacher["config"],training,valid,vocabulary,fixed_epochs=teacher["fixed_epochs"])
        teacher_path = root/"runs"/teacher_name/"best.pt"
    config = {**recipe["student"]["config"],"seed":args.seed}
    result = train_run(root,args.name,config,training,valid,vocabulary,teacher_path=teacher_path,
                       fixed_epochs=recipe["student"]["fixed_epochs"])
    print(json.dumps(result["selection"]))
elif args.command == "evaluate":
    if args.split == "test" and (not args.final or not (root/"study/frozen.json").exists()):
        raise ValueError("Test requires frozen.json and --final; it is descriptive, previously seen in round1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model,_ = load_deployment(args.checkpoint,device)
    result = evaluate_model(model,load_data(root.parent,args.split),device,
                            ["clean"]+MAIN+(SUPPLEMENT if args.supplementary else []),
                            out=root/"evaluation"/args.split,batch_size=64)
    save_json(root/"evaluation"/f"metrics_{args.split}.json",result)
    print(json.dumps(result["selection"]))
else:
    import runpy
    runpy.run_path(str(root/"scripts/build_report.py"),run_name="__main__")
