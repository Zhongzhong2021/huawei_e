import argparse
import json
from pathlib import Path
from .data import prepare
from .engine import train, evaluate, predict, naive_baselines, save_scenarios, load_data


def main():
    p = argparse.ArgumentParser(description="E question 2 reproducible pipeline")
    p.add_argument("--root", default=".")
    subs = p.add_subparsers(dest="command", required=True)
    audit = subs.add_parser("audit")
    audit.add_argument("--source", required=True)
    audit.add_argument("--special", required=True)
    training = subs.add_parser("train")
    training.add_argument("--config", required=True)
    training.add_argument("--run", required=True)
    evaluation = subs.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--split", choices=["valid", "test"], default="valid")
    evaluation.add_argument("--supplementary", action="store_true")
    evaluation.add_argument("--final", action="store_true")
    prediction = subs.add_parser("predict")
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--out", required=True)
    prediction.add_argument("--input-dir")
    subs.add_parser("report")
    a = p.parse_args()
    if a.command == "audit":
        result = prepare(a.root, a.source, a.special)
        save_scenarios(a.root, load_data(a.root, "valid"), "valid")
        naive_baselines(a.root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif a.command == "train":
        print(json.dumps(train(a.root, json.loads(Path(a.config).read_text()), a.run), indent=2))
    elif a.command == "evaluate":
        print(json.dumps(evaluate(a.root, a.checkpoint, a.split, a.supplementary, a.final), indent=2))
    elif a.command == "predict":
        print(json.dumps(predict(a.root, a.checkpoint, a.out, a.input_dir), indent=2))
    else:
        from .report import report
        report(a.root)


if __name__ == "__main__":
    main()
