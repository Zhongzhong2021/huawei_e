"""Portable bundle checks; set Q3B_SPECIAL for attachment 4 replay."""
import csv
import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
NAMES = {"uncertainty": "uncertainty__macro_f1",
         "selfmm": "selfmm__macro_f1", "tetfn": "tetfn_source__macro_f1"}


@pytest.mark.parametrize("model", NAMES)
def test_bundle_has_local_config_and_strict_checkpoint(model):
    sys.path.insert(0, str(ROOT))
    from run import bundle, configuration
    from q3b.experiment import build_experiment_model
    import torch

    config = configuration(model)
    assert Path(config["model"]["pretrained_name"]) == ROOT / "assets/bert-base-uncased"
    model_instance = build_experiment_model(config)
    state = torch.load(bundle(model) / "best.pt", map_location="cpu", weights_only=True)
    model_instance.load_state_dict(state["model_state_dict"], strict=True)


def test_training_entry_passes_pretrained_data_and_epoch_cap(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT))
    import run

    pretrained = tmp_path / "bert"
    pretrained.mkdir()
    (pretrained / "config.json").write_text("{}")
    (pretrained / "model.safetensors").touch()
    aligned = tmp_path / "aligned.pkl"
    aligned.touch()
    captured = {}
    monkeypatch.setattr(run, "fit", lambda config, output: captured.update(config=config, output=output))
    run.train(Namespace(model="uncertainty", pretrained=pretrained, aligned=aligned,
                        output=tmp_path / "run", device="cpu", epochs=20))
    config = captured["config"]
    assert config["model"]["pretrained_name"] == str(pretrained)
    assert config["data"]["aligned_path"] == str(aligned)
    assert config["train"]["max_epochs"] == 20
    assert config["train"]["patience"] == 4


def test_prediction_rejects_reused_output_before_loading(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT))
    import run

    output = tmp_path / "prediction"
    output.mkdir()
    (output / "explanations.jsonl").write_text("stale")
    monkeypatch.setattr(run, "load_samples", lambda *args: pytest.fail("loaded data before output guard"))
    with pytest.raises(FileExistsError, match="empty directory"):
        run.predict(Namespace(model="selfmm", input=tmp_path, split="special",
                              output=output, device="cpu", bundle=None))
    assert (output / "explanations.jsonl").read_text() == "stale"


def test_bundle_rejects_wrong_model_family():
    sys.path.insert(0, str(ROOT))
    import run

    with pytest.raises(ValueError, match="does not match"):
        run.bundle("selfmm", ROOT / "weights/uncertainty")


def test_evaluate_with_embedded_or_joined_labels(tmp_path):
    sys.path.insert(0, str(ROOT))
    import run

    predictions = tmp_path / "predictions.csv"
    predictions.write_text("id,polarity,intensity,raw_intensity,true_intensity,true_class\n"
                           "01,2,0.5,0.6,1,2\n02,0,-0.5,-0.4,-1,0\n", encoding="utf-8-sig")
    output = tmp_path / "metrics.json"
    run.evaluate(Namespace(predictions=predictions, output=output, labels=None))
    embedded = json.loads(output.read_text())
    assert embedded["n"] == 2
    assert embedded["final"]["accuracy"] == 1
    assert embedded["raw_mae"] == pytest.approx(0.5)
    labels = tmp_path / "labels.csv"
    labels.write_text("sample_id,class_id,sentiment\n02,0,-1\n01,2,1\n", encoding="utf-8-sig")
    run.evaluate(Namespace(predictions=predictions, output=output, labels=labels))
    assert json.loads(output.read_text()) == embedded
    labels.write_text("sample_id,class_id,sentiment\n01,2,1\n01,2,1\n")
    with pytest.raises(ValueError, match="unique"):
        run.evaluate(Namespace(predictions=predictions, output=output, labels=labels))
    labels.write_text("sample_id,class_id,sentiment\n01,2,1\n03,0,-1\n")
    with pytest.raises(ValueError, match="match exactly"):
        run.evaluate(Namespace(predictions=predictions, output=output, labels=labels))


@pytest.mark.skipif(not os.getenv("Q3B_SPECIAL"), reason="set Q3B_SPECIAL for frozen replay")
@pytest.mark.parametrize("model,reference", NAMES.items())
def test_offline_special_matches_freeze(tmp_path, model, reference):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for 1e-6 frozen replay")
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    subprocess.run([sys.executable, str(ROOT / "run.py"), "predict", "--model", model,
                    "--input", os.environ["Q3B_SPECIAL"], "--split", "special",
                    "--output", str(tmp_path), "--device", "cuda:0"],
                   cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    with (tmp_path / "predictions.csv").open() as stream:
        current = list(csv.DictReader(stream))
    reference_path = ((Path(os.environ["Q3B_REFERENCE"]) / reference / "special_predictions.csv")
                      if os.getenv("Q3B_REFERENCE") else ROOT / "results" / model / "special_predictions.csv")
    with reference_path.open(encoding="utf-8-sig") as stream:
        frozen = list(csv.DictReader(stream))
    assert len(current) == len(frozen) == 20
    assert [row["id"] for row in current] == [row["id"] for row in frozen]
    for actual, expected in zip(current, frozen):
        assert actual["polarity"] == expected["polarity"]
        for key in ("intensity", "raw_intensity", "aux_logit_0", "aux_logit_1", "aux_logit_2"):
            assert abs(float(actual[key]) - float(expected[key])) <= 1e-6
    if model == "uncertainty":
        records = [json.loads(line) for line in (tmp_path / "explanations.jsonl").open()]
        assert len(records) == 20
        assert max(row["accounting_error"] for row in records) < 1e-4
    else:
        assert not (tmp_path / "explanations.jsonl").exists()
