import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def metrics(labels, targets, probabilities, predictions):
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    classes = probabilities.argmax(1)
    if not np.isfinite(probabilities).all() or not np.isfinite(predictions).all():
        raise ValueError("Nonfinite prediction")
    pearson = None
    reason = None
    if len(targets) < 2 or np.ptp(targets) == 0 or np.ptp(predictions) == 0:
        reason = "Pearson undefined: fewer than two samples or constant vector"
    else:
        pearson = float(np.corrcoef(targets, predictions)[0, 1])
    return {"n": len(labels), "accuracy": float(accuracy_score(labels, classes)),
            "macro_f1": float(f1_score(labels, classes, labels=[0, 1, 2], average="macro", zero_division=0)),
            "class_f1": f1_score(labels, classes, labels=[0, 1, 2], average=None, zero_division=0).tolist(),
            "mae": float(np.mean(np.abs(targets - predictions))), "pearson": pearson,
            "pearson_note": reason, "confusion_matrix": confusion_matrix(labels, classes, labels=[0, 1, 2]).tolist()}


def selection_score(results, scenarios):
    f1 = float(np.mean([results[s]["macro_f1"] for s in scenarios]))
    mae = float(np.mean([results[s]["mae"] for s in scenarios]))
    return {"mean_missing_macro_f1": f1, "mean_missing_mae": mae,
            "score": .5 * f1 + .5 * (1 - mae / 6)}
