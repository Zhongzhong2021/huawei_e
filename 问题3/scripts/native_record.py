"""Frozen q3b score decomposition."""
import numpy as np

def native_record(out, batch, i, policy, predicted_class):
    """Explain the score actually used by the policy; pair allocation is explicit."""
    if not {"single", "pair", "local", "modal", "bias", "q"} <= out.keys():
        return None
    scores = out["score"][i].detach().cpu().numpy()
    contrast = np.zeros(4, dtype=np.float32)
    if policy["decision_mode"] == "regression":
        contrast[0] = 1
        target, policy_bias = "regression_preactivation_q", 0.0
    else:
        logits = scores[1:].copy()
        logits[1] += policy["neutral_bias"]
        runner = max((c for c in range(3) if c != predicted_class), key=lambda c: logits[c])
        contrast[1 + predicted_class], contrast[1 + runner] = 1, -1
        policy_bias = policy["neutral_bias"] * ((predicted_class == 1) - (runner == 1))
        target = f"calibrated_logit_margin_{predicted_class}_vs_{runner}"
    single = out["single"][i].detach().cpu().numpy() @ contrast
    pair = out["pair"][i].detach().cpu().numpy() @ contrast
    local = out["local"][i].detach().cpu().numpy() @ contrast
    modal = out["modal"][i].detach().cpu().numpy() @ contrast
    bias = float(out["bias"].detach().cpu().numpy() @ contrast) + policy_bias
    score = float(scores @ contrast) + policy_bias
    magnitude = np.abs(modal)
    share = magnitude / magnitude.sum() if magnitude.sum() > 1e-6 else np.zeros(3)
    source = batch["source_index"][i].detach().cpu().numpy()
    principal = int(magnitude.argmax()) if magnitude.sum() > 1e-6 else None
    evidence = {}
    for m, modality in enumerate(("T", "A", "V")):
        valid = batch["observed"][modality][i].any(dim=-1).detach().cpu().numpy()
        top = sorted(np.flatnonzero(valid), key=lambda j: -abs(local[j, m]))[:3]
        evidence[modality] = [dict(window=int(j), source_indices=[int(s) for s in source[j] if s >= 0],
                                   signed_score=float(local[j, m])) for j in top]
    return dict(target=target, scope="contextual_readout_not_independent_input_causal_effect",
                interaction_allocation="each_pair_split_equally_between_its_two_modalities",
                score=score, bias=bias, accounting_error=abs(score - bias - float(modal.sum())),
                modal_scores=dict(zip(("T", "A", "V"), map(float, modal))),
                modality_shares=dict(zip(("T", "A", "V"), map(float, share))),
                principal_modality=("T", "A", "V")[principal] if principal is not None else None,
                single=single.tolist(), pair=pair.tolist(), local=local.tolist(),
                source_index=source.tolist(), top_evidence=evidence)
