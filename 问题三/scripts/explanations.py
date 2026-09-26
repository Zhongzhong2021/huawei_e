"""Frozen q3b native explanation routines."""
import numpy as np

from .native_record import native_record

def explanation_record(out, batch, i, policy, predicted_class, scope):
    if "ordinal_score" in out and policy["decision_mode"] == "classification":
        # Ordinal log probabilities are nonlinear: decompose only their shared s.
        transformed = dict(out)
        for key in ("score", "single", "pair", "local", "modal", "bias"):
            transformed[key] = out[key].clone()
            transformed[key][..., 0] = out[key][..., 3] - out[key][..., 1]
        transformed["q"] = out["ordinal_score"]
        result = native_record(transformed, batch, i, {"decision_mode": "regression"}, predicted_class)
        result["target"] = "ordinal_shared_score_s_not_probability_or_logit_margin"
        logits = out["aux_logits"][i].detach().clone()
        logits[1] += policy["neutral_bias"]
        competitors = [c for c in range(3) if c != predicted_class]
        result.update(ordinal_thresholds=out["ordinal_thresholds"].detach().cpu().tolist(),
                      ordinal_probabilities=out["aux_logits"][i].detach().exp().cpu().tolist(),
                      policy_neutral_bias=policy["neutral_bias"],
                      actual_policy_logit_margin=float(logits[predicted_class] - logits[competitors].max()),
                      probability_decomposition_claimed=False)
    else:
        result = native_record(out, batch, i, policy, predicted_class)
    if result is None:
        return None
    result["scope"] = scope
    if "precision_weights" in out:
        result.update(precision_weights=out["precision_weights"][i].detach().cpu().tolist(),
                      unimodal_residual_variances=out["variance"][i].detach().cpu().tolist(),
                      precision_is_contribution=False, variance_is_fused_predictive_interval=False)
    if "coefficients" in out:
        result.update(prototype_activations=out["concepts"][i].detach().cpu().tolist(),
                      coefficients=out["coefficients"][i].detach().cpu().tolist(),
                      coefficients_are_causal_contributions=False)
    if result["accounting_error"] > 1e-4:
        raise AssertionError("score accounting failed")
    return result
