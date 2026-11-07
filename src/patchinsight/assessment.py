"""Extended multilabel evaluation; never changes the frozen v0.1 benchmark."""
import math

import numpy as np
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

from . import LABELS
from .metrics import metrics


def arrays(targets, scores):
    y, p = np.asarray(targets), np.asarray(scores, dtype=float)
    if (y.ndim != 2 or y.shape != p.shape or y.shape[1] != len(LABELS)
            or not len(y) or not np.isin(y, [0, 1]).all()
            or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()):
        raise ValueError("targets binários e scores finitos [0,1] devem ter formato (n,5)")
    return y, p


def temperature_scale(scores, temperature):
    if not math.isfinite(temperature) or not .1 <= temperature <= 10:
        raise ValueError("temperatura deve estar entre 0.1 e 10")
    p = np.clip(np.asarray(scores, dtype=float), 1e-7, 1 - 1e-7)
    logits = np.log(p / (1 - p)) / temperature
    return 1 / (1 + np.exp(-logits))


def binary_log_loss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def fit_policy(targets, scores, max_error=.05, min_coverage=.5):
    """One global temperature and decision-confidence cutoff, development only."""
    y, p = arrays(targets, scores)
    candidates = sorted(set(np.geomspace(.25, 4, 81).tolist() + [1.0]))
    temperature = min(candidates, key=lambda t: (
        binary_log_loss(y, temperature_scale(p, t)), abs(t - 1)))
    calibrated = temperature_scale(p, temperature)
    confidence = np.maximum(calibrated, 1 - calibrated)
    wrong = (calibrated >= .5) != y
    cutoff = None
    for value in [0.5, .6, .7, .8, .85, .9, .95, .975, .99]:
        accepted = confidence >= value
        if accepted.mean() >= min_coverage and wrong[accepted].mean() <= max_error:
            cutoff = value
            break
    return {"temperature": float(temperature), "confidence_cutoff": cutoff,
            "target_development_error": max_error, "minimum_development_coverage": min_coverage,
            "unit": "component_decision", "thresholds": [.5] * len(LABELS),
            "warning": "Exploratório; erro observado em desenvolvimento não é garantia de risco."}


def selective_metrics(targets, scores, cutoff):
    y, p = arrays(targets, scores)
    accepted = np.maximum(p, 1 - p) >= cutoff
    n = int(accepted.sum())
    wrong = (p >= .5) != y
    return {"confidence_cutoff": cutoff, "accepted_decisions": n,
            "coverage": float(accepted.mean()),
            "error_on_accepted": float(wrong[accepted].mean()) if n else None,
            "complete_text_coverage": float(accepted.all(axis=1).mean())}


def extended_metrics(targets, scores, thresholds=None):
    y, p = arrays(targets, scores)
    thresholds = np.asarray([.5] * len(LABELS) if thresholds is None else thresholds, dtype=float)
    if thresholds.shape != (len(LABELS),) or not np.isfinite(thresholds).all() or ((thresholds <= 0) | (thresholds >= 1)).any():
        raise ValueError("cinco limiares finitos entre 0 e 1 são obrigatórios")
    result = metrics(y.tolist(), (p >= thresholds).astype(int).tolist())
    result["hamming_loss"] = float(np.mean((p >= thresholds) != y))
    result["mean_binary_brier"] = float(np.mean((p - y) ** 2))
    result["mean_binary_log_loss"] = binary_log_loss(y, p)
    aps, aucs, mccs = [], [], []
    for i, label in enumerate(LABELS):
        ap = float(average_precision_score(y[:, i], p[:, i])) if y[:, i].sum() else None
        auc = float(roc_auc_score(y[:, i], p[:, i])) if len(set(y[:, i])) == 2 else None
        predicted = p[:, i] >= thresholds[i]
        mcc = (float(matthews_corrcoef(y[:, i], predicted))
               if len(set(y[:, i])) == 2 and len(set(predicted)) == 2 else 0.0)
        bins = []
        for index in range(5):
            mask = np.minimum((p[:, i] * 5).astype(int), 4) == index
            bins.append({"lower": index / 5, "upper": (index + 1) / 5,
                         "n": int(mask.sum()),
                         "mean_score": float(p[mask, i].mean()) if mask.any() else None,
                         "positive_rate": float(y[mask, i].mean()) if mask.any() else None})
        result["per_label"][label].update(
            average_precision=ap, roc_auc=auc, mcc=mcc,
            brier=float(np.mean((p[:, i] - y[:, i]) ** 2)),
            log_loss=binary_log_loss(y[:, i], p[:, i]), reliability=bins)
        aps.append(ap)
        aucs.append(auc)
        mccs.append(mcc)
    # Undefined labels are explicit; no optimistic averaging over a subset.
    result["macro_average_precision"] = float(np.mean(aps)) if None not in aps else None
    result["macro_roc_auc"] = float(np.mean(aucs)) if None not in aucs else None
    result["macro_mcc"] = float(np.mean(mccs))
    result["risk_coverage"] = [selective_metrics(y, p, c) for c in
                               [.5, .6, .7, .8, .85, .9, .95, .975, .99]]
    if not np.all(thresholds == .5):
        result.pop("risk_coverage")
        result["thresholds"] = thresholds.tolist()
    result["probability_note"] = "Brier/log loss combinam discriminação e calibração."
    result["log_loss_clip"] = 1e-7
    return result


def paired_group_bootstrap(targets, first, second, groups, repeats=1000):
    """CI for first minus second; re-sample complete origins, not decisions."""
    y, a = arrays(targets, first)
    _, b = arrays(targets, second)
    if len(groups) != len(y):
        raise ValueError("um grupo por texto é obrigatório")
    unique = sorted(set(groups))
    members = {g: np.flatnonzero(np.asarray(groups) == g) for g in unique}
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(repeats):
        indices = np.concatenate([members[g] for g in rng.choice(unique, len(unique), replace=True)])
        left = metrics(y[indices].tolist(), (a[indices] >= .5).astype(int).tolist())
        right = metrics(y[indices].tolist(), (b[indices] >= .5).astype(int).tolist())
        differences.append(left["macro_f1"] - right["macro_f1"])
    return {"metric": "macro_f1_first_minus_second", "groups": len(unique),
            "repeats": repeats, "seed": 42, "percentile_95": np.quantile(
                differences, [.025, .975]).tolist(),
            "scope": "amostra sintética; não estima generalização populacional"}
