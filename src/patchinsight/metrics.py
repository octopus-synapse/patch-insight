"""Metrics without runtime dependencies; same implementation for every method."""
from . import LABELS


def metrics(targets, predictions):
    if not targets or len(targets) != len(predictions):
        raise ValueError("targets e predictions devem ter mesmo tamanho não nulo")
    if any(len(row) != len(LABELS) or any(v not in (0, 1) for v in row)
           for row in [*targets, *predictions]):
        raise ValueError("matrizes binárias com cinco colunas são obrigatórias")
    per_label = {}
    totals = [0, 0, 0]
    for i, label in enumerate(LABELS):
        tp = sum(y[i] == p[i] == 1 for y, p in zip(targets, predictions))
        fp = sum(y[i] == 0 and p[i] == 1 for y, p in zip(targets, predictions))
        fn = sum(y[i] == 1 and p[i] == 0 for y, p in zip(targets, predictions))
        tn = len(targets) - tp - fp - fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_label[label] = dict(precision=precision, recall=recall, f1=f1,
                                support=tp + fn, tp=tp, fp=fp, fn=fn, tn=tn)
        totals = [a + b for a, b in zip(totals, (tp, fp, fn))]
    tp, fp, fn = totals
    return {
        "n": len(targets),
        "macro_f1": sum(v["f1"] for v in per_label.values()) / len(LABELS),
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "exact_match": sum(list(y) == list(p) for y, p in zip(targets, predictions)) / len(targets),
        "per_label": per_label,
        "zero_division": 0,
    }
