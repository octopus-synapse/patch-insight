"""Shared local inference, loading the model once per process."""
import json
from pathlib import Path

from . import LABELS
from .data import sha256
from .rules import heuristic_scores, structural_rules
from .schema import Component, Input, Prediction, Versions


class Analyzer:
    def __init__(self, backend="bert", model="artifacts/model", device="cpu", policy=None):
        if backend not in ("bert", "heuristic"):
            raise ValueError("backend inválido")
        self.backend = backend
        self.policy = None
        if backend == "bert":
            from .model import BertPredictor
            self.predictor = BertPredictor(model, device=device)
        if policy:
            if backend != "bert":
                raise ValueError("política experimental exige backend bert")
            if self.predictor.metadata.get("calibration"):
                raise ValueError("bundle já contém calibração; política adicional recusada")
            from .assessment import temperature_scale
            candidate = json.loads(Path(policy).read_text())
            if candidate.get("method") != "bert_finetuned":
                raise ValueError("política pertence a outro método")
            if candidate["model_manifest_sha256"] != sha256(Path(model) / "manifest.json"):
                raise ValueError("política pertence a outro bundle")
            if candidate["labels"] != list(LABELS) or candidate["thresholds"] != [.5] * 5:
                raise ValueError("política incompatível")
            if self.predictor.metadata["thresholds"] != candidate["thresholds"]:
                raise ValueError("limiares do bundle incompatíveis com política")
            temperature_scale([[.5] * 5], candidate["temperature"])
            cutoff = candidate["confidence_cutoff"]
            if cutoff is not None and not .5 <= cutoff <= 1:
                raise ValueError("limiar de confiança inválido")
            self.policy = candidate

    def analyze_many(self, texts):
        texts = [Input(text=t).text for t in texts]
        if not texts:
            return []
        if self.backend == "bert":
            raw = self.predictor.predict_many(texts)
            version = self.predictor.version
        else:
            raw = []
            version = "heuristic-0.1.0"
            for text in texts:
                scores = heuristic_scores(text)
                raw.append((scores, {k: "present" if v else "absent" for k, v in scores.items()},
                            False, None))
        results = []
        for text, (scores, states, truncated, tokens) in zip(texts, raw):
            kind = "binary_rule" if self.backend == "heuristic" else "sigmoid_uncalibrated"
            if self.backend == "bert" and getattr(self.predictor, "metadata", {}).get("calibration"):
                kind = "temperature_scaled_experimental"
            uncertain = []
            if self.policy:
                from .assessment import temperature_scale
                kind = "temperature_scaled_experimental"
                for label in LABELS:
                    if scores[label] is None:
                        continue
                    value = float(temperature_scale([scores[label]], self.policy["temperature"])[0])
                    cutoff = self.policy["confidence_cutoff"]
                    if cutoff is not None and max(value, 1 - value) < cutoff:
                        states[label], scores[label] = "unknown", None
                        uncertain.append(f"review_{label}")
                    else:
                        scores[label] = value
            structures, task, techs, diagnostics = structural_rules(text, states)
            diagnostics += uncertain
            if self.policy:
                diagnostics.append("experimental_policy_" + self.policy["policy_id"])
                if self.policy["confidence_cutoff"] is None:
                    diagnostics.append("abstention_target_not_met_in_development")
            if truncated:
                diagnostics.append("input_truncated_unknown_negatives")
            results.append(Prediction(
                text=text, components={k: Component(state=states[k], score=scores[k]) for k in LABELS},
                structures=structures, task=task, technologies=techs, diagnostics=diagnostics,
                truncated=truncated, tokens=tokens, versions=Versions(model_version=version),
                backend=self.backend, score_kind=kind))
        return results

    def analyze(self, text):
        return self.analyze_many([text])[0]
