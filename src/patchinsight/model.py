"""Local-only inference from an integrity-checked, fine-tuned bundle."""
import json
from pathlib import Path

from . import LABELS
from .data import sha256
from .text import normalize


def write_manifest(directory):
    directory = Path(directory)
    files = {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob("*"))
             if p.is_file() and p.name != "manifest.json"}
    (directory / "manifest.json").write_text(json.dumps({"sha256": files}, indent=2) + "\n")


def verify_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for filename, digest in manifest["sha256"].items():
        path = (directory / filename).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path) != digest:
            raise ValueError(f"artefato inválido: {filename}")
    if not {"patchinsight.json", "config.json", "model.safetensors"} <= set(manifest["sha256"]):
        raise ValueError("bundle incompleto")
    metadata = json.loads((directory / "patchinsight.json").read_text())
    if metadata["labels"] != list(LABELS) or metadata["trained"] is not True:
        raise ValueError("checkpoint não identifica treinamento e rótulos esperados")
    thresholds = metadata["thresholds"]
    if len(thresholds) != 5 or any(not 0 < t < 1 for t in thresholds):
        raise ValueError("thresholds inválidos")
    if not 4 <= metadata["max_length"] <= 512:
        raise ValueError("max_length inválido")
    return metadata


class BertPredictor:
    def __init__(self, directory, device="cpu"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.metadata = verify_bundle(directory)
        self.version = self.metadata["model_version"]
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            directory, local_files_only=True, use_safetensors=True).to(self.device).eval()
        if self.model.config.num_labels != len(LABELS) or [
            self.model.config.id2label[i] for i in range(len(LABELS))
        ] != list(LABELS):
            raise ValueError("cabeça do modelo incompatível")

    def score_batch(self, texts):
        import torch
        texts = [normalize(t) for t in texts]
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                                max_length=self.metadata["max_length"], return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(**inputs.to(self.device)).logits
        if not torch.isfinite(logits).all():
            raise ValueError("modelo produziu logits não finitos")
        return logits.sigmoid().cpu().tolist()

    def predict(self, text):
        return self.predict_many([text])[0]

    def predict_many(self, texts):
        if not texts:
            return []
        texts = [normalize(t) for t in texts]
        counts = [len(ids) for ids in self.tokenizer(texts, truncation=False)["input_ids"]]
        results = []
        for probabilities, token_count in zip(self.score_batch(texts), counts):
            truncated = token_count > self.metadata["max_length"]
            states, scores = {}, {}
            for label, prob, threshold in zip(LABELS, probabilities, self.metadata["thresholds"]):
                state = "present" if prob >= threshold else ("unknown" if truncated else "absent")
                states[label] = state
                scores[label] = None if state == "unknown" else prob
            results.append((scores, states, truncated, token_count))
        return results
