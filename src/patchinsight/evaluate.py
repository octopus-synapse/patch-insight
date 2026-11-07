"""One explicit final evaluation of all three methods on the same frozen test."""
import argparse
import json
from pathlib import Path

from . import LABELS, RULES_VERSION
from .data import load_split, sha256, verify_freeze
from .experiment import binary, embeddings, environment, save_json, targets
from .metrics import metrics
from .model import BertPredictor
from .rules import heuristic_scores


def main():
    import numpy as np
    from transformers import AutoModel, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--run", type=Path, default=Path("artifacts/run-42"))
    parser.add_argument("--output", type=Path, default=Path("reports/final"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--final", action="store_true", required=True,
                        help="Confirma que configuração e seleção foram encerradas")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("diretório final já existe; avaliação não será sobrescrita")
    predictor = BertPredictor(args.run / "model", device=args.device)
    frozen = json.loads((args.run / "frozen.json").read_text())
    manifest = verify_freeze(args.data)
    if (predictor.metadata["provenance"]["dataset"] != manifest
            or frozen["provenance"]["dataset"] != manifest):
        raise ValueError("artefatos foram treinados em outra versão dos dados")
    if frozen["labels"] != list(LABELS):
        raise ValueError("rótulos da regressão incompatíveis")
    config = frozen["experiment"]
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=config["base_revision"])
    encoder = AutoModel.from_pretrained(config["base_model"], revision=config["base_revision"]).to(args.device)
    rows = load_split(args.data, "test")
    texts = [r["text"] for r in rows]
    x = embeddings(texts, tokenizer, encoder, args.device, config["max_length"], config["batch_size"])
    x = (x - np.asarray(frozen["mean"])) / np.asarray(frozen["scale"])
    logits = x @ np.asarray(frozen["weights"]).T + np.asarray(frozen["bias"])
    frozen_scores = (1 / (1 + np.exp(-np.clip(logits, -80, 80)))).tolist()
    bert_scores = []
    for start in range(0, len(texts), config["batch_size"]):
        bert_scores.extend(predictor.score_batch(texts[start:start + config["batch_size"]]))
    scores = {
        "heuristic": [[heuristic_scores(t)[k] for k in LABELS] for t in texts],
        "frozen_logistic": frozen_scores,
        "bert_finetuned": bert_scores,
    }
    thresholds = {"heuristic": [.5] * 5, "frozen_logistic": config["thresholds"],
                  "bert_finetuned": predictor.metadata["thresholds"]}
    y = targets(rows)
    results = {name: metrics(y, binary(values, thresholds[name])) for name, values in scores.items()}
    # Exclusive creation only after all models completed successfully.
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "benchmark.json", {
        "reference": "synthetic_ai_reviewed", "split": "test", "dataset": manifest,
        "environment": environment(), "rules_version": RULES_VERSION,
        "model_manifest_sha256": sha256(args.run / "model" / "manifest.json"),
        "frozen_sha256": sha256(args.run / "frozen.json"), "thresholds": thresholds,
        "methods": results})
    records = [{"id": row["id"], "target": y[i], "scores": {n: s[i] for n, s in scores.items()}}
               for i, row in enumerate(rows)]
    (args.output / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    lines = ["# Benchmark final", "", "Referências sintéticas revisadas por IA; sem validação humana.",
             "", "| Método | Macro-F1 | Micro-F1 | Exact match |", "|---|---:|---:|---:|"]
    for name, result in results.items():
        lines.append(f"| {name} | {result['macro_f1']:.4f} | {result['micro_f1']:.4f} | {result['exact_match']:.4f} |")
    lines += ["", "Suporte e contagens TP/FP/FN/TN por rótulo: `benchmark.json`.",
              "", "Não usar estes erros para ampliar os dados ou ajustar a v0.1."]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
