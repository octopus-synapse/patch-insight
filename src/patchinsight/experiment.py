"""Training and baselines. This module never opens the test split."""
import argparse
import importlib.metadata
import json
import platform
import random
import shutil
from pathlib import Path

from . import LABELS, __version__
from .data import load_split, sha256, verify_freeze
from .metrics import metrics
from .rules import heuristic_scores


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def targets(rows):
    return [[int(row["labels"][k]) for k in LABELS] for row in rows]


def binary(scores, thresholds):
    return [[int(v >= t) for v, t in zip(row, thresholds)] for row in scores]


def environment():
    packages = {}
    for name in ("torch", "transformers", "numpy", "scikit-learn", "pydantic"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    code = Path(__file__).resolve().parent
    sources = {p.name: sha256(p) for p in sorted(code.glob("*.py"))}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "source_sha256": sources}


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(2)


def tokenized_dataset(rows, tokenizer, max_length):
    import torch
    encoded = tokenizer([r["text"] for r in rows], padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")
    labels = torch.tensor(targets(rows), dtype=torch.float32)

    class Rows(torch.utils.data.Dataset):
        def __len__(self):
            return len(labels)

        def __getitem__(self, index):
            return {**{k: v[index] for k, v in encoded.items()}, "labels": labels[index]}

    return Rows()


def fine_tune(config, train, validation, output, device, provenance):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from .model import write_manifest

    seed_all(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=config["base_revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["base_model"], revision=config["base_revision"], num_labels=len(LABELS),
        problem_type="multi_label_classification", id2label=dict(enumerate(LABELS)),
        label2id={k: i for i, k in enumerate(LABELS)}).to(device)
    train_loader = torch.utils.data.DataLoader(
        tokenized_dataset(train, tokenizer, config["max_length"]),
        batch_size=config["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]))
    val_loader = torch.utils.data.DataLoader(
        tokenized_dataset(validation, tokenizer, config["max_length"]),
        batch_size=config["batch_size"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)
    best, history, best_epoch = -1.0, [], 0
    checkpoint = output / "model"
    for epoch in range(config["epochs"]):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        accumulation = config["gradient_accumulation"]
        for index, batch in enumerate(train_loader):
            # Normalize the last partial accumulation window correctly.
            window_start = (index // accumulation) * accumulation
            window_size = min(accumulation, len(train_loader) - window_start)
            loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
            total_loss += loss.item()
            (loss / window_size).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        model.eval()
        scores = []
        with torch.inference_mode():
            for batch in val_loader:
                scores.extend(model(**{k: v.to(device) for k, v in batch.items()
                                       if k != "labels"}).logits.sigmoid().cpu().tolist())
        report = metrics(targets(validation), binary(scores, config["thresholds"]))
        history.append({"epoch": epoch + 1, "train_loss": total_loss / len(train_loader),
                        "validation": report})
        print(json.dumps(history[-1]), flush=True)
        if report["macro_f1"] > best:
            best, best_epoch = report["macro_f1"], epoch + 1
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
    metadata = {"trained": True, "labels": list(LABELS), "model_version": __version__,
                "max_length": config["max_length"], "thresholds": config["thresholds"],
                "selected_epoch": best_epoch, "validation_macro_f1": best,
                "experiment": config, "provenance": provenance, "environment": environment()}
    save_json(checkpoint / "patchinsight.json", metadata)
    save_json(output / "training.json", {"history": history, "selected_epoch": best_epoch,
                                        "selection": "validation_macro_f1; earliest tie"})
    root = Path(__file__).resolve().parents[2]
    for filename in ("LICENSE", "licenses/BERTIMBAU-MIT.txt", "docs/MODEL_CARD.md"):
        source = root / filename
        if source.exists():
            shutil.copy(source, checkpoint / ("README.md" if "MODEL_CARD" in filename else source.name))
    write_manifest(checkpoint)


def embeddings(texts, tokenizer, encoder, device, max_length, batch_size):
    import numpy as np
    import torch
    vectors = []
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt").to(device)
            hidden = encoder(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            vectors.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy())
    return np.concatenate(vectors)


def frozen_baseline(config, train, validation, output, device, provenance):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoModel, AutoTokenizer

    seed_all(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=config["base_revision"])
    encoder = AutoModel.from_pretrained(config["base_model"], revision=config["base_revision"]).to(device)
    encoder.requires_grad_(False)
    x = embeddings([r["text"] for r in train], tokenizer, encoder, device,
                   config["max_length"], config["batch_size"])
    val_x = embeddings([r["text"] for r in validation], tokenizer, encoder, device,
                       config["max_length"], config["batch_size"])
    scaler = StandardScaler().fit(x)
    x, val_x = scaler.transform(x), scaler.transform(val_x)
    y = np.asarray(targets(train))
    weights, biases, scores = [], [], []
    for i in range(len(LABELS)):
        reg = LogisticRegression(C=config["logistic_C"], max_iter=config["logistic_max_iter"],
                                 random_state=config["seed"]).fit(x, y[:, i])
        weights.append(reg.coef_[0].tolist())
        biases.append(float(reg.intercept_[0]))
        scores.append(reg.predict_proba(val_x)[:, 1])
    payload = {"labels": list(LABELS), "weights": weights, "bias": biases,
               "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
               "experiment": config, "provenance": provenance, "environment": environment()}
    save_json(output / "frozen.json", payload)
    save_json(output / "frozen-validation.json", metrics(targets(validation), binary(
        np.asarray(scores).T.tolist(), config["thresholds"])))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["heuristic", "frozen", "finetune"])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/run-42"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if not (1 <= config["epochs"] <= 5 and config["batch_size"] > 0
            and config["gradient_accumulation"] > 0):
        raise ValueError("configuração de treino inválida")
    train, validation = load_split(args.data, "train"), load_split(args.data, "validation")
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {"dataset": verify_freeze(args.data), "config_sha256": sha256(args.config)}
    if args.method == "heuristic":
        report = metrics(targets(validation), [
            [int(heuristic_scores(row["text"])[k]) for k in LABELS] for row in validation])
        save_json(args.output / "heuristic-validation.json", {
            "split": "validation", "metrics": report, "provenance": provenance,
            "environment": environment(), "rules_version": __version__})
    elif args.method == "frozen":
        if (args.output / "frozen.json").exists():
            raise ValueError("baseline já existe; escolha outro --output")
        frozen_baseline(config, train, validation, args.output, args.device, provenance)
    else:
        if (args.output / "model").exists():
            raise ValueError("checkpoint já existe; escolha outro --output")
        fine_tune(config, train, validation, args.output, args.device, provenance)


if __name__ == "__main__":
    main()
