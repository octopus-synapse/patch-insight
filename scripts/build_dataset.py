"""Render reviewed synthetic scenario clauses; never silently replace a freeze.

Run only before experiments, into a new directory. The checked-in data is the
release input; this script documents its compositional generation mechanism.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from patchinsight import LABELS
from patchinsight.text import normalize


ROOT = Path(__file__).resolve().parents[1]
MASKS = ("10000", "10010", "10100", "11000", "11100", "01000", "00010",
         "01010", "01110", "00111", "00011", "01001", "10110", "11010",
         "01101", "10001", "01100", "10101", "11110", "11011", "11001")
VERBS = {"Investiguei": "investigar", "Revisei": "revisar", "Analisei": "analisar",
         "Ajustei": "ajustar", "Comparei": "comparar", "Examinei": "examinar",
         "Inspecionei": "inspecionar", "Avaliei": "avaliar"}


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def generate(source):
    provenance = json.loads((source / "prompts.json").read_text())
    dev = list(csv.DictReader((source / "development.tsv").open(), delimiter="\t"))
    test = list(csv.DictReader((source / "test.tsv").open(), delimiter="\t"))
    rng = random.Random(42)
    order = list(range(len(dev)))
    rng.shuffle(order)
    validation = set(order[:15])
    rows = []
    for index, scenario in enumerate(dev + test):
        test_only = index >= len(dev)
        split = "test" if test_only else ("validation" if index in validation else "train")
        group = scenario["id"]
        # All choices are made before experiments and independent of model outputs.
        masks = ["11111", MASKS[index % len(MASKS)], MASKS[(index + 7) % len(MASKS)]]
        for variant in range(4):
            if variant < 3:
                labels = {k: bit == "1" for k, bit in zip(LABELS, masks[variant])}
                parts = [scenario[k] for k in ("context", "action", "method", "result", "measurement")
                         if labels[k]]
                transformation = "composition_mask_" + masks[variant]
            else:
                verb, rest = scenario["action"].split(" ", 1)
                infinitive = VERBS[verb]
                if index % 2:
                    negative = "Não cheguei a " + infinitive + " " + rest
                    transformation = "negated_action"
                else:
                    negative = "Pretendo " + infinitive + " " + rest
                    transformation = "future_action"
                labels = dict(zip(LABELS, (False, index % 3 == 0, index % 3 == 1,
                                          False, index % 4 == 0)))
                parts = ([scenario["context"]] if labels["context"] else []) + [negative]
                parts += [scenario[k] for k in ("method", "measurement") if labels[k]]
            text = normalize(" ".join(parts))
            record_number = index * 4 + variant + 1
            rows.append({
                "id": f"pi-{record_number:04d}", "text": text, "labels": labels,
                "origin_id": group, "group_id": group, "split": split,
                "batch": index // 25 + 1, "pilot": record_number <= 50,
                "domain": "software_data_ai", "language": "pt-BR",
                "synthetic": True, "human_reviewed": False,
                "generator": provenance["generator"], "reviewer": provenance["reviewer"],
                "guide_version": provenance["guide_version"], "dataset_version": "0.1.0",
                "prompt_ref": "source/prompts.json#generation",
                "review_prompt_ref": "source/prompts.json#review",
                "source_ref": f"source/{'test' if test_only else 'development'}.tsv#{group}",
                "transformation": transformation,
                "review_history": [{"revision": 1, "actor": "same_ai_assistant",
                    "operation": "clause_semantics_and_composition_review",
                    "note": "Rótulos pela composição revisada das cláusulas; sem anotação independente."}],
                "license": "CC-BY-4.0",
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "freeze.json").exists():
        raise SystemExit("Já congelado. Não sobrescrever dados de um experimento.")
    rows = generate(ROOT / "data/source")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "splits").mkdir(exist_ok=True)
    jsonl(args.output / "dataset.jsonl", rows)
    for split in ("train", "validation", "test"):
        jsonl(args.output / "splits" / f"{split}.jsonl", [r for r in rows if r["split"] == split])
    paths = ["dataset.jsonl", *[f"splits/{s}.jsonl" for s in ("train", "validation", "test")]]
    # Copy sources when reproducing into a separate directory.
    import shutil
    if (args.output / "source").resolve() != (ROOT / "data/source").resolve():
        shutil.copytree(ROOT / "data/source", args.output / "source", dirs_exist_ok=True)
    paths += ["source/development.tsv", "source/test.tsv", "source/prompts.json"]
    hashes = {p: hashlib.sha256((args.output / p).read_bytes()).hexdigest() for p in paths}
    manifest = {"dataset_version": "0.1.0", "seed": 42,
                "protocol": "compositional_groups_v1; fixed before experiments",
                "counts": {"train": 280, "validation": 60, "test": 60}, "sha256": hashes}
    (args.output / "freeze.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
