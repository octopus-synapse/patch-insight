"""Dataset audit and freeze verification, usable without the ML environment."""
import hashlib
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from . import LABELS
from .text import folded, normalize


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verify_freeze(root):
    root = Path(root)
    manifest = json.loads((root / "freeze.json").read_text())
    for filename, digest in manifest["sha256"].items():
        if sha256(root / filename) != digest:
            raise ValueError(f"congelamento inválido: {filename}")
    return manifest


def load_split(root, split):
    if split not in ("train", "validation", "test"):
        raise ValueError("split inválido")
    verify_freeze(root)
    return read_jsonl(Path(root) / "splits" / f"{split}.jsonl")


def audit(root):
    root = Path(root)
    rows = read_jsonl(root / "dataset.jsonl")
    ids, texts, groups = set(), {}, {}
    support = {s: Counter() for s in ("train", "validation", "test")}
    sizes, combinations = Counter(), Counter()
    for row in rows:
        if row["id"] in ids:
            raise ValueError(f"id duplicado: {row['id']}")
        ids.add(row["id"])
        if normalize(row["text"]) != row["text"]:
            raise ValueError(f"texto não normalizado: {row['id']}")
        key = folded(row["text"])
        if key in texts:
            raise ValueError(f"duplicata: {texts[key]} / {row['id']}")
        texts[key] = row["id"]
        if set(row["labels"]) != set(LABELS) or any(
                type(v) is not bool for v in row["labels"].values()):
            raise ValueError(f"rótulos inválidos: {row['id']}")
        split, group = row["split"], row["group_id"]
        if groups.setdefault(group, split) != split:
            raise ValueError(f"grupo vazado: {group}")
        if not row["synthetic"] or row["human_reviewed"] or not row["review_history"]:
            raise ValueError(f"proveniência inválida: {row['id']}")
        support[split].update(k for k, v in row["labels"].items() if v)
        sizes[split] += 1
        combinations["".join(str(int(row["labels"][k])) for k in LABELS)] += 1
    if dict(sizes) != {"train": 280, "validation": 60, "test": 60}:
        raise ValueError(f"tamanhos inesperados: {sizes}")
    for split, counts in support.items():
        if any(not 0 < counts[k] < sizes[split] for k in LABELS):
            raise ValueError(f"falta cobertura positiva/negativa: {split}")
        actual = read_jsonl(root / "splits" / f"{split}.jsonl")
        if actual != [r for r in rows if r["split"] == split]:
            raise ValueError(f"split divergente: {split}")
    # Similarity is a flag, not proof of independence. Do not expose test errors.
    near = []
    dev = [r for r in rows if r["split"] != "test"]
    for i, a in enumerate(dev):
        for b in dev[i + 1:]:
            if a["group_id"] == b["group_id"]:
                continue
            ratio = SequenceMatcher(None, folded(a["text"]), folded(b["text"])).ratio()
            if ratio >= .85:
                near.append({"ids": [a["id"], b["id"]], "similarity": round(ratio, 4)})
    return {"n": len(rows), "groups": len(groups), "splits": dict(sizes),
            "support": support, "label_combinations": combinations,
            "development_near_duplicates": near, "similarity_threshold": .85,
            "limitations": "Composição sintética; grupos não garantem independência semântica."}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verify_freeze(args.data)
    report = audit(args.data)
    output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
