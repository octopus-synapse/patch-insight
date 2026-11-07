"""Prepare Hub directories for review; does not authenticate or upload."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from patchinsight.data import verify_freeze  # noqa: E402
from patchinsight.model import verify_bundle, write_manifest  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", required=True, help="Usuário ou organização do Hugging Face")
    parser.add_argument("--stage", choices=["scaffold", "trained"], default="scaffold")
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/run-42")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "reports/final/benchmark.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/hub")
    args = parser.parse_args()
    if not args.namespace or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                                 for c in args.namespace):
        raise ValueError("namespace inválido")
    if args.output.exists():
        raise ValueError("diretório já existe; escolha outro --output")
    manifest = verify_freeze(ROOT / "data")
    args.output.mkdir(parents=True)
    dataset = args.output / "dataset"
    shutil.copytree(ROOT / "data", dataset)
    (dataset / "docs").mkdir()
    for name in ("annotation-guide.md", "data-review.md"):
        shutil.copy2(ROOT / "docs" / name, dataset / "docs" / name)
    card = (dataset / "README.md").read_text().replace("../docs/", "docs/")
    (dataset / "README.md").write_text(card + "\nCódigo: https://github.com/octopus-synapse/patch-insight\n")
    shutil.copy2(ROOT / "licenses/DATA-AND-ARTICLE.md", dataset / "LICENSE.md")
    model = args.output / "model"
    if args.stage == "scaffold":
        model.mkdir()
        shutil.copy2(ROOT / "docs/MODEL_CARD.md", model / "README.md")
        shutil.copy2(ROOT / "LICENSE", model / "LICENSE")
        shutil.copy2(ROOT / "licenses/BERTIMBAU-MIT.txt", model / "BERTIMBAU-MIT.txt")
    else:
        metadata = verify_bundle(args.run / "model")
        benchmark = json.loads(args.benchmark.read_text())
        if metadata["provenance"]["dataset"] != manifest or benchmark["dataset"] != manifest:
            raise ValueError("modelo/benchmark incompatível com o dataset")
        from patchinsight.data import sha256
        if benchmark["model_manifest_sha256"] != sha256(args.run / "model/manifest.json"):
            raise ValueError("benchmark foi produzido para outro checkpoint")
        shutil.copytree(args.run / "model", model)
        shutil.copy2(args.run / "model/manifest.json", model / "training-manifest.json")
        card = (model / "README.md").read_text()
        start = card.index("**Status:")
        end = card.index("\n\n", start)
        card = card[:start] + "**Status: checkpoint ajustado; métricas sintéticas abaixo.**" + card[end:]
        card = card.replace("Model Card em preparação", "Model Card")
        card = card.replace("Treino previsto:", "Treino executado:")
        start = card.index("Avaliação pendente:")
        end = card.index("\n\n", start)
        lines = ["Avaliação final contra referências sintéticas revisadas pela mesma IA:", "",
                 "| Método | Macro-F1 | Micro-F1 | Exact match |", "|---|---:|---:|---:|"]
        for name, values in benchmark["methods"].items():
            lines.append(f"| {name} | {values['macro_f1']:.4f} | {values['micro_f1']:.4f} | {values['exact_match']:.4f} |")
        lines += ["", f"Dataset: https://huggingface.co/datasets/{args.namespace}/patchinsight-synthetic-ptbr",
                  f"Época selecionada por validação: {metadata['selected_epoch']}.",
                  "Contagens, suportes e ambiente completos em benchmark.json e patchinsight.json."]
        card = card[:start] + "\n".join(lines) + card[end:]
        card += ("\n\nTreino executado em GPU local NVIDIA RTX A1000 6 GB, Python 3.11.14. "
                 "Não houve sessão Kaggle/Colab nesta execução nem contratação de computação. "
                 "O PyTorch avisou sobre atenção backward não determinística; não se alega "
                 "identidade bit a bit entre retreinos. O ganho observado é pequeno e não "
                 "comprova superioridade em currículos reais.\n\n"
                 "O benchmark referencia o manifest original do treino, preservado em "
                 "`training-manifest.json`; `manifest.json` cobre o bundle publicado, "
                 "incluindo esta card atualizada.\n")
        (model / "README.md").write_text(card)
        shutil.copy2(args.benchmark, model / "benchmark.json")
        write_manifest(model)
    print(json.dumps({"dataset": str(dataset), "model": str(model), "stage": args.stage,
                      "model_has_weights": args.stage == "trained"}, indent=2))


if __name__ == "__main__":
    main()
