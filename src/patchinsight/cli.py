import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="patchinsight")
    sub = parser.add_subparsers(dest="command", required=True)
    predict = sub.add_parser("predict", help="Detectar componentes sem avaliar pessoas")
    predict.add_argument("--text", required=True)
    predict.add_argument("--json", action="store_true", help="Saída JSON (também é o padrão)")
    predict.add_argument("--backend", choices=["bert", "heuristic"], default="bert")
    predict.add_argument("--model", type=Path, default=Path("artifacts/model"))
    args = parser.parse_args(argv)
    try:
        from .schema import Component, Input, Prediction, Versions
        from .rules import heuristic_scores, structural_rules
        text = Input(text=args.text).text
        if args.backend == "heuristic":
            scores = heuristic_scores(text)
            states = {k: "present" if v else "absent" for k, v in scores.items()}
            truncated, tokens, model_version = False, None, "heuristic-0.1.0"
        else:
            from .model import BertPredictor
            model = BertPredictor(args.model)
            scores, states, truncated, tokens = model.predict(text)
            model_version = model.version
        structures, task, techs, diagnostics = structural_rules(text, states)
        if truncated:
            diagnostics.append("input_truncated_unknown_negatives")
        result = Prediction(
            text=text, components={k: Component(state=states[k], score=scores[k]) for k in scores},
            structures=structures, task=task, technologies=techs, diagnostics=diagnostics,
            truncated=truncated, tokens=tokens, versions=Versions(model_version=model_version),
            backend=args.backend,
            score_kind="binary_rule" if args.backend == "heuristic" else "sigmoid_uncalibrated")
        print(result.model_dump_json())
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
