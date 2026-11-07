import argparse
import json
import sys
import time
from pathlib import Path


def batch_records(stream, analyzer, batch_size=8, max_items=10000, timeout=300):
    """Bounded JSONL batches. Timeout is cooperative, between batches."""
    from .schema import Input
    started = time.monotonic()
    count = 0
    while True:
        pending = []
        eof = False
        for _ in range(batch_size):
            line = stream.readline(100001)
            if not line:
                eof = True
                break
            count += 1
            if count > max_items:
                raise ValueError("limite de itens excedido; saída anterior permanece válida")
            if len(line) > 100000:
                raise ValueError("linha JSONL excede 100000 caracteres")
            identifier = None
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) - {"id", "text"}:
                    raise ValueError("cada linha exige objeto com text e id opcional")
                identifier = row.get("id")
                if identifier is not None and (not isinstance(identifier, str) or len(identifier) > 200):
                    identifier = None
                    raise ValueError("id deve ser string de até 200 caracteres")
                text = Input(text=row.get("text")).text
                pending.append({"line": count, "id": identifier, "text": text})
            except (ValueError, TypeError) as exc:
                pending.append({"line": count, "id": identifier, "error": str(exc)})
        if not pending:
            return
        if time.monotonic() - started > timeout:
            raise ValueError("timeout entre lotes; saída anterior permanece válida")
        valid = [r for r in pending if "error" not in r]
        results = iter(analyzer.analyze_many([r["text"] for r in valid]))
        for row in pending:
            if "error" not in row:
                row.pop("text")
                row["prediction"] = next(results).model_dump(mode="json")
            yield row
        if eof:
            return


def main(argv=None):
    parser = argparse.ArgumentParser(prog="patchinsight")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("predict", "batch", "serve"):
        command = sub.add_parser(name)
        command.add_argument("--backend", choices=["bert", "heuristic"], default="bert")
        command.add_argument("--model", type=Path, default=Path("artifacts/model"))
        command.add_argument("--device", default="cpu")
        command.add_argument("--policy", type=Path, help="Calibração/abstenção experimental opt-in")
        if name == "predict":
            command.add_argument("--text", required=True)
            command.add_argument("--json", action="store_true")
        elif name == "batch":
            command.add_argument("--input", type=Path, help="JSONL; omitido lê stdin")
            command.add_argument("--batch-size", type=int, default=8, choices=range(1, 65))
            command.add_argument("--max-items", type=int, default=10000)
            command.add_argument("--timeout", type=float, default=300)
        else:
            command.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        if args.command == "batch" and (args.max_items < 1 or not 0 < args.timeout <= 86400):
            raise ValueError("max-items deve ser positivo; timeout entre 0 e 86400 segundos")
        if args.command == "predict":
            from .schema import Input
            Input(text=args.text)
        from .service import Analyzer
        analyzer = Analyzer(args.backend, args.model, args.device, args.policy)
        if args.command == "predict":
            print(analyzer.analyze(args.text).model_dump_json())
        elif args.command == "batch":
            stream = args.input.open(encoding="utf-8") if args.input else sys.stdin
            errors = False
            try:
                for row in batch_records(stream, analyzer, args.batch_size, args.max_items, args.timeout):
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    errors |= "error" in row
            finally:
                if args.input:
                    stream.close()
            return 1 if errors else 0
        else:
            from .demo import serve
            serve(analyzer, args.port)
        return 0
    except (ValueError, OSError, ImportError, RuntimeError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
