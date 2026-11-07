"""Write evidence of checks actually run, with explicit unexecuted gates."""
import ast
import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    sources = [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("tests/*.py")]
    for path in sources:
        ast.parse(path.read_text(), filename=str(path), feature_version=(3, 11))
    import os
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    if args.model:
        env["PATCHINSIGHT_MODEL"] = str(args.model.resolve())
    command = ([sys.executable, "-m", "pytest", "-v", "--junitxml=reports/pytest.xml"]
               if args.full else [sys.executable, "-m", "unittest", "discover", "-s", "tests",
                                  "-p", "test_core.py", "-v"])
    result = subprocess.run(command, cwd=ROOT, env=env,
                            text=True, capture_output=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "python_311_runtime_verified": sys.version_info[:2] == (3, 11),
        "python_311_syntax": {"status": "passed", "files": len(sources)},
        "tests": {"scope": "full" if args.full else "core", "command": command,
                  "real_model_requested": bool(args.model), "returncode": result.returncode,
                  "stdout": result.stdout, "stderr": result.stderr},
        "available_modules": {name: importlib.util.find_spec(name) is not None
                              for name in ("pydantic", "pytest", "torch", "transformers", "sklearn")},
        "evidence_files": {name: (ROOT / path).exists() for name, path in {
            "finetuning": "artifacts/run-42/training.json",
            "frozen_encoder_baseline": "artifacts/run-42/frozen.json",
            "final_benchmark": "reports/final/benchmark.json",
            "docker_offline": "artifacts/docker-prediction.json",
            "clean_python311_reproduction": "reports/clean-room.json"}.items()},
        "lockfile_exists": (ROOT / "uv.lock").exists(),
    }
    path = ROOT / "reports/verification.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(result.stdout, end="")
    print(result.stderr, end="")
    print(f"Evidence: {path}")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
