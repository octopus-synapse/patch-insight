"""Create a reviewable public source bundle by allowlist, excluding private inputs."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOW = ["src", "tests", "data", "configs", "scripts", "notebooks", "docs", "licenses",
         ".github", "README.md", "LICENSE", "pyproject.toml", ".python-version", ".gitignore",
         ".dockerignore", "Dockerfile", "reports/data-audit.json", "reports/development",
         "reports/verification.json", "reports/training", "reports/final",
         "reports/clean-room.json", "reports/publication.json", "reports/docker.json",
         "reports/v02", "reports/v02-tests.xml", "reports/pytest.xml", "reports/pytest-initial.xml", "artifacts/.gitkeep"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/patchinsight-source-0.2.0.tar.gz")
    args = parser.parse_args()
    paths = []
    for name in ALLOW + (["uv.lock"] if (ROOT / "uv.lock").exists() else []):
        path = ROOT / name
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(p for p in path.rglob("*") if p.is_file()
                         and "__pycache__" not in p.parts and p.suffix != ".pyc")
    paths = sorted(set(paths))
    if any(p.suffix.lower() == ".pdf" or p.is_symlink() for p in paths):
        raise ValueError("arquivo privado ou symlink na lista de publicação")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=f"patchinsight/{path.relative_to(ROOT)}", recursive=False)
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{args.output}: {len(paths)} arquivos; PDF e especificação privada excluídos")


if __name__ == "__main__":
    main()
