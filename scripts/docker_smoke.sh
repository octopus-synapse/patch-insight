#!/usr/bin/env bash
set -euo pipefail
if [[ ! -f uv.lock || ! -f artifacts/model/manifest.json ]]; then
  echo 'É necessário uv.lock resolvido e checkpoint real em artifacts/model.' >&2
  exit 2
fi
release_version="$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
prediction_path="artifacts/docker-prediction-${release_version}.json"
docker build -t "patchinsight:${release_version}" .
docker run --rm --network none --read-only --tmpfs /tmp "patchinsight:${release_version}" \
  --text 'Revisei as consultas com SQL e reduzi a espera em 20%.' > "$prediction_path"
uv run --frozen python -c \
  'import sys; from pathlib import Path; from patchinsight.schema import Prediction; Prediction.model_validate_json(Path(sys.argv[1]).read_text())' "$prediction_path"
echo 'Inferência offline validada.'
