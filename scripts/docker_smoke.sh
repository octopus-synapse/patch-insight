#!/usr/bin/env bash
set -euo pipefail
if [[ ! -f uv.lock || ! -f artifacts/model/manifest.json ]]; then
  echo 'É necessário uv.lock resolvido e checkpoint real em artifacts/model.' >&2
  exit 2
fi
docker build -t patchinsight:0.1.0 .
docker run --rm --network none --read-only --tmpfs /tmp patchinsight:0.1.0 \
  --text 'Revisei as consultas com SQL e reduzi a espera em 20%.' > artifacts/docker-prediction.json
uv run --frozen python -c \
  'from pathlib import Path; from patchinsight.schema import Prediction; Prediction.model_validate_json(Path("artifacts/docker-prediction.json").read_text())'
echo 'Inferência offline validada.'
