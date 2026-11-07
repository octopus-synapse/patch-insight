#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
if [[ ! -f uv.lock ]]; then
  echo 'Falta uv.lock. Com rede: uv lock --python 3.11; revisar e versionar antes dos experimentos.' >&2
  exit 2
fi
uv sync --frozen --extra ml --extra dev
uv run --frozen python -m patchinsight.data --output reports/data-audit.json
uv run --frozen pytest
uv run --frozen python -m patchinsight.experiment heuristic --output artifacts/run-42
uv run --frozen python -m patchinsight.experiment frozen --output artifacts/run-42 --device "${PATCHINSIGHT_DEVICE:-cpu}"
uv run --frozen python -m patchinsight.experiment finetune --output artifacts/run-42 --device "${PATCHINSIGHT_DEVICE:-cpu}"
PATCHINSIGHT_MODEL=artifacts/run-42/model uv run --frozen pytest -m real_model
echo 'Seleção concluída. Verifique training.json e finalize antes de executar evaluate --final.'
