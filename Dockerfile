# Requires a resolved uv.lock and a real model prepared in artifacts/model.
FROM ghcr.io/astral-sh/uv:0.9.7 AS uv
FROM python:3.11.14-slim-bookworm AS build
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --extra ml --no-dev --no-editable

FROM python:3.11.14-slim-bookworm
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
COPY --from=build /app/.venv /app/.venv
COPY LICENSE ./
COPY licenses ./licenses
COPY artifacts/model ./model
RUN python -c "from patchinsight.model import BertPredictor; BertPredictor('/app/model')"
USER 65532:65532
ENTRYPOINT ["patchinsight", "predict", "--model", "/app/model", "--json"]
