import json
import subprocess
import sys

import pytest

pytest.importorskip("pydantic")
from pydantic import ValidationError
from patchinsight import LABELS
from patchinsight.schema import Component, Input, Prediction, Versions


def test_strict_input_and_output():
    with pytest.raises(ValidationError):
        Input(text=" ")
    with pytest.raises(ValidationError):
        Input(text="a", quality=100)
    with pytest.raises(ValidationError):
        Component(state="present", score=float("nan"))
    with pytest.raises(ValidationError):
        Prediction(text="A", components={}, structures=[], task="unknown", technologies=[],
                   diagnostics=[], truncated=False, versions=Versions(model_version="test"),
                   backend="heuristic", score_kind="binary_rule")


def test_cli_json_and_stderr():
    result = subprocess.run([sys.executable, "-m", "patchinsight", "predict", "--text",
                             "Corrigi a consulta com SQL.", "--backend", "heuristic", "--json"],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    parsed = Prediction.model_validate_json(result.stdout)
    assert set(parsed.components) == set(LABELS)
    assert parsed.versions.schema_version == "0.2.0"
    invalid = subprocess.run([sys.executable, "-m", "patchinsight", "predict", "--text", " "],
                             text=True, capture_output=True)
    assert invalid.returncode == 2
    assert not invalid.stdout
    assert "error" in json.loads(invalid.stderr)


def test_no_silent_heuristic_fallback():
    result = subprocess.run([sys.executable, "-m", "patchinsight", "predict", "--text", "Olá",
                             "--model", "/nonexistent/patchinsight"], text=True, capture_output=True)
    assert result.returncode == 2
    assert not result.stdout
    assert "error" in json.loads(result.stderr)
