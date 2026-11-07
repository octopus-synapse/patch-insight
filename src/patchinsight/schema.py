from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import LABELS, RULES_VERSION, SCHEMA_VERSION
from .text import normalize


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class State(str, Enum):
    present = "present"
    absent = "absent"
    unknown = "unknown"


class Input(StrictModel):
    text: str

    @field_validator("text")
    @classmethod
    def valid_text(cls, value: str) -> str:
        return normalize(value)


class Component(StrictModel):
    state: State
    score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


class Versions(StrictModel):
    schema_version: Literal["0.2.0"] = SCHEMA_VERSION
    model_version: str
    rules_version: Literal["0.2.0"] = RULES_VERSION


class Prediction(StrictModel):
    text: str
    components: dict[str, Component]
    structures: list[Literal["XYZ", "STAR", "Action–Result", "Action–Technology", "Action–Method"]]
    task: State
    technologies: list[str]
    diagnostics: list[str]
    truncated: bool
    tokens: int | None = Field(default=None, ge=0)
    versions: Versions
    backend: Literal["heuristic", "bert"]
    score_kind: Literal["binary_rule", "sigmoid_uncalibrated", "temperature_scaled_experimental"]

    @model_validator(mode="after")
    def labels_complete(self):
        if set(self.components) != set(LABELS):
            raise ValueError("components deve conter exatamente os cinco rótulos")
        for component in self.components.values():
            if component.state == State.unknown and component.score is not None:
                raise ValueError("unknown exige score null")
        return self
