"""Deliberately small, deterministic lexicon and structural rules.

Structures describe co-occurrence, not proof of causal relations or writing quality.
"""
import re

from . import LABELS
from .text import folded, normalize

TECHNOLOGIES = {
    "Python": (r"python",), "SQL": (r"sql",), "PostgreSQL": (r"postgres(?:ql)?",),
    "JavaScript": (r"javascript", r"node\.?js"), "TypeScript": (r"typescript",),
    "React": (r"react(?:\.js)?",), "Java": (r"java",), "Docker": (r"docker",),
    "Kubernetes": (r"kubernetes", r"k8s"), "Git": (r"git",),
    "GitHub Actions": (r"github actions",), "PyTorch": (r"pytorch",),
    "TensorFlow": (r"tensorflow",), "scikit-learn": (r"scikit-learn", r"sklearn"),
    "Airflow": (r"airflow",), "Spark": (r"(?:apache )?spark", r"pyspark"),
    "dbt": (r"dbt",), "Redis": (r"redis",), "Kafka": (r"kafka",),
    "Terraform": (r"terraform",), "AWS": (r"aws", r"amazon web services"),
    "FastAPI": (r"fastapi",), "OAuth": (r"oauth(?:\s*2\.0)?",),
    "BERTimbau": (r"bertimbau",), "Hugging Face": (r"hugging face",),
    "Power BI": (r"power\s*bi",), "pandas": (r"pandas",),
}


def technologies(text: str) -> list[str]:
    low = text.casefold()
    return sorted(name for name, aliases in TECHNOLOGIES.items()
                  if any(re.search(r"(?<!\w)(?:" + alias + r")(?!\w)", low)
                         for alias in aliases))


def measurement(text: str) -> bool:
    # Versions, years and issue IDs alone have no measurable referent.
    low = folded(text)
    low = re.sub(r"\b(?:oauth|python|java|versao|v)\s*\d+(?:\.\d+)*", "", low)
    return bool(re.search(
        r"\b\d+(?:[.,]\d+)?\s*(?:%|por cento|ms\b|segundos?\b|minutos?\b|horas?\b|"
        r"dias?\b|semanas?\b|meses?\b|anos?\b|reais\b|registros?\b|linhas?\b|"
        r"testes?\b|servicos?\b|endpoints?\b|usuarios?\b|equipes?\b|pipelines?\b|"
        r"modelos?\b|arquivos?\b|documentos?\b|relatorios?\b|tabelas?\b|jobs?\b|"
        r"requisicoes?\b|incidentes?\b|releases?\b|rotinas?\b|amostras?\b|gb\b|mb\b)", low))


def heuristic_scores(text: str) -> dict[str, float]:
    low = folded(normalize(text))
    # Clause-wise negation/future handling avoids suppressing a later performed action.
    clauses = re.split(r"[.;!?]|\bmas\b", low)
    performed = [c for c in clauses if not re.search(
        r"\b(nao|nunca|pretendo|planejo|farei|vou|gostaria|quero|sera|serao)\b", c)]
    action_re = r"\b(implementei|criei|desenvolvi|automatizei|migrei|treinei|ajustei|corrigi|"
    action_re += r"construi|publiquei|entreguei|configurei|refatorei|documentei|revisei|"
    action_re += r"integrei|reduzi|validei|mapeei|instrumentei|escrevi|extra[ií]|otimizei)\b"
    action = any(re.search(action_re, c) for c in performed)
    result = any(re.search(
        r"\b(reduz\w*|aument\w*|elimin\w*|entreguei|publiquei|disponibiliz\w*|"
        r"permit\w*|passou|passaram|ficou|ficaram|conclu\w*|gerou|geraram|"
        r"liberou|liberaram|restabelec\w*)\b", c) for c in performed)
    context = bool(re.search(r"\b(para (?:o|a|os|as|uma|um) |durante|diante|no time|"
                             r"na equipe|no setor|em um|em uma|quando|havia)\b", low))
    method = bool(re.search(r"\b(usando|utilizando|por meio|com base|via|"
                            r"testes de|validacao cruzada|revisao por pares)\b", low))
    method |= bool(re.search(r"\bcom\b", low) and technologies(text))
    return dict(zip(LABELS, map(float, (action, context, method, result, measurement(text)))))


def structural_rules(text: str, states: dict[str, str]):
    techs = technologies(text)
    def present(label):
        return states[label] == "present"
    # Explicit objective/responsibility only. Absence of a pattern is inconclusive.
    task = "present" if re.search(
        r"\b(?:meu objetivo era|minha responsabilidade era|tinha a tarefa de|"
        r"fui encarregad[oa] de)\b", folded(text)) else "unknown"
    structures = []
    if all(present(k) for k in ("action", "result", "measurement", "method")):
        structures.append("XYZ")
    if all(present(k) for k in ("context", "action", "result")) and task == "present":
        structures.append("STAR")
    if present("action") and present("result"):
        structures.append("Action–Result")
    if present("action") and techs:
        structures.append("Action–Technology")
    if present("action") and present("method"):
        structures.append("Action–Method")
    diagnostics = [f"missing_{k}" for k in LABELS if states[k] == "absent"]
    diagnostics += [f"unknown_{k}" for k in LABELS if states[k] == "unknown"]
    if task == "unknown":
        diagnostics.append("task_unknown")
    if not structures:
        diagnostics.append("unclassified")
    return structures, task, techs, diagnostics
