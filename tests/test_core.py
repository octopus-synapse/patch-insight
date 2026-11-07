"""The dependency-free core is also runnable with python -m unittest."""
import json
import tempfile
import unittest
from pathlib import Path

from patchinsight.data import audit, load_split, verify_freeze
from patchinsight.metrics import metrics
from patchinsight.rules import heuristic_scores, measurement, structural_rules, technologies
from patchinsight.text import normalize

ROOT = Path(__file__).resolve().parents[1]


class TextTests(unittest.TestCase):
    def test_normalization_preserves_case_accents(self):
        self.assertEqual(normalize("  Ac\u0327a\u0303o\n com\tSQL  "), "Ação com SQL")

    def test_invalid(self):
        for value in ("", "\t ", "x\x00", "x" * 20001, None):
            with self.assertRaises(ValueError):
                normalize(value)


class RuleTests(unittest.TestCase):
    def test_technology_boundaries_aliases(self):
        self.assertEqual(technologies("Postgres, K8s, node.js, sklearn e OAuth 2.0"),
                         ["JavaScript", "Kubernetes", "OAuth", "PostgreSQL", "scikit-learn"])
        self.assertEqual(technologies("JavaScript não é Java; digital não é git"),
                         ["Git", "Java", "JavaScript"])
        self.assertEqual(technologies("digital javali reatividade"), [])

    def test_measurement(self):
        for text in ("OAuth 2.0", "Python 3.11", "Em 2026", "INC-42", "versão 12"):
            self.assertFalse(measurement(text), text)
        for text in ("12 serviços", "20%", "2 horas", "OAuth 2.0 em 3 endpoints", "500 ms"):
            self.assertTrue(measurement(text), text)

    def test_negation_and_future(self):
        for text in ("Não implementei cache.", "Pretendo reduzir a latência.", "Atuei com IA."):
            self.assertEqual(heuristic_scores(text)["action"], 0)
        self.assertEqual(heuristic_scores("Não implementei cache; corrigi o timeout.")["action"], 1)

    def test_structural_fixtures(self):
        for case in json.loads((ROOT / "tests/fixtures/structures.json").read_text()):
            with self.subTest(case=case["name"]):
                structures, task, _, diagnostics = structural_rules(case["text"], case["states"])
                self.assertEqual(structures, case["structures"])
                self.assertEqual(task, case["task"])
                if not structures:
                    self.assertIn("unclassified", diagnostics)


class MetricTests(unittest.TestCase):
    def test_known_confusion(self):
        r = metrics([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
                    [[1, 1, 0, 0, 0], [0, 0, 0, 0, 0]])
        self.assertEqual(r["per_label"]["context"]["fp"], 1)
        self.assertEqual(r["per_label"]["context"]["fn"], 1)
        self.assertEqual(r["macro_f1"], .2)
        self.assertEqual(r["micro_f1"], .5)
        self.assertEqual(r["exact_match"], 0)

    def test_invalid(self):
        for y, p in (([], []), ([[1]], [[1]]), ([[2] * 5], [[1] * 5])):
            with self.assertRaises(ValueError):
                metrics(y, p)


class DataTests(unittest.TestCase):
    def test_freeze_and_splits(self):
        manifest = verify_freeze(ROOT / "data")
        self.assertEqual(manifest["counts"], {"train": 280, "validation": 60, "test": 60})
        splits = {s: load_split(ROOT / "data", s) for s in ("train", "validation", "test")}
        groups = {s: {r["group_id"] for r in rows} for s, rows in splits.items()}
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])
        self.assertFalse(groups["train"] & groups["validation"])
        report = audit(ROOT / "data")
        self.assertEqual(report["n"], 400)
        self.assertEqual(report["groups"], 100)

    def test_reject_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "freeze.json").write_text('{"sha256": {"file": "wrong"}}')
            (root / "file").write_text("tampered")
            with self.assertRaises(ValueError):
                verify_freeze(root)


if __name__ == "__main__":
    unittest.main()
