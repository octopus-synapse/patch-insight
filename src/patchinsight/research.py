"""Versioned, offline exploratory evaluations with development/test policy isolation."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import LABELS
from .assessment import (extended_metrics, fit_policy, paired_group_bootstrap,
                         selective_metrics, temperature_scale)
from .data import sha256, verify_freeze
from .experiment import embeddings, environment, save_json
from .model import BertPredictor
from .rules import heuristic_scores


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def suite_rows(root, split):
    manifest = json.loads((root / 'freeze.json').read_text())
    for name, digest in manifest['sha256'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'suite modificada após congelamento: {name}')
    rows = read_rows(root / f'{split}.jsonl')
    other = read_rows(root / ('test.jsonl' if split == 'development' else 'development.jsonl'))
    if {r['group_id'] for r in rows} & {r['group_id'] for r in other}:
        raise ValueError('origens compartilhadas entre splits')
    if len({r['id'] for r in rows}) != len(rows) or len({r['text'] for r in rows}) != len(rows):
        raise ValueError('IDs/textos duplicados')
    if len(rows) != manifest['counts'][split] or any(r['split'] != split for r in rows):
        raise ValueError('split incompatível com congelamento')
    return rows, manifest


class Methods:
    def __init__(self, run, device):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.device = device
        if device == 'cpu':
            torch.set_num_threads(min(4, torch.get_num_threads()))
        start = time.perf_counter()
        self.bert = BertPredictor(run / 'model', device=device)
        self.frozen = json.loads((run / 'frozen.json').read_text())
        if self.frozen['labels'] != list(LABELS):
            raise ValueError('labels da baseline incompatíveis')
        self.config = self.frozen['experiment']
        if self.config['thresholds'] != [.5] * 5 or self.bert.metadata['thresholds'] != [.5] * 5:
            raise ValueError('esta avaliação exige thresholds congelados em 0.5')
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['base_model'],
            revision=self.config['base_revision'], local_files_only=True)
        self.encoder = AutoModel.from_pretrained(self.config['base_model'],
            revision=self.config['base_revision'], local_files_only=True).to(device).eval()
        self.synchronize()
        self.load_seconds = time.perf_counter() - start

    def synchronize(self):
        if self.device.startswith('cuda'):
            import torch
            torch.cuda.synchronize()

    def score(self, texts, method):
        if method == 'heuristic':
            return np.asarray([[heuristic_scores(t)[k] for k in LABELS] for t in texts])
        if method == 'bert_finetuned':
            return np.asarray(self.bert.score_batch(texts))
        x = embeddings(texts, self.tokenizer, self.encoder, self.device,
                       self.config['max_length'], self.config['batch_size'])
        x = (x - np.asarray(self.frozen['mean'])) / np.asarray(self.frozen['scale'])
        logits = x @ np.asarray(self.frozen['weights']).T + np.asarray(self.frozen['bias'])
        return 1 / (1 + np.exp(-np.clip(logits, -80, 80)))

    def all_scores(self, texts):
        results = {}
        for method in ('heuristic', 'frozen_logistic', 'bert_finetuned'):
            batches = [self.score(texts[i:i+8], method) for i in range(0, len(texts), 8)]
            results[method] = np.concatenate(batches)
        return results


def audit(rows):
    y = np.asarray([[int(r['labels'][k]) for k in LABELS] for r in rows])
    if y.shape != (len(rows), 5) or not np.isin(y, [0, 1]).all():
        raise ValueError('rótulos inválidos')
    if any(sum(y[:, i]) == 0 or sum(y[:, i]) == len(y) for i in range(5)):
        raise ValueError('cada componente deve ter positivos e negativos no split')
    if any(not r['synthetic'] or r['human_reviewed'] or not r['review_note'] for r in rows):
        raise ValueError('proveniência/revisão incompatível')
    return {'n': len(rows), 'origins': len(set(r['group_id'] for r in rows)),
            'positive_support': dict(zip(LABELS, y.sum(axis=0).tolist())),
            'reference': 'synthetic_same_ai_self_reviewed'}, y


def slices(rows, y, scores):
    tags = sorted({tag for row in rows for tag in row['tags']})
    result = {}
    for tag in tags:
        indices = [i for i, row in enumerate(rows) if tag in row['tags']]
        result[tag] = {'n': len(indices), 'methods': {
            name: extended_metrics(y[indices], values[indices]) for name, values in scores.items()}}
    return result


def marker_probe(rows, scores):
    result = {}
    for name, values in scores.items():
        pairs = []
        for group in sorted({r['group_id'] for r in rows if 'marker_probe' in r['tags']}):
            marked = next(i for i, r in enumerate(rows) if r['group_id'] == group and 'marked' in r['tags'])
            plain = next(i for i, r in enumerate(rows) if r['group_id'] == group and 'unmarked' in r['tags'])
            pairs.append({'group': group, 'mean_absolute_score_change': float(np.abs(values[marked] - values[plain]).mean()),
                          'flipped_components': int(((values[marked] >= .5) != (values[plain] >= .5)).sum())})
        result[name] = pairs
    return result


def report_markdown(report):
    phase_name = {'development': 'de desenvolvimento', 'test': 'do novo teste', 'historical_posthoc_v01': 'posterior da v0.1'}[report['phase']]
    lines = ['# Avaliação ' + phase_name, '',
        'Referência sintética revisada pela mesma IA. Sem revisão humana ou evidência populacional.', '',
        '| Método | Macro-F1 | Exact match | Hamming ↓ | AP macro | Brier ↓ | Log loss ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, m in report['methods'].items():
        ap = 'null' if m['macro_average_precision'] is None else f"{m['macro_average_precision']:.4f}"
        lines.append(f"| {name} | {m['macro_f1']:.4f} | {m['exact_match']:.4f} | {m['hamming_loss']:.4f} | {ap} | {m['mean_binary_brier']:.4f} | {m['mean_binary_log_loss']:.4f} |")
    lines += ['', 'Métricas por rótulo, suportes, ROC-AUC, MCC, confiabilidade e risco/cobertura: `results.json`.',
              'Scores heurísticos são decisões binárias, sem interpretação probabilística calibrada.',
              'Log loss usa clipping em 1e-7. Esta análise não altera a seleção da v0.1.']
    if 'selected_policies' in report:
        lines += ['', '| Política experimental | Temperatura | Cutoff | Cobertura | Erro nas decisões aceitas |',
                  '|---|---:|---:|---:|---:|']
        for name, p in report['selected_policies'].items():
            selected = p['observed']
            error = selected['error_on_accepted']
            lines.append(f"| {name} | {p['temperature']:.4f} | {p['confidence_cutoff'] if p['confidence_cutoff'] is not None else 'desativado'} | {selected['coverage']:.4f} | {error if error is not None else 'null'} |")
        lines += ['', 'Política ajustada somente em desenvolvimento; desempenho em teste não é garantia de risco.',
                  'Nenhum cutoff cumpriu a meta de erro/cobertura em desenvolvimento. O erro na tabela é sem abstenção, não o risco de uma política aprovada.']
    return '\n'.join(lines) + '\n'


def latency(methods, texts):
    import torch
    result = {'device': methods.device, 'environment': environment(), 'load_all_methods_seconds': methods.load_seconds,
              'scope': 'tokenização + modelo + scores; exclui JSON, regras e I/O; mesmo processo, textos de desenvolvimento',
              'warmup_batches': 3, 'repetitions': 12, 'methods': {}}
    for name in ('heuristic', 'frozen_logistic', 'bert_finetuned'):
        result['methods'][name] = {}
        for size in (1, 8):
            batch = texts[:size]
            for _ in range(3):
                methods.score(batch, name)
            methods.synchronize()
            if methods.device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats()
            samples = []
            for _ in range(12):
                methods.synchronize()
                start = time.perf_counter()
                methods.score(batch, name)
                methods.synchronize()
                samples.append(time.perf_counter() - start)
            result['methods'][name][str(size)] = {
                'batch_size': size, 'batch_p50_ms': float(np.quantile(samples, .5) * 1000),
                'batch_p95_ms': float(np.quantile(samples, .95) * 1000),
                'texts_per_second': len(batch) / float(np.mean(samples)),
                'process_cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if methods.device.startswith('cuda') else None}
    result['memory_note'] = 'Pico CUDA do processo com ambos encoders residentes; não é memória isolada por método. CPU não medida.'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['historical', 'development', 'test', 'latency'])
    parser.add_argument('--suite', type=Path, default=Path('data/v02'))
    parser.add_argument('--run', type=Path, default=Path('artifacts/run-42'))
    parser.add_argument('--root-output', type=Path, default=Path('reports/v02'))
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    output = args.root_output / args.phase
    if output.exists():
        raise ValueError('relatório já existe; não será sobrescrito')
    if args.phase == 'historical':
        path = Path('reports/final/predictions.jsonl')
        records = read_rows(path)
        y = [r['target'] for r in records]
        report = {'phase': 'historical_posthoc_v01', 'source_sha256': sha256(path),
                  'selection_changed': False, 'methods': {name: extended_metrics(y, [r['scores'][name] for r in records])
                      for name in records[0]['scores']}, 'environment': environment()}
    else:
        split = 'development' if args.phase == 'latency' else args.phase
        rows, manifest = suite_rows(args.suite, split)
        data_audit, y = audit(rows)
        policies = {}
        # Read and verify frozen policies BEFORE computing test predictions.
        if args.phase == 'test':
            dev = args.root_output / 'development'
            selection = json.loads((dev / 'selection.json').read_text())
            for name, digest in selection['sha256'].items():
                if sha256(dev / name) != digest:
                    raise ValueError('política de desenvolvimento alterada')
                if name.endswith('-policy.json'):
                    policies[name.removesuffix('-policy.json')] = json.loads((dev/name).read_text())
            if selection['suite_manifest_sha256'] != sha256(args.suite / 'freeze.json'):
                raise ValueError('suite diferente do desenvolvimento')
            if selection['frozen_sha256'] != sha256(args.run / 'frozen.json'):
                raise ValueError('baseline diferente do desenvolvimento')
            if any(p['model_manifest_sha256'] != sha256(args.run / 'model/manifest.json') for p in policies.values()):
                raise ValueError('bundle diferente do desenvolvimento')
        methods = Methods(args.run, args.device)
        if args.phase == 'latency':
            report = latency(methods, [r['text'] for r in rows])
            output.mkdir(parents=True, exist_ok=False)
            save_json(output / 'results.json', report)
            print(json.dumps({'phase': 'latency', 'output': str(output)}))
            return
        scores = methods.all_scores([r['text'] for r in rows])
        if args.phase == 'development':
            for name in ('frozen_logistic', 'bert_finetuned'):
                policies[name] = dict(fit_policy(y, scores[name]), labels=list(LABELS),
                    model_manifest_sha256=sha256(args.run / 'model/manifest.json'),
                    development_sha256=sha256(args.suite / 'development.jsonl'),
                    method=name, policy_id=f'v02-{name}')
        selected = {}
        all_scores = dict(scores)
        for name, policy in policies.items():
            values = temperature_scale(scores[name], policy['temperature'])
            all_scores[name + '_temperature'] = values
            selected[name] = dict(policy, observed=selective_metrics(y, values,
                policy['confidence_cutoff'] if policy['confidence_cutoff'] is not None else .5))
        report = {'phase': args.phase, 'audit': data_audit, 'dataset': manifest,
            'suite_manifest_sha256': sha256(args.suite / 'freeze.json'),
            'model_manifest_sha256': sha256(args.run / 'model/manifest.json'),
            'frozen_sha256': sha256(args.run / 'frozen.json'), 'environment': environment(),
            'methods': {name: extended_metrics(y, p) for name, p in all_scores.items()},
            'selected_policies': selected, 'slices': slices(rows, y, all_scores),
            'marker_probe': marker_probe(rows, all_scores),
            'uncertainty': paired_group_bootstrap(y, scores['bert_finetuned'], scores['frozen_logistic'],
                                                 [r['group_id'] for r in rows])}
        records = [dict(id=r['id'], group_id=r['group_id'], target=y[i].tolist(),
            scores={name: p[i].tolist() for name, p in all_scores.items()}) for i, r in enumerate(rows)]
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'results.json', report)
    (output / 'README.md').write_text(report_markdown(report))
    if args.phase != 'historical':
        (output / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    if args.phase == 'development':
        for name, policy in policies.items():
            save_json(output / f'{name}-policy.json', policy)
        save_json(output / 'selection.json', {
            'suite_manifest_sha256': sha256(args.suite / 'freeze.json'),
            'frozen_sha256': sha256(args.run / 'frozen.json'),
            'sha256': {p.name: sha256(p) for p in sorted(output.glob('*-policy.json'))}})
    print(json.dumps({'phase': args.phase, 'output': str(output),
                      'macro_f1': {name: m['macro_f1'] for name, m in report['methods'].items()}}))


if __name__ == '__main__':
    main()
