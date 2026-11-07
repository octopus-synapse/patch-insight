"""Render authored evaluation sources, audit them, and freeze before scoring."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ('action', 'context', 'method', 'result', 'measurement')


def main():
    root = ROOT / 'data/v02'
    rows = []
    for split in ('development', 'test'):
        for index, line in enumerate((root / 'source' / f'{split}.tsv').read_text().splitlines(), 1):
            bits, tags, text, note = line.split('\t')
            assert len(bits) == 5 and set(bits) <= {'0', '1'}
            rows.append(dict(id=f'v02-{split}-{index:03}', group_id=f'v02-{split}-{index:03}',
                split=split, text=text, labels=dict(zip(LABELS, map(lambda b: b == '1', bits))),
                tags=tags.split(','), review_note=note, synthetic=True, human_reviewed=False,
                generator='assistant_unspecified', reviewer='same_assistant',
                review_method='semantic_self_review_before_predictions', guide_version='0.1.0',
                source_ref=f'source/{split}.tsv:{index}', dataset_version='0.2.0',
                license='CC-BY-4.0'))
        # Paired marker probes stay in one origin and are separate from free-form cases.
        topics = {
            'development': [
                ('Na operação de cobrança havia atrasos.', 'Comparei os lotes.', 'reconciliação manual', '6 lotes'),
                ('No agendamento de consultas havia conflitos.', 'Investiguei os horários.', 'entrevistas com operadores', '9 usuários')],
            'test': [
                ('No processamento de cupons havia falhas.', 'Inspecionei as regras.', 'simulação de entradas', '11 testes'),
                ('No controle de estoque havia divergências.', 'Auditei as movimentações.', 'checagem cruzada', '16 registros')],
        }[split]
        for index, (context, action, method, scope) in enumerate(topics, 1):
            for marked in (True, False):
                text = (f'{context} {action} Abordagem: {method}. Escopo: {scope}.' if marked else
                        f'{context} {action[:-1]} usando {method}, abrangendo {scope}.')
                rows.append(dict(id=f'v02-{split}-probe-{index}-{int(marked)}',
                    group_id=f'v02-{split}-probe-{index}', split=split, text=text,
                    labels=dict(zip(LABELS, [True, True, True, False, True])),
                    tags=['marker_probe', 'marked' if marked else 'unmarked'],
                    review_note='Par mantém cenário, ação, meio e volume; não afirma resultado.',
                    synthetic=True, human_reviewed=False, generator='assistant_unspecified',
                    reviewer='same_assistant', review_method='semantic_self_review_before_predictions',
                    guide_version='0.1.0', source_ref=f'build_v02_suite.py:{split}:{index}',
                    dataset_version='0.2.0', license='CC-BY-4.0'))
    assert len({r['text'] for r in rows}) == len(rows)
    originals = {json.loads(line)['text'] for line in (ROOT/'data/dataset.jsonl').read_text().splitlines()}
    assert not originals & {r['text'] for r in rows}
    payload = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
    frozen_files = {f'source/{s}.tsv': (root/'source'/f'{s}.tsv').read_bytes()
                    for s in ('development', 'test')}
    frozen_files['suite.jsonl'] = payload.encode()
    frozen_files['README.md'] = (root/'README.md').read_bytes()
    for split in ('development', 'test'):
        frozen_files[f'{split}.jsonl'] = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows if r['split'] == split).encode()
    frozen_files['../../scripts/build_v02_suite.py'] = Path(__file__).read_bytes()
    manifest = {'version': '0.2.0', 'reference': 'synthetic_same_ai_self_reviewed',
                'counts': {s: sum(r['split'] == s for r in rows) for s in ('development', 'test')},
                'sha256': {name: hashlib.sha256(value).hexdigest() for name, value in frozen_files.items()},
                'protocol': 'data/v02/README.md; development fits policy; test only reports'}
    freeze = root/'freeze.json'
    if freeze.exists() and json.loads(freeze.read_text()) != manifest:
        raise ValueError('suite já congelada; mudanças exigem nova versão e protocolo')
    for name, content in frozen_files.items():
        if name.endswith('.jsonl'):
            (root/name).write_bytes(content)
    freeze.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['counts']))


if __name__ == '__main__':
    main()
