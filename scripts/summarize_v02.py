"""Build readable findings and demo comparisons from saved, immutable predictions."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ('action', 'context', 'method', 'result', 'measurement')
METRICS = {'macro_f1': 'max', 'micro_f1': 'max', 'exact_match': 'max', 'hamming_loss': 'min',
           'macro_average_precision': 'max', 'macro_roc_auc': 'max', 'macro_mcc': 'max',
           'mean_binary_brier': 'min', 'mean_binary_log_loss': 'min'}


def main():
    report_root = ROOT/'reports/v02'
    sources = {s: report_root/s/'results.json' for s in ('historical', 'test')}
    reports = {s: json.loads(p.read_text()) for s, p in sources.items()}
    comparison = {'source_sha256': {s: hashlib.sha256(p.read_bytes()).hexdigest() for s,p in sources.items()},
                  'directions': METRICS, 'datasets': {}}
    for split, report in reports.items():
        methods = {name: {metric: m[metric] for metric in METRICS} for name, m in report['methods'].items()}
        comparison['datasets'][split] = {'n': next(iter(report['methods'].values()))['n'], 'methods': methods,
            'reference': 'sintético; mesma IA autora e revisora',
            'ranking': {metric: sorted(methods, key=lambda name: methods[name][metric],
                        reverse=direction=='max') for metric, direction in METRICS.items()}}
    encoded = json.dumps(comparison, ensure_ascii=False, indent=2)+'\n'
    (report_root/'comparison.json').write_text(encoded)
    (ROOT/'src/patchinsight/web/comparison.json').write_text(encoded)
    report = reports['test']
    rows = {r['id']: r for r in map(json.loads, (ROOT/'data/v02/test.jsonl').read_text().splitlines())}
    predictions = list(map(json.loads, (report_root/'test/predictions.jsonl').read_text().splitlines()))
    lines = ['# Erros e interpretação da avaliação 0.2', '',
        'Análise posterior ao teste; não orientou correções, regras, limiares ou retreino nesta versão.', '',
        '| Método | Componente | FP | FN | Suporte positivo |', '|---|---|---:|---:|---:|']
    for name in ('heuristic', 'frozen_logistic', 'bert_finetuned'):
        for label, m in report['methods'][name]['per_label'].items():
            lines.append(f"| {name} | {label} | {m['fp']} | {m['fn']} | {m['support']} |")
    lines += ['', '## Exemplos de erros do BERT', '',
        'Amostra determinística: primeiros 12 textos com erro na ordem dos IDs. Referência revisada pela mesma IA, sem revisão humana.', '']
    count = 0
    for record in predictions:
        expected = record['target']
        predicted = [int(p >= .5) for p in record['scores']['bert_finetuned']]
        if expected != predicted:
            failures = [f"{k}: {'FP' if p else 'FN'}" for k,y,p in zip(LABELS,expected,predicted) if y!=p]
            row = rows[record['id']]
            lines += [f"- `{record['id']}` — {'; '.join(failures)}. {row['text']}",
                      f"  Interpretação de referência registrada antes dos scores: {row['review_note']}"]
            count += 1
            if count == 12:
                break
    lines += ['', '## Sensibilidade a marcadores', '',
        'Dois pares de origens, cada um com versão marcada e livre, preservam os rótulos. Isso é uma sondagem pequena, não uma estimativa populacional.', '',
        '| Método | Origem | Componentes que mudaram de decisão | Mudança absoluta média do score |',
        '|---|---|---:|---:|']
    for name in ('heuristic', 'frozen_logistic', 'bert_finetuned'):
        for pair in report['marker_probe'][name]:
            lines.append(f"| {name} | {pair['group']} | {pair['flipped_components']} | {pair['mean_absolute_score_change']:.4f} |")
    lines += ['', '## Leitura dos resultados', '',
        'O BERT e a regressão acertam action com frequência maior que a heurística, mas perdem cobertura em method e measurement. A vantagem global das heurísticas nesta amostra não significa que elas sejam superiores em qualquer domínio.', '',
        'Os pares sem marcadores mudam duas decisões por origem na regressão e três no BERT, embora os rótulos de referência permaneçam iguais. O comportamento é compatível com dependência de sinais de superfície; dois pares não isolam uma explicação causal.', '',
        'A temperatura da regressão reduziu Brier/log loss, mantendo F1. A temperatura do BERT alterou pouco os scores. Nenhum cutoff de confiança cumpriu a meta de desenvolvimento; as políticas salvam abstenção desativada. Não selecionar um cutoff pelos erros deste teste.', '',
        'Recortes podem não ter positivos em todos os rótulos: macro-F1 com zero_division=0 pode ser zero apesar de todas as decisões negativas estarem corretas. Usar suportes, Hamming e FP/FN na leitura desses recortes.', '',
        'As fontes de desenvolvimento e teste têm categorias e algumas construções paralelas; a autoria/revisão única e a pequena amostra limitam independência. Não são textos reais, validação humana, avaliação de spans ou benchmark de estruturas. O teste original continua preservado.']
    (report_root/'error-analysis.md').write_text('\n'.join(lines)+'\n')
    print('comparison.json, error-analysis.md e resumo da demo atualizados a partir dos relatórios salvos')


if __name__ == '__main__':
    main()
