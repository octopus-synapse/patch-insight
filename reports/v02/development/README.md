# Avaliação de desenvolvimento

Referência sintética revisada pela mesma IA. Sem revisão humana ou evidência populacional.

| Método | Macro-F1 | Exact match | Hamming ↓ | AP macro | Brier ↓ | Log loss ↓ |
|---|---:|---:|---:|---:|---:|---:|
| heuristic | 0.6797 | 0.4167 | 0.1889 | 0.7221 | 0.1889 | 3.0445 |
| frozen_logistic | 0.5967 | 0.3056 | 0.1889 | 0.8507 | 0.1664 | 0.9170 |
| bert_finetuned | 0.5507 | 0.2500 | 0.2167 | 0.7905 | 0.1642 | 0.5061 |
| frozen_logistic_temperature | 0.5967 | 0.3056 | 0.1889 | 0.8507 | 0.1384 | 0.4330 |
| bert_finetuned_temperature | 0.5507 | 0.2500 | 0.2167 | 0.7905 | 0.1641 | 0.5060 |

Métricas por rótulo, suportes, ROC-AUC, MCC, confiabilidade e risco/cobertura: `results.json`.
Scores heurísticos são decisões binárias, sem interpretação probabilística calibrada.
Log loss usa clipping em 1e-7. Esta análise não altera a seleção da v0.1.

| Política experimental | Temperatura | Cutoff | Cobertura | Erro nas decisões aceitas |
|---|---:|---:|---:|---:|
| frozen_logistic | 4.0000 | desativado | 1.0000 | 0.18888888888888888 |
| bert_finetuned | 0.9659 | desativado | 1.0000 | 0.21666666666666667 |

Política ajustada somente em desenvolvimento; desempenho em teste não é garantia de risco.
Nenhum cutoff cumpriu a meta de erro/cobertura em desenvolvimento. O erro na tabela é sem abstenção, não o risco de uma política aprovada.
